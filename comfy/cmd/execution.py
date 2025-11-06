from __future__ import annotations

import asyncio
import copy
import heapq
import inspect
import json
import logging
import sys
import threading
import time
import traceback
import typing
from contextlib import nullcontext
from enum import Enum
from os import PathLike
from typing import List, Optional, Tuple, Literal

# order matters
from .main_pre import tracer
import torch
from opentelemetry.trace import get_current_span, StatusCode, Status

from comfy_api.internal import _ComfyNodeInternal, _NodeOutputInternal, first_real_override, is_class, \
    make_locked_method_func
from comfy_api.latest import io
from comfy_compatibility.vanilla import vanilla_environment_node_execution_hooks
from comfy_execution.caching import HierarchicalCache, LRUCache, CacheKeySetInputSignature, CacheKeySetID, \
    DependencyAwareCache, \
    BasicCache
from comfy_execution.graph import get_input_info, ExecutionList, DynamicPrompt, ExecutionBlocker
from comfy_execution.graph_types import FrozenTopologicalSort
from comfy_execution.graph_utils import is_link, GraphBuilder
from comfy_execution.progress import get_progress_state, reset_progress_state, add_progress_handler, \
    WebUIProgressHandler, \
    ProgressRegistry
from comfy_execution.utils import CurrentNodeContext
from comfy_execution.validation import validate_node_input
from .. import interruption
from .. import model_management
from ..component_model.abstract_prompt_queue import AbstractPromptQueue
from ..component_model.executor_types import ExecutorToClientProgress, ValidationTuple, ValidateInputsTuple, \
    ValidationErrorDict, NodeErrorsDictValue, ValidationErrorExtraInfoDict, FormattedValue, RecursiveExecutionTuple, \
    RecursiveExecutionErrorDetails, RecursiveExecutionErrorDetailsInterrupted, ExecutionResult, DuplicateNodeError, \
    HistoryResultDict, ExecutionErrorMessage, ExecutionInterruptedMessage, ComboOptions
from ..component_model.files import canonicalize_path
from ..component_model.module_property import create_module_properties
from ..component_model.queue_types import QueueTuple, HistoryEntry, QueueItem, MAXIMUM_HISTORY_SIZE, ExecutionStatus, \
    ExecutionStatusAsDict
from ..execution_context import context_execute_node, context_execute_prompt
from ..execution_context import current_execution_context, context_set_execution_list_and_inputs
from ..execution_ext import should_panic_on_exception
from ..node_requests_caching import use_requests_caching
from ..nodes.package_typing import InputTypeSpec, FloatSpecOptions, IntSpecOptions, CustomNode
from ..nodes_context import get_nodes

_module_properties = create_module_properties()
logger = logging.getLogger(__name__)


@_module_properties.getter
def _nodes():
    return get_nodes()


class IsChangedCache:
    def __init__(self, prompt_id: str, dynprompt: DynamicPrompt, outputs_cache: BasicCache):
        self.prompt_id = prompt_id
        self.dynprompt = dynprompt
        self.outputs_cache = outputs_cache
        self.is_changed = {}

    async def get(self, node_id):
        if node_id in self.is_changed:
            return self.is_changed[node_id]

        node = self.dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = get_nodes().NODE_CLASS_MAPPINGS[class_type]
        has_is_changed = False
        is_changed_name = None
        if issubclass(class_def, _ComfyNodeInternal) and first_real_override(class_def, "fingerprint_inputs") is not None:
            has_is_changed = True
            is_changed_name = "fingerprint_inputs"
        elif hasattr(class_def, "IS_CHANGED"):
            has_is_changed = True
            is_changed_name = "IS_CHANGED"
        if not has_is_changed:
            self.is_changed[node_id] = False
            return self.is_changed[node_id]

        if "is_changed" in node:
            self.is_changed[node_id] = node["is_changed"]
            return self.is_changed[node_id]

        # Intentionally do not use cached outputs here. We only want constants in IS_CHANGED
        input_data_all, _, hidden_inputs = get_input_data(node["inputs"], class_def, node_id, None)
        try:
            is_changed = await _async_map_node_over_list(self.prompt_id, node_id, class_def, input_data_all, is_changed_name)
            is_changed = await resolve_map_node_over_list_results(is_changed)
            node["is_changed"] = [None if isinstance(x, ExecutionBlocker) else x for x in is_changed]
        except:
            node["is_changed"] = float("NaN")
        finally:
            self.is_changed[node_id] = node["is_changed"]
        return self.is_changed[node_id]


class CacheType(Enum):
    CLASSIC = 0
    LRU = 1
    DEPENDENCY_AWARE = 2


class CacheSet:
    def __init__(self, cache_type=None, cache_size=None):
        if cache_type == CacheType.DEPENDENCY_AWARE:
            self.init_dependency_aware_cache()
            logger.info("Disabling intermediate node cache.")
        elif cache_type == CacheType.LRU:
            if cache_size is None:
                cache_size = 0
            self.init_lru_cache(cache_size)
            logger.info("Using LRU cache")
        else:
            self.init_classic_cache()

        self.all = [self.outputs, self.ui, self.objects]

    # Performs like the old cache -- dump data ASAP
    def init_classic_cache(self):
        self.outputs = HierarchicalCache(CacheKeySetInputSignature)
        self.ui = HierarchicalCache(CacheKeySetInputSignature)
        self.objects = HierarchicalCache(CacheKeySetID)

    def init_lru_cache(self, cache_size):
        self.outputs = LRUCache(CacheKeySetInputSignature, max_size=cache_size)
        self.ui = LRUCache(CacheKeySetInputSignature, max_size=cache_size)
        self.objects = HierarchicalCache(CacheKeySetID)

    # only hold cached items while the decendents have not executed
    def init_dependency_aware_cache(self):
        self.outputs = DependencyAwareCache(CacheKeySetInputSignature)
        self.ui = DependencyAwareCache(CacheKeySetInputSignature)
        self.objects = DependencyAwareCache(CacheKeySetID)

    def recursive_debug_dump(self):
        result = {
            "outputs": self.outputs.recursive_debug_dump(),
            "ui": self.ui.recursive_debug_dump(),
        }
        return result


SENSITIVE_EXTRA_DATA_KEYS = ("auth_token_comfy_org", "api_key_comfy_org")


def get_input_data(inputs, class_def, unique_id, outputs=None, dynprompt=None, extra_data=None):
    if extra_data is None:
        extra_data = {}
    is_v3 = issubclass(class_def, _ComfyNodeInternal)
    if is_v3:
        valid_inputs, schema = class_def.INPUT_TYPES(include_hidden=False, return_schema=True)
    else:
        valid_inputs = class_def.INPUT_TYPES()
    input_data_all = {}
    missing_keys = {}
    hidden_inputs_v3 = {}
    for x in inputs:
        input_data = inputs[x]
        _, input_category, input_info = get_input_info(class_def, x, valid_inputs)

        def mark_missing():
            missing_keys[x] = True
            input_data_all[x] = (None,)

        if is_link(input_data) and (not input_info or not input_info.get("rawLink", False)):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if outputs is None:
                mark_missing()
                continue  # This might be a lazily-evaluated input
            cached_output = outputs.get(input_unique_id)
            if cached_output is None:
                mark_missing()
                continue
            if output_index >= len(cached_output):
                mark_missing()
                continue
            obj = cached_output[output_index]
            input_data_all[x] = obj
        elif input_category is not None:
            input_data_all[x] = [input_data]

    # todo: this should be retrieved from the execution context
    if is_v3:
        if schema.hidden:
            if io.Hidden.prompt in schema.hidden:
                hidden_inputs_v3[io.Hidden.prompt] = dynprompt.get_original_prompt() if dynprompt is not None else {}
            if io.Hidden.dynprompt in schema.hidden:
                hidden_inputs_v3[io.Hidden.dynprompt] = dynprompt
            if io.Hidden.extra_pnginfo in schema.hidden:
                hidden_inputs_v3[io.Hidden.extra_pnginfo] = extra_data.get('extra_pnginfo', None)
            if io.Hidden.unique_id in schema.hidden:
                hidden_inputs_v3[io.Hidden.unique_id] = unique_id
            if io.Hidden.auth_token_comfy_org in schema.hidden:
                hidden_inputs_v3[io.Hidden.auth_token_comfy_org] = extra_data.get("auth_token_comfy_org", None)
            if io.Hidden.api_key_comfy_org in schema.hidden:
                hidden_inputs_v3[io.Hidden.api_key_comfy_org] = extra_data.get("api_key_comfy_org", None)
    else:
        if "hidden" in valid_inputs:
            h = valid_inputs["hidden"]
            for x in h:
                if h[x] == "PROMPT":
                    input_data_all[x] = [dynprompt.get_original_prompt() if dynprompt is not None else {}]
                if h[x] == "DYNPROMPT":
                    input_data_all[x] = [dynprompt]
                if h[x] == "EXTRA_PNGINFO":
                    input_data_all[x] = [extra_data.get('extra_pnginfo', None)]
                if h[x] == "UNIQUE_ID":
                    input_data_all[x] = [unique_id]
                if h[x] == "AUTH_TOKEN_COMFY_ORG":
                    input_data_all[x] = [extra_data.get("auth_token_comfy_org", None)]
                if h[x] == "API_KEY_COMFY_ORG":
                    input_data_all[x] = [extra_data.get("api_key_comfy_org", None)]
    return input_data_all, missing_keys, hidden_inputs_v3


def map_node_over_list(obj, input_data_all: typing.Dict[str, typing.Any], func: str, allow_interrupt=False, execution_block_cb=None, pre_execute_cb=None):
    raise ValueError("")


async def resolve_map_node_over_list_results(results):
    remaining = [x for x in results if isinstance(x, asyncio.Task) and not x.done()]
    if len(remaining) == 0:
        return [x.result() if isinstance(x, asyncio.Task) else x for x in results]
    else:
        done, pending = await asyncio.wait(remaining)
        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc
        return [x.result() if isinstance(x, asyncio.Task) else x for x in results]


@tracer.start_as_current_span("Execute Node")
async def _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, func, allow_interrupt=False, execution_block_cb=None, pre_execute_cb=None, hidden_inputs=None, execution_list=None, executed=None):
    with context_set_execution_list_and_inputs(FrozenTopologicalSort.from_topological_sort(execution_list) if execution_list is not None else None, frozenset(executed) if executed is not None else None):
        return await __async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, func, allow_interrupt, execution_block_cb, pre_execute_cb, hidden_inputs)


async def __async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, func, allow_interrupt=False, execution_block_cb=None, pre_execute_cb=None, hidden_inputs=None):
    span = get_current_span()
    class_type = obj.__class__.__name__
    span.set_attribute("class_type", class_type)
    if input_data_all is not None:
        for kwarg_name, kwarg_value in input_data_all.items():
            if isinstance(kwarg_value, str) or isinstance(kwarg_value, bool) or isinstance(kwarg_value, int) or isinstance(kwarg_value, float):
                span.set_attribute(f"input_data_all.{kwarg_name}", kwarg_value)
            else:
                try:
                    items_to_display = []
                    if hasattr(kwarg_value, "shape"):
                        # if the object has a shape attribute (likely a NumPy array or similar), get up to the first ten elements
                        flat_values = kwarg_value.flatten() if hasattr(kwarg_value, "flatten") else kwarg_value
                        items_to_display = [flat_values[i] for i in range(min(10, flat_values.size))]
                    elif hasattr(kwarg_value, "__getitem__") and hasattr(kwarg_value, "__len__"):
                        # If the object is indexable and has a length, get the first ten items
                        items_to_display = [kwarg_value[i] for i in range(min(10, len(kwarg_value)))]

                    filtered_items = [
                        item for item in items_to_display if isinstance(item, (str, bool, int, float))
                    ]

                    if filtered_items:
                        span.set_attribute(f"input_data_all.{kwarg_name}", filtered_items)
                except TypeError:
                    pass
    # check if node wants the lists
    input_is_list = getattr(obj, "INPUT_IS_LIST", False)

    if len(input_data_all) == 0:
        max_len_input = 0
    else:
        max_len_input = max(len(x) for x in input_data_all.values())

    # get a slice of inputs, repeat last input when list isn't long enough
    def slice_dict(d, i):
        return {k: v[i if len(v) > i else -1] for k, v in d.items()}

    results = []

    async def process_inputs(inputs, index=None, input_is_list=False):
        if allow_interrupt:
            interruption.throw_exception_if_processing_interrupted()
        execution_block = None
        for k, v in inputs.items():
            if input_is_list:
                for e in v:
                    if isinstance(e, ExecutionBlocker):
                        v = e
                        break
            if isinstance(v, ExecutionBlocker):
                execution_block = execution_block_cb(v) if execution_block_cb else v
                break
        if execution_block is None:
            if pre_execute_cb is not None and index is not None:
                pre_execute_cb(index)
            # V3
            if isinstance(obj, _ComfyNodeInternal) or (is_class(obj) and issubclass(obj, _ComfyNodeInternal)):
                # if is just a class, then assign no resources or state, just create clone
                if is_class(obj):
                    type_obj = obj
                    obj.VALIDATE_CLASS()
                    class_clone = obj.PREPARE_CLASS_CLONE(hidden_inputs)
                # otherwise, use class instance to populate/reuse some fields
                else:
                    type_obj = type(obj)
                    type_obj.VALIDATE_CLASS()
                    class_clone = type_obj.PREPARE_CLASS_CLONE(hidden_inputs)
                f = make_locked_method_func(type_obj, func, class_clone)
            # V1
            else:
                f = getattr(obj, func)
            if inspect.iscoroutinefunction(f):
                async def async_wrapper(f, prompt_id, unique_id, list_index, args):
                    # todo: this is redundant with other parts of the hiddenswitch fork, but we've shimmed it for compatibility
                    with CurrentNodeContext(prompt_id, unique_id, list_index):
                        return await f(**args)

                task = asyncio.create_task(async_wrapper(f, prompt_id, unique_id, index, args=inputs))
                # Give the task a chance to execute without yielding
                await asyncio.sleep(0)
                if task.done():
                    result = task.result()
                    results.append(result)
                else:
                    results.append(task)
            else:
                with CurrentNodeContext(prompt_id, unique_id, index):
                    result = f(**inputs)
                results.append(result)
        else:
            results.append(execution_block)

    if input_is_list:
        await process_inputs(input_data_all, 0, input_is_list=input_is_list)
    elif max_len_input == 0:
        await process_inputs({})
    else:
        for i in range(max_len_input):
            input_dict = slice_dict(input_data_all, i)
            await process_inputs(input_dict, i)
    return results


def merge_result_data(results, obj):
    # check which outputs need concatenating
    output = []
    output_is_list = [False] * len(results[0])
    if hasattr(obj, "OUTPUT_IS_LIST"):
        output_is_list = obj.OUTPUT_IS_LIST

    # merge node execution results
    for i, is_list in zip(range(len(results[0])), output_is_list):
        if is_list:
            value = []
            for o in results:
                if isinstance(o[i], ExecutionBlocker):
                    value.append(o[i])
                else:
                    value.extend(o[i])
            output.append(value)
        else:
            output.append([o[i] for o in results])
    return output


async def get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=None, pre_execute_cb=None, hidden_inputs=None, inputs=None, execution_list=None, executed=None):
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, hidden_inputs=hidden_inputs, execution_list=execution_list, executed=executed)
    has_pending_task = any(isinstance(r, asyncio.Task) and not r.done() for r in return_values)
    if has_pending_task:
        return return_values, {}, False, has_pending_task
    output, ui, has_subgraph = get_output_from_returns(return_values, obj)
    return output, ui, has_subgraph, False


def get_output_from_returns(return_values, obj):
    results = []
    uis = []
    subgraph_results = []
    has_subgraph = False
    for i in range(len(return_values)):
        r = return_values[i]
        if isinstance(r, dict):
            if 'ui' in r:
                uis.append(r['ui'])
            if 'expand' in r:
                # Perform an expansion, but do not append results
                has_subgraph = True
                new_graph = r['expand']
                result = r.get("result", None)
                if isinstance(result, ExecutionBlocker):
                    result = tuple([result] * len(obj.RETURN_TYPES))
                subgraph_results.append((new_graph, result))
            elif 'result' in r:
                result = r.get("result", None)
                if isinstance(result, ExecutionBlocker):
                    result = tuple([result] * len(obj.RETURN_TYPES))
                results.append(result)
                subgraph_results.append((None, result))
        elif isinstance(r, _NodeOutputInternal):
            # V3
            if r.ui is not None:
                if isinstance(r.ui, dict):
                    uis.append(r.ui)
                else:
                    uis.append(r.ui.as_dict())
            if r.expand is not None:
                has_subgraph = True
                new_graph = r.expand
                result = r.result
                if r.block_execution is not None:
                    result = tuple([ExecutionBlocker(r.block_execution)] * len(obj.RETURN_TYPES))
                subgraph_results.append((new_graph, result))
            elif r.result is not None:
                result = r.result
                if r.block_execution is not None:
                    result = tuple([ExecutionBlocker(r.block_execution)] * len(obj.RETURN_TYPES))
                results.append(result)
                subgraph_results.append((None, result))
        else:
            if isinstance(r, ExecutionBlocker):
                r = tuple([r] * len(obj.RETURN_TYPES))
            results.append(r)
            subgraph_results.append((None, r))

    if has_subgraph:
        output = subgraph_results
    elif len(results) > 0:
        output = merge_result_data(results, obj)
    else:
        output = []
    ui = dict()
    # TODO: Think there's an existing bug here
    # If we're performing a subgraph expansion, we probably shouldn't be returning UI values yet.
    # They'll get cached without the completed subgraphs. It's an edge case and I'm not aware of
    # any nodes that use both subgraph expansion and custom UI outputs, but might be a problem in the future.
    if len(uis) > 0:
        ui = {k: [y for x in uis for y in x[k]] for k in uis[0].keys()}
    return output, ui, has_subgraph


def format_value(x) -> FormattedValue:
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    elif isinstance(x, dict) and not any(isinstance(v, torch.Tensor) for v in x.values()):
        return str(x)
    else:
        return str(x.__class__)


async def execute(server: ExecutorToClientProgress, dynprompt: DynamicPrompt, caches, node_id: str, extra_data: dict, executed, prompt_id, execution_list, pending_subgraph_results, pending_async_nodes) -> RecursiveExecutionTuple:
    """
    Executes a prompt
    :param server:
    :param dynprompt:
    :param caches:
    :param node_id: the node id
    :param extra_data:
    :param executed:
    :param prompt_id:
    :param execution_list:
    :param pending_subgraph_results:
    :return:
    """
    with (
        context_execute_node(node_id),
        vanilla_environment_node_execution_hooks(),
        use_requests_caching(),
    ):
        return await _execute(server, dynprompt, caches, node_id, extra_data, executed, prompt_id, execution_list, pending_subgraph_results, pending_async_nodes)


async def _execute(server, dynprompt: DynamicPrompt, caches: CacheSet, current_item: str, extra_data, executed, prompt_id, execution_list: ExecutionList, pending_subgraph_results, pending_async_nodes) -> RecursiveExecutionTuple:
    unique_id = current_item
    real_node_id = dynprompt.get_real_node_id(unique_id)
    display_node_id = dynprompt.get_display_node_id(unique_id)
    parent_node_id = dynprompt.get_parent_node_id(unique_id)
    inputs = dynprompt.get_node(unique_id)['inputs']
    class_type = dynprompt.get_node(unique_id)['class_type']
    class_def = get_nodes().NODE_CLASS_MAPPINGS[class_type]
    if caches.outputs.get(unique_id) is not None:
        if server.client_id is not None:
            cached_output = caches.ui.get(unique_id) or {}
            server.send_sync("executed", {"node": unique_id, "display_node": display_node_id, "output": cached_output.get("output", None), "prompt_id": prompt_id}, server.client_id)
        get_progress_state().finish_progress(unique_id)
        return RecursiveExecutionTuple(ExecutionResult.SUCCESS, None, None)

    input_data_all = None
    try:
        if unique_id in pending_async_nodes:
            results = []
            for r in pending_async_nodes[unique_id]:
                if isinstance(r, asyncio.Task):
                    try:
                        results.append(r.result())
                    except Exception as ex:
                        # An async task failed - propagate the exception up
                        del pending_async_nodes[unique_id]
                        raise ex
                else:
                    results.append(r)
            del pending_async_nodes[unique_id]
            output_data, output_ui, has_subgraph = get_output_from_returns(results, class_def)
        elif unique_id in pending_subgraph_results:
            cached_results = pending_subgraph_results[unique_id]
            resolved_outputs = []
            for is_subgraph, result in cached_results:
                if not is_subgraph:
                    resolved_outputs.append(result)
                else:
                    resolved_output = []
                    for r in result:
                        if is_link(r):
                            source_node, source_output = r[0], r[1]
                            node_output = caches.outputs.get(source_node)[source_output]
                            for o in node_output:
                                resolved_output.append(o)

                        else:
                            resolved_output.append(r)
                    resolved_outputs.append(tuple(resolved_output))
            output_data = merge_result_data(resolved_outputs, class_def)
            output_ui = []
            has_subgraph = False
        else:
            get_progress_state().start_progress(unique_id)
            input_data_all, missing_keys, hidden_inputs = get_input_data(inputs, class_def, unique_id, caches.outputs, dynprompt, extra_data)
            if server.client_id is not None:
                server.last_node_id = display_node_id
                server.send_sync("executing", {"node": unique_id, "display_node": display_node_id, "prompt_id": prompt_id}, server.client_id)

            obj = caches.objects.get(unique_id)
            if obj is None:
                obj = class_def()
                caches.objects.set(unique_id, obj)

            if issubclass(class_def, _ComfyNodeInternal):
                lazy_status_present = first_real_override(class_def, "check_lazy_status") is not None
            else:
                lazy_status_present = getattr(obj, "check_lazy_status", None) is not None
            if lazy_status_present:
                required_inputs = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, "check_lazy_status", allow_interrupt=True, hidden_inputs=hidden_inputs, execution_list=execution_list, executed=executed)
                required_inputs = await resolve_map_node_over_list_results(required_inputs)
                required_inputs = set(sum([r for r in required_inputs if isinstance(r, list)], []))
                required_inputs = [x for x in required_inputs if isinstance(x, str) and (
                        x not in input_data_all or x in missing_keys
                )]
                if len(required_inputs) > 0:
                    for i in required_inputs:
                        execution_list.make_input_strong_link(unique_id, i)
                    return RecursiveExecutionTuple(ExecutionResult.PENDING, None, None)

            def execution_block_cb(block):
                if block.message is not None:
                    mes: ExecutionErrorMessage = {
                        "prompt_id": prompt_id,
                        "node_id": unique_id,
                        "node_type": class_type,
                        "executed": list(executed),

                        "exception_message": f"Execution Blocked: {block.message}",
                        "exception_type": "ExecutionBlocked",
                        "traceback": [],
                        "current_inputs": [],
                        "current_outputs": [],
                    }
                    server.send_sync("execution_error", mes, server.client_id)
                    return ExecutionBlocker(None)
                else:
                    return block

            def pre_execute_cb(call_index):
                # TODO - How to handle this with async functions without contextvars (which requires Python 3.12)?
                GraphBuilder.set_default_prefix(unique_id, call_index, 0)

            output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, hidden_inputs=hidden_inputs, inputs=inputs, execution_list=execution_list, executed=executed)
            if has_pending_tasks:
                pending_async_nodes[unique_id] = output_data
                unblock = execution_list.add_external_block(unique_id)

                async def await_completion():
                    tasks = [x for x in output_data if isinstance(x, asyncio.Task)]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    unblock()

                asyncio.create_task(await_completion())
                return RecursiveExecutionTuple(ExecutionResult.PENDING, None, None)
        if len(output_ui) > 0:
            caches.ui.set(unique_id, {
                "meta": {
                    "node_id": unique_id,
                    "display_node": display_node_id,
                    "parent_node": parent_node_id,
                    "real_node_id": real_node_id,
                },
                "output": output_ui
            })
            if server.client_id is not None:
                server.send_sync("executed", {"node": unique_id, "display_node": display_node_id, "output": output_ui, "prompt_id": prompt_id},
                                 server.client_id)
        if has_subgraph:
            cached_outputs = []
            new_node_ids = []
            new_output_ids = []
            new_output_links = []
            for i in range(len(output_data)):
                new_graph, node_outputs = output_data[i]
                if new_graph is None:
                    cached_outputs.append((False, node_outputs))
                else:
                    # Check for conflicts
                    for node_id in new_graph.keys():
                        if dynprompt.has_node(node_id):
                            raise DuplicateNodeError(f"Attempt to add duplicate node {node_id}. Ensure node ids are unique and deterministic or use graph_utils.GraphBuilder.")
                    for node_id, node_info in new_graph.items():
                        new_node_ids.append(node_id)
                        display_id = node_info.get("override_display_id", unique_id)
                        dynprompt.add_ephemeral_node(node_id, node_info, unique_id, display_id)
                        # Figure out if the newly created node is an output node
                        class_type = node_info["class_type"]
                        class_def = get_nodes().NODE_CLASS_MAPPINGS[class_type]
                        if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                            new_output_ids.append(node_id)
                    for i in range(len(node_outputs)):
                        if is_link(node_outputs[i]):
                            from_node_id, from_socket = node_outputs[i][0], node_outputs[i][1]
                            new_output_links.append((from_node_id, from_socket))
                    cached_outputs.append((True, node_outputs))
            new_node_ids = set(new_node_ids)
            for cache in caches.all:
                subcache = await cache.ensure_subcache_for(unique_id, new_node_ids)
                subcache.clean_unused()
            for node_id in new_output_ids:
                execution_list.add_node(node_id)
            for link in new_output_links:
                execution_list.add_strong_link(link[0], link[1], unique_id)
            pending_subgraph_results[unique_id] = cached_outputs
            return RecursiveExecutionTuple(ExecutionResult.PENDING, None, None)
        caches.outputs.set(unique_id, output_data)
    except interruption.InterruptProcessingException as iex:
        logger.info("Processing interrupted")

        # skip formatting inputs/outputs
        error_details: RecursiveExecutionErrorDetailsInterrupted = {
            "node_id": real_node_id,
        }

        return RecursiveExecutionTuple(ExecutionResult.FAILURE, error_details, iex)
    except Exception as ex:
        typ, _, tb = sys.exc_info()
        exception_type = full_type_name(typ)
        input_data_formatted = {}
        if input_data_all is not None:
            input_data_formatted = {}
            for name, inputs in input_data_all.items():
                input_data_formatted[name] = [format_value(x) for x in inputs]

        logger.error("An error occurred while executing a workflow", exc_info=ex)
        logger.error(traceback.format_exc())
        tips = ""

        if isinstance(ex, model_management.OOM_EXCEPTION):
            tips = "This error means you ran out of memory on your GPU.\n\nTIPS: If the workflow worked before you might have accidentally set the batch_size to a large number."
            logger.error("Got an OOM, unloading all loaded models.")
            model_management.unload_all_models()

        error_details: RecursiveExecutionErrorDetails = {
            "node_id": real_node_id,
            "exception_message": "{}\n{}".format(ex, tips),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted
        }

        if should_panic_on_exception(ex, current_execution_context().configuration.panic_when):
            logger.error(f"The exception {ex} was configured as unrecoverable, scheduling an exit")

            def sys_exit(*args):
                sys.exit(1)

            asyncio.get_event_loop().call_soon_threadsafe(sys_exit, ())

        return RecursiveExecutionTuple(ExecutionResult.FAILURE, error_details, ex)

    get_progress_state().finish_progress(unique_id)
    executed.add(unique_id)

    return RecursiveExecutionTuple(ExecutionResult.SUCCESS, None, None)


class PromptExecutor:
    def __init__(self, server: ExecutorToClientProgress, cache_type: CacheType | Literal[False] = False, cache_size: int | None = None):
        self.success = None
        self.cache_size = cache_size
        self.cache_type = cache_type
        self.server = server
        self.raise_exceptions = False
        self.reset()
        self.history_result: HistoryResultDict | None = None

    def reset(self):
        self.success = True
        self.caches = CacheSet(cache_type=self.cache_type, cache_size=self.cache_size)
        self.status_messages = []

    def add_message(self, event, data: dict, broadcast: bool):
        data = {
            **data,
            # todo: use a real time library
            "timestamp": int(time.time() * 1000),
        }
        self.status_messages.append((event, data))
        if self.server.client_id is not None or broadcast:
            self.server.send_sync(event, data, self.server.client_id)

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        current_span = get_current_span()
        current_span.set_status(Status(StatusCode.ERROR))
        current_span.record_exception(ex)
        try:
            encoded_prompt = json.dumps(prompt)
            current_span.set_attribute("prompt", encoded_prompt)
        except Exception as exc_info:
            pass

        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]

        # First, send back the status to the frontend depending
        # on the exception type
        if isinstance(ex, interruption.InterruptProcessingException):
            mes: ExecutionInterruptedMessage = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
            }
            self.add_message("execution_interrupted", mes, broadcast=True)
        else:
            mes: ExecutionErrorMessage = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
                "exception_message": error["exception_message"],
                "exception_type": error["exception_type"],
                "traceback": error["traceback"],
                "current_inputs": error["current_inputs"],
                "current_outputs": list(current_outputs),
            }
            self.add_message("execution_error", mes, broadcast=False)

        if ex is not None and self.raise_exceptions:
            raise ex

    def execute(self, prompt, prompt_id, extra_data=None, execute_outputs=None):
        if execute_outputs is None:
            execute_outputs = []
        if extra_data is None:
            extra_data = {}
        asyncio.run(self.execute_async(prompt, prompt_id, extra_data, execute_outputs))

    async def execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        # torchao and potentially other optimization approaches break when the models are created in inference mode
        # todo: this should really be backpropagated to code which creates ModelPatchers via lazy evaluation rather than globally checked here
        inference_mode = all(not hasattr(node_class, "INFERENCE_MODE") or node_class.INFERENCE_MODE for node_class in iterate_obj_classes(prompt))
        dynamic_prompt = DynamicPrompt(prompt)
        reset_progress_state(prompt_id, dynamic_prompt)
        with context_execute_prompt(self.server, prompt_id, progress_registry=ProgressRegistry(prompt_id, dynamic_prompt), inference_mode=inference_mode):
            await self._execute_async(dynamic_prompt, prompt_id, extra_data, execute_outputs)

    async def _execute_async(self, prompt: DynamicPrompt, prompt_id, extra_data=None, execute_outputs: list[str] = None, inference_mode: bool = True):
        if execute_outputs is None:
            execute_outputs = []
        if extra_data is None:
            extra_data = {}

        interruption.interrupt_current_processing(False)

        if "client_id" in extra_data:
            self.server.client_id = extra_data["client_id"]
        else:
            self.server.client_id = None

        self.status_messages = []
        self.add_message("execution_start", {"prompt_id": prompt_id}, broadcast=False)

        with torch.inference_mode() if inference_mode else nullcontext():
            dynamic_prompt = prompt
            prompt: dict = prompt.original_prompt
            add_progress_handler(WebUIProgressHandler(self.server))
            is_changed_cache = IsChangedCache(prompt_id, dynamic_prompt, self.caches.outputs)
            for cache in self.caches.all:
                await cache.set_prompt(dynamic_prompt, prompt.keys(), is_changed_cache)
                cache.clean_unused()

            cached_nodes = []
            for node_id in prompt:
                if self.caches.outputs.get(node_id) is not None:
                    cached_nodes.append(node_id)

            model_management.cleanup_models_gc()
            self.add_message("execution_cached",
                             {"nodes": cached_nodes, "prompt_id": prompt_id},
                             broadcast=False)
            pending_subgraph_results = {}
            pending_async_nodes = {}  # TODO - Unify this with pending_subgraph_results
            executed = set()
            execution_list = ExecutionList(dynamic_prompt, self.caches.outputs)
            current_outputs = self.caches.outputs.all_node_ids()
            for node_id in list(execute_outputs):
                execution_list.add_node(node_id)

            while not execution_list.is_empty():
                node_id, error, ex = await execution_list.stage_node_execution()
                node_id: str
                if error is not None:
                    self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
                    break

                assert node_id is not None, "Node ID should not be None at this point"
                result, error, ex = await execute(self.server, dynamic_prompt, self.caches, node_id, extra_data, executed, prompt_id, execution_list, pending_subgraph_results, pending_async_nodes)
                self.success = result != ExecutionResult.FAILURE
                if result == ExecutionResult.FAILURE:
                    self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
                    break
                elif result == ExecutionResult.PENDING:
                    execution_list.unstage_node_execution()
                else:  # result == ExecutionResult.SUCCESS:
                    execution_list.complete_node_execution()
            else:
                # Only execute when the while-loop ends without break
                self.add_message("execution_success", {"prompt_id": prompt_id}, broadcast=False)

            ui_outputs = {}
            meta_outputs = {}
            all_node_ids = self.caches.ui.all_node_ids()
            for node_id in all_node_ids:
                ui_info = self.caches.ui.get(node_id)
                if ui_info is not None:
                    ui_outputs[node_id] = ui_info["output"]
                    meta_outputs[node_id] = ui_info["meta"]
            self.history_result = {
                "outputs": ui_outputs,
                "meta": meta_outputs,
            }
            self.server.last_node_id = None
            if model_management.DISABLE_SMART_MEMORY:
                model_management.unload_all_models()

    @property
    def outputs_ui(self) -> dict | None:
        return self.history_result["outputs"] if self.history_result is not None else None


def iterate_obj_classes(prompt: dict[str, typing.Any]) -> typing.Generator[typing.Type[CustomNode], None, None]:
    for _, node in prompt.items():
        yield get_nodes().NODE_CLASS_MAPPINGS[node['class_type']]


async def validate_inputs(prompt_id: typing.Any, prompt, item, validated: typing.Dict[str, ValidateInputsTuple]) -> ValidateInputsTuple:
    # todo: this should check if LoadImage / LoadImageMask paths exist
    # todo: or, nodes should provide a way to validate their values
    unique_id = item
    if unique_id in validated:
        return validated[unique_id]

    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    obj_class = get_nodes().NODE_CLASS_MAPPINGS[class_type]

    class_inputs = obj_class.INPUT_TYPES()
    valid_inputs = set(class_inputs.get('required', {})).union(set(class_inputs.get('optional', {})))

    error: ValidationErrorDict
    errors = []
    valid = True

    validate_function_inputs = []
    validate_has_kwargs = False
    if issubclass(obj_class, _ComfyNodeInternal):
        validate_function_name = "validate_inputs"
        validate_function = first_real_override(obj_class, validate_function_name)
    else:
        validate_function_name = "VALIDATE_INPUTS"
        validate_function = getattr(obj_class, validate_function_name, None)
    if validate_function is not None:
        argspec = inspect.getfullargspec(validate_function)
        validate_function_inputs = argspec.args
        validate_has_kwargs = argspec.varkw is not None
    received_types = {}

    for x in valid_inputs:
        input_type, input_category, extra_info = get_input_info(obj_class, x, class_inputs)
        assert extra_info is not None
        if x not in inputs:
            if input_category == "required":
                error = {
                    "type": "required_input_missing",
                    "message": "Required input is missing",
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x
                    }
                }
                errors.append(error)
            continue

        val = inputs[x]
        info: InputTypeSpec = (input_type, extra_info)
        if isinstance(val, list):
            if len(val) != 2:
                error = {
                    "type": "bad_linked_input",
                    "message": "Bad linked input, must be a length-2 list of [node_id, slot_index]",
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val
                    }
                }
                errors.append(error)
                continue

            o_id = val[0]
            o_class_type = prompt[o_id]['class_type']
            r = get_nodes().NODE_CLASS_MAPPINGS[o_class_type].RETURN_TYPES
            received_type = r[val[1]]
            received_types[x] = received_type
            any_enum = received_type == [] and (isinstance(input_type, list) or isinstance(input_type, tuple))

            if 'input_types' not in validate_function_inputs and not validate_node_input(received_type, input_type) and not any_enum:
                details = f"{x}, {received_type} != {input_type}"
                error = {
                    "type": "return_type_mismatch",
                    "message": "Return type mismatch between linked nodes",
                    "details": details,
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_type": received_type,
                        "linked_node": val
                    }
                }
                errors.append(error)
                continue
            try:
                r2 = await validate_inputs(prompt_id, prompt, o_id, validated)
                if r2[0] is False:
                    # `r` will be set in `validated[o_id]` already
                    valid = False
                    continue
            except Exception as ex:
                typ, _, tb = sys.exc_info()
                valid = False
                exception_type = full_type_name(typ)
                reasons = [{
                    "type": "exception_during_inner_validation",
                    "message": "Exception when validating inner node",
                    "details": str(ex),
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "exception_message": str(ex),
                        "exception_type": exception_type,
                        "traceback": traceback.format_tb(tb),
                        "linked_node": val
                    }
                }]
                validated[o_id] = ValidateInputsTuple(False, reasons, o_id)
                continue
        else:
            try:
                # Unwraps values wrapped in __value__ key. This is used to pass
                # list widget value to execution, as by default list value is
                # reserved to represent the connection between nodes.
                if isinstance(val, dict) and "__value__" in val:
                    val = val["__value__"]
                    inputs[x] = val

                if input_type == "INT":
                    val = int(val)
                    inputs[x] = val
                if input_type == "FLOAT":
                    val = float(val)
                    inputs[x] = val
                if input_type == "STRING":
                    val = str(val)
                    inputs[x] = val
                if input_type == "BOOLEAN":
                    val = bool(val)
                    inputs[x] = val
            except Exception as ex:
                error = {
                    "type": "invalid_input_type",
                    "message": f"Failed to convert an input value to a {input_type} value",
                    "details": f"{x}, {val}, {ex}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val,
                        "exception_message": str(ex)
                    }
                }
                errors.append(error)
                continue

            if x not in validate_function_inputs:
                has_min_max: IntSpecOptions | FloatSpecOptions = info[1]
                if "min" in has_min_max and val < has_min_max["min"]:
                    error = {
                        "type": "value_smaller_than_min",
                        "message": "Value {} smaller than min of {}".format(val, has_min_max["min"]),
                        "details": f"{x}",
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue
                if "max" in has_min_max and val > has_min_max["max"]:
                    error = {
                        "type": "value_bigger_than_max",
                        "message": "Value {} bigger than max of {}".format(val, has_min_max["max"]),
                        "details": f"{x}",
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue

                if isinstance(input_type, ComboOptions) or hasattr(input_type, "view_for_validation"):
                    input_type = input_type.view_for_validation()

                if isinstance(input_type, list):
                    combo_options = input_type
                    if isinstance(val, str) and "\\" in val:
                        # try to normalize paths for comparison purposes
                        val = canonicalize_path(val)
                    if all(isinstance(item, (str, PathLike)) for item in combo_options):
                        combo_options = [canonicalize_path(item) for item in combo_options]
                    if val not in combo_options:
                        input_config = info
                        list_info = ""

                        # Don't send back gigantic lists like if they're lots of
                        # scanned model filepaths
                        if len(combo_options) > 20:
                            list_info = f"(list of length {len(combo_options)})"
                            input_config = None
                        else:
                            list_info = str(combo_options)

                        error = {
                            "type": "value_not_in_list",
                            "message": "Value not in list",
                            "details": f"{x}: '{val}' not in {list_info}",
                            "extra_info": {
                                "input_name": x,
                                "input_config": input_config,
                                "received_value": val,
                            }
                        }
                        errors.append(error)
                        continue

    if len(validate_function_inputs) > 0 or validate_has_kwargs:
        input_data_all, _, hidden_inputs = get_input_data(inputs, obj_class, unique_id)
        input_filtered = {}
        for x in input_data_all:
            if x in validate_function_inputs or validate_has_kwargs:
                input_filtered[x] = input_data_all[x]
        if 'input_types' in validate_function_inputs:
            input_filtered['input_types'] = [received_types]

        ret = await _async_map_node_over_list(prompt_id, unique_id, obj_class, input_filtered, validate_function_name, hidden_inputs=hidden_inputs)
        ret = await resolve_map_node_over_list_results(ret)
        for x in input_filtered:
            for i, r in enumerate(ret):
                if r is not True and not isinstance(r, ExecutionBlocker):
                    details = f"{x}"
                    if r is not False:
                        details += f" - {str(r)}"

                    error = {
                        "type": "custom_validation_failed",
                        "message": "Custom validation failed for node",
                        "details": details,
                        "extra_info": {
                            "input_name": x,
                        }
                    }
                    errors.append(error)
                    continue

    if len(errors) > 0 or valid is not True:
        ret = ValidateInputsTuple(False, errors, unique_id)
    else:
        ret = ValidateInputsTuple(True, [], unique_id)

    validated[unique_id] = ret
    return ret


def full_type_name(klass):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__


@tracer.start_as_current_span("Validate Prompt")
async def validate_prompt(prompt_id: typing.Any, prompt: typing.Mapping[str, typing.Any], partial_execution_list: typing.Union[list[str], None] = None) -> ValidationTuple:
    # todo: partial_execution_list=None, because nobody uses these features
    res = await _validate_prompt(prompt_id, prompt, partial_execution_list)
    if not res.valid:
        span = get_current_span()
        span.set_status(Status(StatusCode.ERROR))
        if res.error is not None and len(res.error) > 0:
            span.set_attributes({
                f"error.{k}": v for k, v in res.error.items() if isinstance(v, (bool, str, bytes, int, float, list))
            })
            if "extra_info" in res.error and isinstance(res.error["extra_info"], dict):
                extra_info: ValidationErrorExtraInfoDict = res.error["extra_info"]
                span.set_attributes({
                    f"error.extra_info.{k}": v for k, v in extra_info.items() if isinstance(v, (str, list))
                })
        if len(res.node_errors) > 0:
            for node_id, node_error in res.node_errors.items():
                for node_error_field, node_error_value in node_error.items():
                    if isinstance(node_error_value, (str, bool, int, float)):
                        span.set_attribute(f"node_errors.{node_id}.{node_error_field}", node_error_value)
    return res


async def _validate_prompt(prompt_id: typing.Any, prompt: typing.Mapping[str, typing.Any], partial_execution_list: typing.Union[list[str], None] = None) -> ValidationTuple:
    outputs = set()
    for x in prompt:
        if 'class_type' not in prompt[x]:
            error = {
                "type": "invalid_prompt",
                "message": "Cannot execute because a node is missing the class_type property.",
                "details": f"Node ID '#{x}'",
                "extra_info": {}
            }
            return ValidationTuple(False, error, [], {})

        class_type = prompt[x]['class_type']
        class_ = get_nodes().NODE_CLASS_MAPPINGS.get(class_type, None)
        if class_ is None:
            error = {
                "type": "invalid_prompt",
                "message": f"Cannot execute because node {class_type} does not exist.",
                "details": f"Node ID '#{x}'",
                "extra_info": {}
            }
            return ValidationTuple(False, error, [], {})

        if hasattr(class_, 'OUTPUT_NODE') and class_.OUTPUT_NODE is True:
            if partial_execution_list is None or x in partial_execution_list:
                outputs.add(x)

    if len(outputs) == 0:
        error = {
            "type": "prompt_no_outputs",
            "message": "Prompt has no outputs",
            "details": "",
            "extra_info": {}
        }
        return ValidationTuple(False, error, [], {})

    good_outputs = set()
    errors = []
    node_errors: typing.Dict[str, NodeErrorsDictValue] = {}
    validated: typing.Dict[str, ValidateInputsTuple] = {}
    for o in outputs:
        valid = False
        reasons: List[ValidationErrorDict] = []
        try:
            m = await validate_inputs(prompt_id, prompt, o, validated)
            valid = m[0]
            reasons = m[1]
        except Exception as ex:
            typ, _, tb = sys.exc_info()
            valid = False
            exception_type = full_type_name(typ)
            reasons = [{
                "type": "exception_during_validation",
                "message": "Exception when validating node",
                "details": str(ex),
                "extra_info": {
                    "exception_type": exception_type,
                    "traceback": traceback.format_tb(tb)
                }
            }]
            validated[o] = ValidateInputsTuple(False, reasons, o)

        if valid is True:
            good_outputs.add(o)
        else:
            msgs: list[str] = [f"Failed to validate prompt for output {o}:"]
            if len(reasons) > 0:
                msgs.append("* (prompt):")
                for reason in reasons:
                    msgs.append(f"  - {reason['message']}: {reason['details']}")
            errors += [(o, reasons)]
            for node_id, result in validated.items():
                valid = result[0]
                reasons = result[1]
                # If a node upstream has errors, the nodes downstream will also
                # be reported as invalid, but there will be no errors attached.
                # So don't return those nodes as having errors in the response.
                if valid is not True and len(reasons) > 0:
                    if node_id not in node_errors:
                        class_type = prompt[node_id]['class_type']
                        node_errors[node_id] = {
                            "errors": reasons,
                            "dependent_outputs": [],
                            "class_type": class_type
                        }
                        msgs.append(f"* {class_type} {node_id}:")
                        for reason in reasons:
                            msgs.append(f"  - {reason['message']}: {reason['details']}")
                    node_errors[node_id]["dependent_outputs"].append(o)
            logger.info(' '.join(msgs))

    if len(good_outputs) == 0:
        errors_list = []
        extra_info = {}
        for o, _errors in errors:
            for error in _errors:
                errors_list.append(f"{error['message']}: {error['details']}")
                # Aggregate exception_type and traceback from validation errors
                if 'extra_info' in error and error['extra_info']:
                    if 'exception_type' in error['extra_info'] and 'exception_type' not in extra_info:
                        extra_info['exception_type'] = error['extra_info']['exception_type']
                    if 'traceback' in error['extra_info'] and 'traceback' not in extra_info:
                        extra_info['traceback'] = error['extra_info']['traceback']

        # Per OpenAPI spec, extra_info must have exception_type and traceback
        # For non-exception validation errors, provide synthetic values
        if 'exception_type' not in extra_info:
            extra_info['exception_type'] = 'ValidationError'
        if 'traceback' not in extra_info:
            # Capture current stack for validation errors that don't have their own traceback
            extra_info['traceback'] = traceback.format_stack()

        # Include detailed node_errors for actionable debugging information
        if node_errors:
            extra_info['node_errors'] = node_errors

        errors_list = "\n".join(errors_list)

        error = {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": errors_list,
            "extra_info": extra_info
        }

        return ValidationTuple(False, error, list(good_outputs), node_errors)

    return ValidationTuple(True, None, list(good_outputs), node_errors)


class PromptQueue(AbstractPromptQueue):
    def __init__(self, server: ExecutorToClientProgress):
        self.server = server
        self.mutex = threading.RLock()
        self.not_empty = threading.Condition(self.mutex)
        self.queue: typing.List[QueueItem] = []
        self.currently_running: typing.Dict[str, QueueItem] = {}
        # history maps the second integer prompt id in the queue tuple to a dictionary with keys "prompt" and "outputs
        # todo: use the new History class for the sake of simplicity
        self.history: typing.Dict[str, HistoryEntry] = {}
        self.flags = {}

    def size(self) -> int:
        return len(self.queue)

    def put(self, item: QueueItem):
        with self.mutex:
            heapq.heappush(self.queue, item)
            self.server.queue_updated()
            self.not_empty.notify()

    def get(self, timeout=None) -> typing.Optional[typing.Tuple[QueueTuple, str]]:
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait(timeout=timeout)
                if timeout is not None and len(self.queue) == 0:
                    return None
            item_with_future: QueueItem = heapq.heappop(self.queue)
            assert item_with_future.prompt_id is not None
            assert item_with_future.prompt_id != ""
            assert item_with_future.prompt_id not in self.currently_running
            assert isinstance(item_with_future.prompt_id, str)
            task_id = item_with_future.prompt_id
            self.currently_running[task_id] = item_with_future
            self.server.queue_updated()
            return copy.deepcopy(item_with_future.queue_tuple), task_id

    def task_done(self, item_id: str, outputs: HistoryResultDict,
                  status: Optional[ExecutionStatus], error_details: Optional[ExecutionErrorMessage] = None):
        history_result = outputs
        with self.mutex:
            queue_item = self.currently_running.pop(item_id)
            prompt = queue_item.queue_tuple
            if len(self.history) > MAXIMUM_HISTORY_SIZE:
                self.history.pop(next(iter(self.history)))

            status_dict = None
            if status is not None:
                status_dict: Optional[ExecutionStatusAsDict] = status.as_dict(error_details=error_details)

            outputs_ = history_result["outputs"]
            # Remove sensitive data from extra_data before storing in history
            for sensitive_val in SENSITIVE_EXTRA_DATA_KEYS:
                if sensitive_val in prompt[3]:
                    prompt[3].pop(sensitive_val)

            history_entry: HistoryEntry = {
                "prompt": prompt,
                "outputs": copy.deepcopy(outputs_),
            }
            if status_dict is not None:
                history_entry["status"] = status_dict
            self.history[prompt[1]] = history_entry
            self.history[prompt[1]].update(history_result)
            self.server.queue_updated()
            if queue_item.completed:
                queue_item.completed.set_result(outputs_)

    # Note: slow
    def get_current_queue(self) -> Tuple[typing.List[QueueTuple], typing.List[QueueTuple]]:
        with self.mutex:
            out: typing.List[QueueTuple] = []
            for x in self.currently_running.values():
                out += [x.queue_tuple]
            return out, copy.deepcopy([item.queue_tuple for item in self.queue])

    # read-safe as long as queue items are immutable
    def get_current_queue_volatile(self):
        with self.mutex:
            running = [x for x in self.currently_running.values()]
            queued = copy.copy(self.queue)
            return (running, queued)

    def get_tasks_remaining(self):
        with self.mutex:
            return len(self.queue) + len(self.currently_running)

    def wipe_queue(self):
        with self.mutex:
            for item in self.queue:
                if item.completed:
                    item.completed.set_exception(Exception("queue cancelled"))
            self.queue = []
            self.server.queue_updated()

    def delete_queue_item(self, function):
        with self.mutex:
            for x in range(len(self.queue)):
                if function(self.queue[x].queue_tuple):
                    if len(self.queue) == 1:
                        self.wipe_queue()
                    else:
                        item = self.queue.pop(x)
                        if item.completed:
                            item.completed.set_exception(Exception("queue item deleted"))
                        heapq.heapify(self.queue)
                    self.server.queue_updated()
                    return True
        return False

    def get_history(self, prompt_id=None, max_items=None, offset=-1, map_function=None):
        with self.mutex:
            if prompt_id is None:
                out = {}
                i = 0
                if offset < 0 and max_items is not None:
                    offset = len(self.history) - max_items
                for k in self.history:
                    if i >= offset:
                        p = self.history[k]
                        if map_function is not None:
                            p = map_function(p)
                        out[k] = p
                        if max_items is not None and len(out) >= max_items:
                            break
                    i += 1
                return out
            elif prompt_id in self.history:
                p = self.history[prompt_id]
                if map_function is None:
                    p = copy.deepcopy(p)
                else:
                    p = map_function(p)
                return {prompt_id: p}
            else:
                return {}

    def wipe_history(self):
        with self.mutex:
            self.history.clear()

    def delete_history_item(self, id_to_delete: str):
        with self.mutex:
            self.history.pop(id_to_delete, None)

    def set_flag(self, name, data):
        with self.mutex:
            self.flags[name] = data
            self.not_empty.notify()

    def get_flags(self, reset=True):
        with self.mutex:
            if reset:
                ret = self.flags
                self.flags = {}
                return ret
            else:
                return self.flags.copy()

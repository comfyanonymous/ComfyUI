import sys
import copy
import logging
import threading
import heapq
import traceback
from enum import Enum
import inspect
from typing import List, Literal, NamedTuple, Optional

import torch
import nodes

import comfy.model_management
import comfy.graph_utils
from comfy.graph import get_input_info, ExecutionList, DynamicPrompt, ExecutionBlocker
from comfy.graph_utils import is_link, GraphBuilder
from comfy.caching import HierarchicalCache, LRUCache, CacheKeySetInputSignature, CacheKeySetInputSignatureWithID, CacheKeySetID
from comfy.cli_args import args

class ExecutionResult(Enum):
    SUCCESS = 0
    FAILURE = 1
    PENDING = 2

class DuplicateNodeError(Exception):
    pass

class IsChangedCache:
    def __init__(self, dynprompt, outputs_cache):
        self.dynprompt = dynprompt
        self.outputs_cache = outputs_cache
        self.is_changed = {}

    def get(self, node_id):
        if node_id in self.is_changed:
            return self.is_changed[node_id]

        node = self.dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        if not hasattr(class_def, "IS_CHANGED"):
            self.is_changed[node_id] = False
            return self.is_changed[node_id]

        if "is_changed" in node:
            self.is_changed[node_id] = node["is_changed"]
            return self.is_changed[node_id]

        input_data_all, _ = get_input_data(node["inputs"], class_def, node_id, self.outputs_cache)
        try:
            is_changed = map_node_over_list(class_def, input_data_all, "IS_CHANGED")
            node["is_changed"] = [None if isinstance(x, ExecutionBlocker) else x for x in is_changed]
        except:
            node["is_changed"] = float("NaN")
        finally:
            self.is_changed[node_id] = node["is_changed"]
        return self.is_changed[node_id]

class CacheSet:
    def __init__(self, lru_size=None):
        if lru_size is None or lru_size == 0:
            self.init_classic_cache() 
        else:
            self.init_lru_cache(lru_size)
        self.all = [self.outputs, self.ui, self.objects]

    # Useful for those with ample RAM/VRAM -- allows experimenting without
    # blowing away the cache every time
    def init_lru_cache(self, cache_size):
        self.outputs = LRUCache(CacheKeySetInputSignature, max_size=cache_size)
        self.ui = LRUCache(CacheKeySetInputSignatureWithID, max_size=cache_size)
        self.objects = HierarchicalCache(CacheKeySetID)

    # Performs like the old cache -- dump data ASAP
    def init_classic_cache(self):
        self.outputs = HierarchicalCache(CacheKeySetInputSignature)
        self.ui = HierarchicalCache(CacheKeySetInputSignatureWithID)
        self.objects = HierarchicalCache(CacheKeySetID)

    def recursive_debug_dump(self):
        result = {
            "outputs": self.outputs.recursive_debug_dump(),
            "ui": self.ui.recursive_debug_dump(),
        }
        return result

def get_input_data(inputs, class_def, unique_id, outputs=None, dynprompt=None, extra_data={}):
    valid_inputs = class_def.INPUT_TYPES()
    input_data_all = {}
    missing_keys = {}
    for x in inputs:
        input_data = inputs[x]
        input_type, input_category, input_info = get_input_info(class_def, x)
        def mark_missing():
            missing_keys[x] = True
            input_data_all[x] = (None,)
        if is_link(input_data) and (not input_info or not input_info.get("rawLink", False)):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if outputs is None:
                mark_missing()
                continue # This might be a lazily-evaluated input
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
    return input_data_all, missing_keys

def map_node_over_list(obj, input_data_all, func, allow_interrupt=False, execution_block_cb=None, pre_execute_cb=None):
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
    def process_inputs(inputs, index=None):
        if allow_interrupt:
            nodes.before_node_execution()
        execution_block = None
        for k, v in inputs.items():
            if isinstance(v, ExecutionBlocker):
                execution_block = execution_block_cb(v) if execution_block_cb else v
                break
        if execution_block is None:
            if pre_execute_cb is not None and index is not None:
                pre_execute_cb(index)
            results.append(getattr(obj, func)(**inputs))
        else:
            results.append(execution_block)

    if input_is_list:
        process_inputs(input_data_all, 0)
    elif max_len_input == 0:
        process_inputs({})
    else: 
        for i in range(max_len_input):
            input_dict = slice_dict(input_data_all, i)
            process_inputs(input_dict, i)
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
            output.append([x for o in results for x in o[i]])
        else:
            output.append([o[i] for o in results])
    return output

def get_output_data(obj, input_data_all, execution_block_cb=None, pre_execute_cb=None):
    
    results = []
    uis = []
    subgraph_results = []
    return_values = map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
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
    if len(uis) > 0:
        ui = {k: [y for x in uis for y in x[k]] for k in uis[0].keys()}
    return output, ui, has_subgraph

def format_value(x):
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)

def execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id, execution_list, pending_subgraph_results):
    unique_id = current_item
    real_node_id = dynprompt.get_real_node_id(unique_id)
    display_node_id = dynprompt.get_display_node_id(unique_id)
    parent_node_id = dynprompt.get_parent_node_id(unique_id)
    inputs = dynprompt.get_node(unique_id)['inputs']
    class_type = dynprompt.get_node(unique_id)['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    if caches.outputs.get(unique_id) is not None:
        if server.client_id is not None:
            cached_output = caches.ui.get(unique_id) or {}
            server.send_sync("executed", { "node": unique_id, "display_node": display_node_id, "output": cached_output.get("output",None), "prompt_id": prompt_id }, server.client_id)
        return (ExecutionResult.SUCCESS, None, None)

    input_data_all = None
    try:
        if unique_id in pending_subgraph_results:
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
            input_data_all, missing_keys = get_input_data(inputs, class_def, unique_id, caches.outputs, dynprompt, extra_data)
            if server.client_id is not None:
                server.last_node_id = display_node_id
                server.send_sync("executing", { "node": unique_id, "display_node": display_node_id, "prompt_id": prompt_id }, server.client_id)

            obj = caches.objects.get(unique_id)
            if obj is None:
                obj = class_def()
                caches.objects.set(unique_id, obj)

            if hasattr(obj, "check_lazy_status"):
                required_inputs = map_node_over_list(obj, input_data_all, "check_lazy_status", allow_interrupt=True)
                required_inputs = set(sum([r for r in required_inputs if isinstance(r,list)], []))
                required_inputs = [x for x in required_inputs if isinstance(x,str) and (
                    x not in input_data_all or x in missing_keys
                )]
                if len(required_inputs) > 0:
                    for i in required_inputs:
                        execution_list.make_input_strong_link(unique_id, i)
                    return (ExecutionResult.PENDING, None, None)

            def execution_block_cb(block):
                if block.message is not None:
                    mes = {
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
                GraphBuilder.set_default_prefix(unique_id, call_index, 0)
            output_data, output_ui, has_subgraph = get_output_data(obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
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
                server.send_sync("executed", { "node": unique_id, "display_node": display_node_id, "output": output_ui, "prompt_id": prompt_id }, server.client_id)
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
                        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
                        if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                            new_output_ids.append(node_id)
                    for i in range(len(node_outputs)):
                        if is_link(node_outputs[i]):
                            from_node_id, from_socket = node_outputs[i][0], node_outputs[i][1]
                            new_output_links.append((from_node_id, from_socket))
                    cached_outputs.append((True, node_outputs))
            new_node_ids = set(new_node_ids)
            for cache in caches.all:
                cache.ensure_subcache_for(unique_id, new_node_ids).clean_unused()
            for node_id in new_output_ids:
                execution_list.add_node(node_id)
            for link in new_output_links:
                execution_list.add_strong_link(link[0], link[1], unique_id)
            pending_subgraph_results[unique_id] = cached_outputs
            return (ExecutionResult.PENDING, None, None)
        caches.outputs.set(unique_id, output_data)
    except comfy.model_management.InterruptProcessingException as iex:
        logging.info("Processing interrupted")

        # skip formatting inputs/outputs
        error_details = {
            "node_id": real_node_id,
        }

        return (ExecutionResult.FAILURE, error_details, iex)
    except Exception as ex:
        typ, _, tb = sys.exc_info()
        exception_type = full_type_name(typ)
        input_data_formatted = {}
        if input_data_all is not None:
            input_data_formatted = {}
            for name, inputs in input_data_all.items():
                input_data_formatted[name] = [format_value(x) for x in inputs]

        logging.error(f"!!! Exception during processing !!! {ex}")
        logging.error(traceback.format_exc())

        error_details = {
            "node_id": real_node_id,
            "exception_message": str(ex),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted
        }
        return (ExecutionResult.FAILURE, error_details, ex)

    executed.add(unique_id)

    return (ExecutionResult.SUCCESS, None, None)

class PromptExecutor:
    def __init__(self, server, lru_size=None):
        self.lru_size = lru_size
        self.server = server
        self.reset()

    def reset(self):
        self.caches = CacheSet(self.lru_size)
        self.status_messages = []
        self.success = True

    def add_message(self, event, data, broadcast: bool):
        self.status_messages.append((event, data))
        if self.server.client_id is not None or broadcast:
            self.server.send_sync(event, data, self.server.client_id)

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]

        # First, send back the status to the frontend depending
        # on the exception type
        if isinstance(ex, comfy.model_management.InterruptProcessingException):
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
            }
            self.add_message("execution_interrupted", mes, broadcast=True)
        else:
            mes = {
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
        
    def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        nodes.interrupt_processing(False)

        if "client_id" in extra_data:
            self.server.client_id = extra_data["client_id"]
        else:
            self.server.client_id = None

        self.status_messages = []
        self.add_message("execution_start", { "prompt_id": prompt_id}, broadcast=False)

        with torch.inference_mode():
            dynamic_prompt = DynamicPrompt(prompt)
            is_changed_cache = IsChangedCache(dynamic_prompt, self.caches.outputs)
            for cache in self.caches.all:
                cache.set_prompt(dynamic_prompt, prompt.keys(), is_changed_cache)
                cache.clean_unused()

            current_outputs = self.caches.outputs.all_node_ids()

            comfy.model_management.cleanup_models(keep_clone_weights_loaded=True)
            self.add_message("execution_cached",
                          { "nodes": list(current_outputs) , "prompt_id": prompt_id},
                          broadcast=False)
            pending_subgraph_results = {}
            executed = set()
            execution_list = ExecutionList(dynamic_prompt, self.caches.outputs)
            for node_id in list(execute_outputs):
                execution_list.add_node(node_id)

            while not execution_list.is_empty():
                node_id, error, ex = execution_list.stage_node_execution()
                if error is not None:
                    self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
                    break

                result, error, ex = execute(self.server, dynamic_prompt, self.caches, node_id, extra_data, executed, prompt_id, execution_list, pending_subgraph_results)
                if result == ExecutionResult.FAILURE:
                    self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
                    break
                elif result == ExecutionResult.PENDING:
                    execution_list.unstage_node_execution()
                else: # result == ExecutionResult.SUCCESS:
                    execution_list.complete_node_execution()

            ui_outputs = {}
            meta_outputs = {}
            for ui_info in self.caches.ui.all_active_values():
                node_id = ui_info["meta"]["node_id"]
                ui_outputs[node_id] = ui_info["output"]
                meta_outputs[node_id] = ui_info["meta"]
            self.history_result = {
                "outputs": ui_outputs,
                "meta": meta_outputs,
            }
            self.server.last_node_id = None
            if comfy.model_management.DISABLE_SMART_MEMORY:
                comfy.model_management.unload_all_models()



def validate_inputs(prompt, item, validated):
    unique_id = item
    if unique_id in validated:
        return validated[unique_id]

    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    obj_class = nodes.NODE_CLASS_MAPPINGS[class_type]

    class_inputs = obj_class.INPUT_TYPES()
    valid_inputs = set(class_inputs.get('required',{})).union(set(class_inputs.get('optional',{})))

    errors = []
    valid = True

    validate_function_inputs = []
    if hasattr(obj_class, "VALIDATE_INPUTS"):
        validate_function_inputs = inspect.getfullargspec(obj_class.VALIDATE_INPUTS).args
    received_types = {}

    for x in valid_inputs:
        type_input, input_category, extra_info = get_input_info(obj_class, x)
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
        info = (type_input, extra_info)
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
            r = nodes.NODE_CLASS_MAPPINGS[o_class_type].RETURN_TYPES
            received_type = r[val[1]]
            received_types[x] = received_type
            if 'input_types' not in validate_function_inputs and received_type != type_input:
                details = f"{x}, {received_type} != {type_input}"
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
                r = validate_inputs(prompt, o_id, validated)
                if r[0] is False:
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
                validated[o_id] = (False, reasons, o_id)
                continue
        else:
            try:
                if type_input == "INT":
                    val = int(val)
                    inputs[x] = val
                if type_input == "FLOAT":
                    val = float(val)
                    inputs[x] = val
                if type_input == "STRING":
                    val = str(val)
                    inputs[x] = val
                if type_input == "BOOLEAN":
                    val = bool(val)
                    inputs[x] = val
            except Exception as ex:
                error = {
                    "type": "invalid_input_type",
                    "message": f"Failed to convert an input value to a {type_input} value",
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
                if "min" in extra_info and val < extra_info["min"]:
                    error = {
                        "type": "value_smaller_than_min",
                        "message": "Value {} smaller than min of {}".format(val, extra_info["min"]),
                        "details": f"{x}",
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue
                if "max" in extra_info and val > extra_info["max"]:
                    error = {
                        "type": "value_bigger_than_max",
                        "message": "Value {} bigger than max of {}".format(val, extra_info["max"]),
                        "details": f"{x}",
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue

                if isinstance(type_input, list):
                    if val not in type_input:
                        input_config = info
                        list_info = ""

                        # Don't send back gigantic lists like if they're lots of
                        # scanned model filepaths
                        if len(type_input) > 20:
                            list_info = f"(list of length {len(type_input)})"
                            input_config = None
                        else:
                            list_info = str(type_input)

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

    if len(validate_function_inputs) > 0:
        input_data_all, _ = get_input_data(inputs, obj_class, unique_id)
        input_filtered = {}
        for x in input_data_all:
            if x in validate_function_inputs:
                input_filtered[x] = input_data_all[x]
        if 'input_types' in validate_function_inputs:
            input_filtered['input_types'] = [received_types]

        #ret = obj_class.VALIDATE_INPUTS(**input_filtered)
        ret = map_node_over_list(obj_class, input_filtered, "VALIDATE_INPUTS")
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
        ret = (False, errors, unique_id)
    else:
        ret = (True, [], unique_id)

    validated[unique_id] = ret
    return ret

def full_type_name(klass):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__

def validate_prompt(prompt):
    outputs = set()
    for x in prompt:
        class_ = nodes.NODE_CLASS_MAPPINGS[prompt[x]['class_type']]
        if hasattr(class_, 'OUTPUT_NODE') and class_.OUTPUT_NODE == True:
            outputs.add(x)

    if len(outputs) == 0:
        error = {
            "type": "prompt_no_outputs",
            "message": "Prompt has no outputs",
            "details": "",
            "extra_info": {}
        }
        return (False, error, [], [])

    good_outputs = set()
    errors = []
    node_errors = {}
    validated = {}
    for o in outputs:
        valid = False
        reasons = []
        try:
            m = validate_inputs(prompt, o, validated)
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
            validated[o] = (False, reasons, o)

        if valid is True:
            good_outputs.add(o)
        else:
            logging.error(f"Failed to validate prompt for output {o}:")
            if len(reasons) > 0:
                logging.error("* (prompt):")
                for reason in reasons:
                    logging.error(f"  - {reason['message']}: {reason['details']}")
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
                        logging.error(f"* {class_type} {node_id}:")
                        for reason in reasons:
                            logging.error(f"  - {reason['message']}: {reason['details']}")
                    node_errors[node_id]["dependent_outputs"].append(o)
            logging.error("Output will be ignored")

    if len(good_outputs) == 0:
        errors_list = []
        for o, errors in errors:
            for error in errors:
                errors_list.append(f"{error['message']}: {error['details']}")
        errors_list = "\n".join(errors_list)

        error = {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": errors_list,
            "extra_info": {}
        }

        return (False, error, list(good_outputs), node_errors)

    return (True, None, list(good_outputs), node_errors)

MAXIMUM_HISTORY_SIZE = 10000

class PromptQueue:
    def __init__(self, server):
        self.server = server
        self.mutex = threading.RLock()
        self.not_empty = threading.Condition(self.mutex)
        self.task_counter = 0
        self.queue = []
        self.currently_running = {}
        self.history = {}
        self.flags = {}
        server.prompt_queue = self

    def put(self, item):
        with self.mutex:
            heapq.heappush(self.queue, item)
            self.server.queue_updated()
            self.not_empty.notify()

    def get(self, timeout=None):
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait(timeout=timeout)
                if timeout is not None and len(self.queue) == 0:
                    return None
            item = heapq.heappop(self.queue)
            i = self.task_counter
            self.currently_running[i] = copy.deepcopy(item)
            self.task_counter += 1
            self.server.queue_updated()
            return (item, i)

    class ExecutionStatus(NamedTuple):
        status_str: Literal['success', 'error']
        completed: bool
        messages: List[str]

    def task_done(self, item_id, history_result,
                  status: Optional['PromptQueue.ExecutionStatus']):
        with self.mutex:
            prompt = self.currently_running.pop(item_id)
            if len(self.history) > MAXIMUM_HISTORY_SIZE:
                self.history.pop(next(iter(self.history)))

            status_dict: Optional[dict] = None
            if status is not None:
                status_dict = copy.deepcopy(status._asdict())

            self.history[prompt[1]] = {
                "prompt": prompt,
                "outputs": {},
                'status': status_dict,
            }
            self.history[prompt[1]].update(history_result)
            self.server.queue_updated()

    def get_current_queue(self):
        with self.mutex:
            out = []
            for x in self.currently_running.values():
                out += [x]
            return (out, copy.deepcopy(self.queue))

    def get_tasks_remaining(self):
        with self.mutex:
            return len(self.queue) + len(self.currently_running)

    def wipe_queue(self):
        with self.mutex:
            self.queue = []
            self.server.queue_updated()

    def delete_queue_item(self, function):
        with self.mutex:
            for x in range(len(self.queue)):
                if function(self.queue[x]):
                    if len(self.queue) == 1:
                        self.wipe_queue()
                    else:
                        self.queue.pop(x)
                        heapq.heapify(self.queue)
                    self.server.queue_updated()
                    return True
        return False

    def get_history(self, prompt_id=None, max_items=None, offset=-1):
        with self.mutex:
            if prompt_id is None:
                out = {}
                i = 0
                if offset < 0 and max_items is not None:
                    offset = len(self.history) - max_items
                for k in self.history:
                    if i >= offset:
                        out[k] = self.history[k]
                        if max_items is not None and len(out) >= max_items:
                            break
                    i += 1
                return out
            elif prompt_id in self.history:
                return {prompt_id: copy.deepcopy(self.history[prompt_id])}
            else:
                return {}

    def wipe_history(self):
        with self.mutex:
            self.history = {}

    def delete_history_item(self, id_to_delete):
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

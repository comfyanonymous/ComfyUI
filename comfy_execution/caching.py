import asyncio
import bisect
import gc
import itertools
import psutil
import time
import torch
from typing import Sequence, Mapping, Dict
from comfy.model_patcher import is_model_patcher_output
from comfy_execution.graph import DynamicPrompt
from abc import ABC, abstractmethod

import nodes

from comfy_execution.graph_utils import is_link

NODE_CLASS_CONTAINS_UNIQUE_ID: Dict[str, bool] = {}


def include_unique_id_in_input(class_type: str) -> bool:
    """Return whether a node class includes UNIQUE_ID among its hidden inputs."""
    if class_type in NODE_CLASS_CONTAINS_UNIQUE_ID:
        return NODE_CLASS_CONTAINS_UNIQUE_ID[class_type]
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    NODE_CLASS_CONTAINS_UNIQUE_ID[class_type] = "UNIQUE_ID" in class_def.INPUT_TYPES().get("hidden", {}).values()
    return NODE_CLASS_CONTAINS_UNIQUE_ID[class_type]

class CacheKeySet(ABC):
    """Base helper for building and storing cache keys for prompt nodes."""
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        """Initialize cache-key storage for a dynamic prompt execution pass."""
        self.keys = {}
        self.subcache_keys = {}

    @abstractmethod
    async def add_keys(self, node_ids):
        """Populate cache keys for the provided node ids."""
        raise NotImplementedError()

    def all_node_ids(self):
        """Return the set of node ids currently tracked by this key set."""
        return set(self.keys.keys())

    def get_used_keys(self):
        """Return the computed cache keys currently in use."""
        return self.keys.values()

    def get_used_subcache_keys(self):
        """Return the computed subcache keys currently in use."""
        return self.subcache_keys.values()

    def get_data_key(self, node_id):
        """Return the cache key for a node, if present."""
        return self.keys.get(node_id, None)

    def get_subcache_key(self, node_id):
        """Return the subcache key for a node, if present."""
        return self.subcache_keys.get(node_id, None)

class Unhashable:
    """Hashable identity sentinel for values that cannot be represented safely in cache keys."""
    pass


_PRIMITIVE_SIGNATURE_TYPES = (int, float, str, bool, bytes, type(None))
_CONTAINER_SIGNATURE_TYPES = (dict, list, tuple, set, frozenset)
_MAX_SIGNATURE_DEPTH = 32
_MAX_SIGNATURE_CONTAINER_VISITS = 10_000
_FAILED_SIGNATURE = object()


def _shallow_is_changed_signature(value):
    """Reduce execution-time `is_changed` values through a fail-closed builtin canonicalizer."""
    value_type = type(value)
    if value_type in _PRIMITIVE_SIGNATURE_TYPES:
        return value

    if value_type not in _CONTAINER_SIGNATURE_TYPES:
        return Unhashable()

    canonical = _signature_to_hashable(value, max_nodes=64)
    if type(canonical) is Unhashable:
        return canonical
    if value_type is list or value_type is tuple:
        container_tag = "is_changed_list" if value_type is list else "is_changed_tuple"
        return (container_tag, canonical[1])

    return canonical


def _primitive_signature_sort_key(obj):
    """Return a deterministic ordering key for primitive signature values."""
    obj_type = type(obj)
    return ("primitive", obj_type.__module__, obj_type.__qualname__, repr(obj))


def _sanitized_sort_key(obj, depth=0, max_depth=_MAX_SIGNATURE_DEPTH, active=None, memo=None):
    """Return a deterministic ordering key for sanitized built-in container content."""
    if depth >= max_depth:
        return ("MAX_DEPTH",)

    if active is None:
        active = set()
    if memo is None:
        memo = {}

    obj_type = type(obj)
    if obj_type is Unhashable:
        return ("UNHASHABLE",)
    elif obj_type in _PRIMITIVE_SIGNATURE_TYPES:
        return (obj_type.__module__, obj_type.__qualname__, repr(obj))
    elif obj_type not in _CONTAINER_SIGNATURE_TYPES:
        return (obj_type.__module__, obj_type.__qualname__, "OPAQUE")

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]
    if obj_id in active:
        return ("CYCLE",)

    active.add(obj_id)
    try:
        if obj_type is dict:
            items = [
                (
                    _sanitized_sort_key(k, depth + 1, max_depth, active, memo),
                    _sanitized_sort_key(v, depth + 1, max_depth, active, memo),
                )
                for k, v in obj.items()
            ]
            items.sort()
            result = ("dict", tuple(items))
        elif obj_type is list:
            result = ("list", tuple(_sanitized_sort_key(i, depth + 1, max_depth, active, memo) for i in obj))
        elif obj_type is tuple:
            result = ("tuple", tuple(_sanitized_sort_key(i, depth + 1, max_depth, active, memo) for i in obj))
        elif obj_type is set:
            result = ("set", tuple(sorted(_sanitized_sort_key(i, depth + 1, max_depth, active, memo) for i in obj)))
        else:
            result = ("frozenset", tuple(sorted(_sanitized_sort_key(i, depth + 1, max_depth, active, memo) for i in obj)))
    finally:
        active.discard(obj_id)

    memo[obj_id] = result
    return result


def _signature_to_hashable_impl(obj, depth=0, max_depth=_MAX_SIGNATURE_DEPTH, active=None, memo=None, budget=None):
    """Canonicalize signature inputs directly into their final hashable form."""
    if depth >= max_depth:
        return _FAILED_SIGNATURE

    if active is None:
        active = set()
    if memo is None:
        memo = {}
    if budget is None:
        budget = {"remaining": _MAX_SIGNATURE_CONTAINER_VISITS}

    obj_type = type(obj)
    if obj_type in _PRIMITIVE_SIGNATURE_TYPES:
        return obj, _primitive_signature_sort_key(obj)
    if obj_type is Unhashable or obj_type not in _CONTAINER_SIGNATURE_TYPES:
        return _FAILED_SIGNATURE

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]
    if obj_id in active:
        return _FAILED_SIGNATURE

    budget["remaining"] -= 1
    if budget["remaining"] < 0:
        return _FAILED_SIGNATURE

    active.add(obj_id)
    try:
        if obj_type is dict:
            try:
                items = list(obj.items())
            except RuntimeError:
                return _FAILED_SIGNATURE

            ordered_items = []
            for key, value in items:
                if type(key) not in _PRIMITIVE_SIGNATURE_TYPES:
                    return _FAILED_SIGNATURE
                key_result = (key, _primitive_signature_sort_key(key))
                value_result = _signature_to_hashable_impl(value, depth + 1, max_depth, active, memo, budget)
                if value_result is _FAILED_SIGNATURE:
                    return _FAILED_SIGNATURE
                key_value, key_sort = key_result
                value_value, value_sort = value_result
                ordered_items.append((key_sort, value_sort, key_value, value_value))

            ordered_items.sort(key=lambda item: (item[0], item[1]))
            for index in range(1, len(ordered_items)):
                previous_key_sort = ordered_items[index - 1][0]
                current_key_sort = ordered_items[index][0]
                if previous_key_sort == current_key_sort:
                    return _FAILED_SIGNATURE

            value = ("dict", tuple((key_value, value_value) for _, _, key_value, value_value in ordered_items))
            sort_key = ("dict", tuple((key_sort, value_sort) for key_sort, value_sort, _, _ in ordered_items))
        elif obj_type is list or obj_type is tuple:
            try:
                items = list(obj)
            except RuntimeError:
                return _FAILED_SIGNATURE

            child_results = []
            for item in items:
                child_result = _signature_to_hashable_impl(item, depth + 1, max_depth, active, memo, budget)
                if child_result is _FAILED_SIGNATURE:
                    return _FAILED_SIGNATURE
                child_results.append(child_result)

            container_tag = "list" if obj_type is list else "tuple"
            value = (container_tag, tuple(child for child, _ in child_results))
            sort_key = (container_tag, tuple(child_sort for _, child_sort in child_results))
        else:
            try:
                items = list(obj)
            except RuntimeError:
                return _FAILED_SIGNATURE

            ordered_items = []
            for item in items:
                child_result = _signature_to_hashable_impl(item, depth + 1, max_depth, active, memo, budget)
                if child_result is _FAILED_SIGNATURE:
                    return _FAILED_SIGNATURE
                child_value, child_sort = child_result
                ordered_items.append((child_sort, child_value))

            ordered_items.sort(key=lambda item: item[0])
            for index in range(1, len(ordered_items)):
                previous_sort_key, previous_value = ordered_items[index - 1]
                current_sort_key, current_value = ordered_items[index]
                if previous_sort_key == current_sort_key and previous_value != current_value:
                    return _FAILED_SIGNATURE

            container_tag = "set" if obj_type is set else "frozenset"
            value = (container_tag, tuple(child_value for _, child_value in ordered_items))
            sort_key = (container_tag, tuple(child_sort for child_sort, _ in ordered_items))
    finally:
        active.discard(obj_id)

    memo[obj_id] = (value, sort_key)
    return memo[obj_id]


def _signature_to_hashable(obj, max_nodes=_MAX_SIGNATURE_CONTAINER_VISITS):
    """Build the final cache-signature representation in one fail-closed pass."""
    try:
        result = _signature_to_hashable_impl(obj, budget={"remaining": max_nodes})
    except RuntimeError:
        return Unhashable()
    if result is _FAILED_SIGNATURE:
        return Unhashable()
    return result[0]


def to_hashable(obj, max_nodes=_MAX_SIGNATURE_CONTAINER_VISITS):
    """Convert sanitized prompt inputs into a stable hashable representation.

    The input is expected to already be sanitized to plain built-in containers,
    but this function still fails safe for anything unexpected. Traversal is
    iterative and memoized so shared built-in substructures do not trigger
    exponential re-walks during cache-key construction.
    """
    obj_type = type(obj)
    if obj_type in _PRIMITIVE_SIGNATURE_TYPES or obj_type is Unhashable:
        return obj
    if obj_type not in _CONTAINER_SIGNATURE_TYPES:
        return Unhashable()

    memo = {}
    active = set()
    snapshots = {}
    sort_memo = {}
    processed = 0
    # Keep traversal state separate from container snapshots/results.
    work_stack = [(obj, False)]

    def resolve_value(value):
        """Resolve a child value from the completed memo table when available."""
        value_type = type(value)
        if value_type in _PRIMITIVE_SIGNATURE_TYPES or value_type is Unhashable:
            return value
        return memo.get(id(value), Unhashable())

    def is_failed(value):
        """Return whether a resolved child value represents failed canonicalization."""
        return type(value) is Unhashable

    def resolve_unordered_values(current_items, container_tag):
        """Resolve a set-like container or fail closed if ordering is ambiguous."""
        try:
            ordered_items = [
                (_sanitized_sort_key(item, memo=sort_memo), resolve_value(item))
                for item in current_items
            ]
            if any(is_failed(value) for _, value in ordered_items):
                return Unhashable()
            ordered_items.sort(key=lambda item: item[0])
        except RuntimeError:
            return Unhashable()

        for index in range(1, len(ordered_items)):
            previous_key, previous_value = ordered_items[index - 1]
            current_key, current_value = ordered_items[index]
            if previous_key == current_key and previous_value != current_value:
                return Unhashable()

        return (container_tag, tuple(value for _, value in ordered_items))

    while work_stack:
        entry = work_stack.pop()
        if len(entry) == 3:
            _, current_id, current_type = entry
            current = None
            expanded = True
        else:
            current, expanded = entry
            current_type = type(current)
            current_id = id(current)

        if not expanded and (current_type in _PRIMITIVE_SIGNATURE_TYPES or current_type is Unhashable):
            continue
        if not expanded and current_type not in _CONTAINER_SIGNATURE_TYPES:
            memo[current_id] = Unhashable()
            continue

        if current_id in memo:
            continue

        if expanded:
            active.discard(current_id)
            try:
                items = snapshots.pop(current_id, None)
                if items is None:
                    memo[current_id] = Unhashable()
                    continue

                if current_type is dict:
                    ordered_items = [
                        (_sanitized_sort_key(k, memo=sort_memo), k, resolve_value(v))
                        for k, v in items
                    ]
                    if any(type(key) not in _PRIMITIVE_SIGNATURE_TYPES or is_failed(value) for _, key, value in ordered_items):
                        memo[current_id] = Unhashable()
                        continue
                    ordered_items.sort(key=lambda item: item[0])
                    for index in range(1, len(ordered_items)):
                        if ordered_items[index - 1][0] == ordered_items[index][0]:
                            memo[current_id] = Unhashable()
                            break
                    else:
                        memo[current_id] = (
                            "dict",
                            tuple((key, value) for _, key, value in ordered_items),
                        )
                elif current_type is list:
                    resolved_items = tuple(resolve_value(item) for item in items)
                    if any(is_failed(item) for item in resolved_items):
                        memo[current_id] = Unhashable()
                    else:
                        memo[current_id] = ("list", resolved_items)
                elif current_type is tuple:
                    resolved_items = tuple(resolve_value(item) for item in items)
                    if any(is_failed(item) for item in resolved_items):
                        memo[current_id] = Unhashable()
                    else:
                        memo[current_id] = ("tuple", resolved_items)
                elif current_type is set:
                    memo[current_id] = resolve_unordered_values(items, "set")
                else:
                    memo[current_id] = resolve_unordered_values(items, "frozenset")
            except RuntimeError:
                memo[current_id] = Unhashable()
            continue

        if current_id in active:
            memo[current_id] = Unhashable()
            continue

        processed += 1
        if processed > max_nodes:
            return Unhashable()

        active.add(current_id)
        if current_type is dict:
            try:
                items = list(current.items())
                snapshots[current_id] = items
            except RuntimeError:
                memo[current_id] = Unhashable()
                active.discard(current_id)
                continue
            for key, value in items:
                if type(key) not in _PRIMITIVE_SIGNATURE_TYPES:
                    snapshots.pop(current_id, None)
                    memo[current_id] = Unhashable()
                    active.discard(current_id)
                    break
            else:
                work_stack.append(("EXPANDED", current_id, current_type))
                for _, value in reversed(items):
                    work_stack.append((value, False))
                continue
            continue
        else:
            try:
                items = list(current)
                snapshots[current_id] = items
            except RuntimeError:
                memo[current_id] = Unhashable()
                active.discard(current_id)
                continue
            work_stack.append(("EXPANDED", current_id, current_type))
            for item in reversed(items):
                work_stack.append((item, False))

    return memo.get(id(obj), Unhashable())

class CacheKeySetID(CacheKeySet):
    """Cache-key strategy that keys nodes by node id and class type."""
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        """Initialize identity-based cache keys for the supplied dynamic prompt."""
        super().__init__(dynprompt, node_ids, is_changed_cache)
        self.dynprompt = dynprompt

    async def add_keys(self, node_ids):
        """Populate identity-based keys for nodes that exist in the dynamic prompt."""
        for node_id in node_ids:
            if node_id in self.keys:
                continue
            if not self.dynprompt.has_node(node_id):
                continue
            node = self.dynprompt.get_node(node_id)
            self.keys[node_id] = (node_id, node["class_type"])
            self.subcache_keys[node_id] = (node_id, node["class_type"])

class CacheKeySetInputSignature(CacheKeySet):
    """Cache-key strategy that hashes a node's immediate inputs plus ancestor references."""
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        """Initialize input-signature-based cache keys for the supplied dynamic prompt."""
        super().__init__(dynprompt, node_ids, is_changed_cache)
        self.dynprompt = dynprompt
        self.is_changed_cache = is_changed_cache

    def include_node_id_in_input(self) -> bool:
        """Return whether node ids should be included in computed input signatures."""
        return False

    async def add_keys(self, node_ids):
        """Populate input-signature-based keys for nodes in the dynamic prompt."""
        for node_id in node_ids:
            if node_id in self.keys:
                continue
            if not self.dynprompt.has_node(node_id):
                continue
            node = self.dynprompt.get_node(node_id)
            self.keys[node_id] = await self.get_node_signature(self.dynprompt, node_id)
            self.subcache_keys[node_id] = (node_id, node["class_type"])

    async def get_node_signature(self, dynprompt, node_id):
        """Build the full cache signature for a node and its ordered ancestors."""
        signature = []
        ancestors, order_mapping = self.get_ordered_ancestry(dynprompt, node_id)
        immediate = await self.get_immediate_node_signature(dynprompt, node_id, order_mapping)
        if type(immediate) is Unhashable:
            return immediate
        signature.append(immediate)
        for ancestor_id in ancestors:
            immediate = await self.get_immediate_node_signature(dynprompt, ancestor_id, order_mapping)
            if type(immediate) is Unhashable:
                return immediate
            signature.append(immediate)
        return tuple(signature)

    async def get_immediate_node_signature(self, dynprompt, node_id, ancestor_order_mapping):
        """Build the immediate cache-signature fragment for a node.

        Link inputs are reduced to ancestor references here. Non-link values
        are canonicalized or failed closed before being appended so the final
        node signature is assembled from already-hashable fragments.
        """
        if not dynprompt.has_node(node_id):
            # This node doesn't exist -- we can't cache it.
            return Unhashable()
        node = dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        is_changed_signature = _shallow_is_changed_signature(await self.is_changed_cache.get(node_id))
        if type(is_changed_signature) is Unhashable:
            return is_changed_signature
        signature = [class_type, is_changed_signature]
        if self.include_node_id_in_input() or (hasattr(class_def, "NOT_IDEMPOTENT") and class_def.NOT_IDEMPOTENT) or include_unique_id_in_input(class_type):
            signature.append(node_id)
        inputs = node["inputs"]
        for key in sorted(inputs.keys()):
            if is_link(inputs[key]):
                (ancestor_id, ancestor_socket) = inputs[key]
                ancestor_index = ancestor_order_mapping[ancestor_id]
                signature.append((key,("ANCESTOR", ancestor_index, ancestor_socket)))
            else:
                value_signature = to_hashable(inputs[key])
                if type(value_signature) is Unhashable:
                    return value_signature
                signature.append((key, value_signature))
        return tuple(signature)

    # This function returns a list of all ancestors of the given node. The order of the list is
    # deterministic based on which specific inputs the ancestor is connected by.
    def get_ordered_ancestry(self, dynprompt, node_id):
        """Return ancestors in deterministic traversal order and their index mapping."""
        ancestors = []
        order_mapping = {}
        self.get_ordered_ancestry_internal(dynprompt, node_id, ancestors, order_mapping)
        return ancestors, order_mapping

    def get_ordered_ancestry_internal(self, dynprompt, node_id, ancestors, order_mapping):
        """Recursively collect ancestors in input order without revisiting prior nodes."""
        if not dynprompt.has_node(node_id):
            return
        inputs = dynprompt.get_node(node_id)["inputs"]
        input_keys = sorted(inputs.keys())
        for key in input_keys:
            if is_link(inputs[key]):
                ancestor_id = inputs[key][0]
                if ancestor_id not in order_mapping:
                    ancestors.append(ancestor_id)
                    order_mapping[ancestor_id] = len(ancestors) - 1
                    self.get_ordered_ancestry_internal(dynprompt, ancestor_id, ancestors, order_mapping)

class BasicCache:
    def __init__(self, key_class, enable_providers=False):
        self.key_class = key_class
        self.initialized = False
        self.enable_providers = enable_providers
        self.dynprompt: DynamicPrompt
        self.cache_key_set: CacheKeySet
        self.cache = {}
        self.subcaches = {}
        self._pending_store_tasks: set = set()

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        self.dynprompt = dynprompt
        self.cache_key_set = self.key_class(dynprompt, node_ids, is_changed_cache)
        await self.cache_key_set.add_keys(node_ids)
        self.is_changed_cache = is_changed_cache
        self.initialized = True

    def all_node_ids(self):
        assert self.initialized
        node_ids = self.cache_key_set.all_node_ids()
        for subcache in self.subcaches.values():
            node_ids = node_ids.union(subcache.all_node_ids())
        return node_ids

    def _clean_cache(self):
        preserve_keys = set(self.cache_key_set.get_used_keys())
        to_remove = []
        for key in self.cache:
            if key not in preserve_keys:
                to_remove.append(key)
        for key in to_remove:
            del self.cache[key]

    def _clean_subcaches(self):
        preserve_subcaches = set(self.cache_key_set.get_used_subcache_keys())

        to_remove = []
        for key in self.subcaches:
            if key not in preserve_subcaches:
                to_remove.append(key)
        for key in to_remove:
            del self.subcaches[key]

    def clean_unused(self):
        assert self.initialized
        self._clean_cache()
        self._clean_subcaches()

    def poll(self, **kwargs):
        pass

    def get_local(self, node_id):
        if not self.initialized:
            return None
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key in self.cache:
            return self.cache[cache_key]
        return None

    def set_local(self, node_id, value):
        assert self.initialized
        cache_key = self.cache_key_set.get_data_key(node_id)
        self.cache[cache_key] = value

    async def _set_immediate(self, node_id, value):
        assert self.initialized
        cache_key = self.cache_key_set.get_data_key(node_id)
        self.cache[cache_key] = value

        await self._notify_providers_store(node_id, cache_key, value)

    async def _get_immediate(self, node_id):
        if not self.initialized:
            return None
        cache_key = self.cache_key_set.get_data_key(node_id)

        if cache_key in self.cache:
            return self.cache[cache_key]

        external_result = await self._check_providers_lookup(node_id, cache_key)
        if external_result is not None:
            self.cache[cache_key] = external_result
            return external_result

        return None

    async def _notify_providers_store(self, node_id, cache_key, value):
        from comfy_execution.cache_provider import (
            _has_cache_providers, _get_cache_providers,
            CacheValue, _contains_self_unequal, _logger
        )

        if not self.enable_providers:
            return
        if not _has_cache_providers():
            return
        if not self._is_external_cacheable_value(value):
            return
        if _contains_self_unequal(cache_key):
            return

        context = self._build_context(node_id, cache_key)
        if context is None:
            return
        cache_value = CacheValue(outputs=value.outputs, ui=value.ui)

        for provider in _get_cache_providers():
            try:
                if provider.should_cache(context, cache_value):
                    task = asyncio.create_task(self._safe_provider_store(provider, context, cache_value))
                    self._pending_store_tasks.add(task)
                    task.add_done_callback(self._pending_store_tasks.discard)
            except Exception as e:
                _logger.warning(f"Cache provider {provider.__class__.__name__} error on store: {e}")

    @staticmethod
    async def _safe_provider_store(provider, context, cache_value):
        from comfy_execution.cache_provider import _logger
        try:
            await provider.on_store(context, cache_value)
        except Exception as e:
            _logger.warning(f"Cache provider {provider.__class__.__name__} async store error: {e}")

    async def _check_providers_lookup(self, node_id, cache_key):
        from comfy_execution.cache_provider import (
            _has_cache_providers, _get_cache_providers,
            CacheValue, _contains_self_unequal, _logger
        )

        if not self.enable_providers:
            return None
        if not _has_cache_providers():
            return None
        if _contains_self_unequal(cache_key):
            return None

        context = self._build_context(node_id, cache_key)
        if context is None:
            return None

        for provider in _get_cache_providers():
            try:
                if not provider.should_cache(context):
                    continue
                result = await provider.on_lookup(context)
                if result is not None:
                    if not isinstance(result, CacheValue):
                        _logger.warning(f"Provider {provider.__class__.__name__} returned invalid type")
                        continue
                    if not isinstance(result.outputs, (list, tuple)):
                        _logger.warning(f"Provider {provider.__class__.__name__} returned invalid outputs")
                        continue
                    from execution import CacheEntry
                    return CacheEntry(ui=result.ui, outputs=list(result.outputs))
            except Exception as e:
                _logger.warning(f"Cache provider {provider.__class__.__name__} error on lookup: {e}")

        return None

    def _is_external_cacheable_value(self, value):
        return hasattr(value, 'outputs') and hasattr(value, 'ui')

    def _get_class_type(self, node_id):
        if not self.initialized or not self.dynprompt:
            return ''
        try:
            return self.dynprompt.get_node(node_id).get('class_type', '')
        except Exception:
            return ''

    def _build_context(self, node_id, cache_key):
        from comfy_execution.cache_provider import CacheContext, _serialize_cache_key, _logger
        try:
            cache_key_hash = _serialize_cache_key(cache_key)
            if cache_key_hash is None:
                return None
            return CacheContext(
                node_id=node_id,
                class_type=self._get_class_type(node_id),
                cache_key_hash=cache_key_hash,
            )
        except Exception as e:
            _logger.warning(f"Failed to build cache context for node {node_id}: {e}")
            return None

    async def _ensure_subcache(self, node_id, children_ids):
        subcache_key = self.cache_key_set.get_subcache_key(node_id)
        subcache = self.subcaches.get(subcache_key, None)
        if subcache is None:
            subcache = BasicCache(self.key_class)
            self.subcaches[subcache_key] = subcache
        await subcache.set_prompt(self.dynprompt, children_ids, self.is_changed_cache)
        return subcache

    def _get_subcache(self, node_id):
        assert self.initialized
        subcache_key = self.cache_key_set.get_subcache_key(node_id)
        if subcache_key in self.subcaches:
            return self.subcaches[subcache_key]
        else:
            return None

    def recursive_debug_dump(self):
        result = []
        for key in self.cache:
            result.append({"key": key, "value": self.cache[key]})
        for key in self.subcaches:
            result.append({"subcache_key": key, "subcache": self.subcaches[key].recursive_debug_dump()})
        return result

class HierarchicalCache(BasicCache):
    def __init__(self, key_class, enable_providers=False):
        super().__init__(key_class, enable_providers=enable_providers)

    def _get_cache_for(self, node_id):
        assert self.dynprompt is not None
        parent_id = self.dynprompt.get_parent_node_id(node_id)
        if parent_id is None:
            return self

        hierarchy = []
        while parent_id is not None:
            hierarchy.append(parent_id)
            parent_id = self.dynprompt.get_parent_node_id(parent_id)

        cache = self
        for parent_id in reversed(hierarchy):
            cache = cache._get_subcache(parent_id)
            if cache is None:
                return None
        return cache

    async def get(self, node_id):
        cache = self._get_cache_for(node_id)
        if cache is None:
            return None
        return await cache._get_immediate(node_id)

    def get_local(self, node_id):
        cache = self._get_cache_for(node_id)
        if cache is None:
            return None
        return BasicCache.get_local(cache, node_id)

    async def set(self, node_id, value):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        await cache._set_immediate(node_id, value)

    def set_local(self, node_id, value):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        BasicCache.set_local(cache, node_id, value)

    async def ensure_subcache_for(self, node_id, children_ids):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        return await cache._ensure_subcache(node_id, children_ids)

class NullCache:

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        pass

    def all_node_ids(self):
        return []

    def clean_unused(self):
        pass

    def poll(self, **kwargs):
        pass

    async def get(self, node_id):
        return None

    def get_local(self, node_id):
        return None

    async def set(self, node_id, value):
        pass

    def set_local(self, node_id, value):
        pass

    async def ensure_subcache_for(self, node_id, children_ids):
        return self

class LRUCache(BasicCache):
    def __init__(self, key_class, max_size=100, enable_providers=False):
        super().__init__(key_class, enable_providers=enable_providers)
        self.max_size = max_size
        self.min_generation = 0
        self.generation = 0
        self.used_generation = {}
        self.children = {}

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        await super().set_prompt(dynprompt, node_ids, is_changed_cache)
        self.generation += 1
        for node_id in node_ids:
            self._mark_used(node_id)

    def clean_unused(self):
        while len(self.cache) > self.max_size and self.min_generation < self.generation:
            self.min_generation += 1
            to_remove = [key for key in self.cache if self.used_generation[key] < self.min_generation]
            for key in to_remove:
                del self.cache[key]
                del self.used_generation[key]
                if key in self.children:
                    del self.children[key]
        self._clean_subcaches()

    async def get(self, node_id):
        self._mark_used(node_id)
        return await self._get_immediate(node_id)

    def _mark_used(self, node_id):
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key is not None:
            self.used_generation[cache_key] = self.generation

    async def set(self, node_id, value):
        self._mark_used(node_id)
        return await self._set_immediate(node_id, value)

    def set_local(self, node_id, value):
        self._mark_used(node_id)
        BasicCache.set_local(self, node_id, value)

    async def ensure_subcache_for(self, node_id, children_ids):
        # Just uses subcaches for tracking 'live' nodes
        await super()._ensure_subcache(node_id, children_ids)

        await self.cache_key_set.add_keys(children_ids)
        self._mark_used(node_id)
        cache_key = self.cache_key_set.get_data_key(node_id)
        self.children[cache_key] = []
        for child_id in children_ids:
            self._mark_used(child_id)
            self.children[cache_key].append(self.cache_key_set.get_data_key(child_id))
        return self


#Small baseline weight used when a cache entry has no measurable CPU tensors.
#Keeps unknown-sized entries in eviction scoring without dominating tensor-backed entries.

RAM_CACHE_DEFAULT_RAM_USAGE = 0.05

#Exponential bias towards evicting older workflows so garbage will be taken out
#in constantly changing setups.

RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER = 1.3

RAM_CACHE_LARGE_INTERMEDIATE = 512 * 1024 ** 2


def all_outputs_dynamic(outputs):
    if outputs is None:
        return False

    for output in outputs:
        if isinstance(output, (list, tuple)):
            if not all_outputs_dynamic(output):
                return False
        elif not hasattr(output, "is_dynamic") or not output.is_dynamic():
            return False

    return True

class RAMPressureCache(LRUCache):

    def __init__(self, key_class, enable_providers=False):
        super().__init__(key_class, 0, enable_providers=enable_providers)
        self.timestamps = {}
        self.active_evictions = False
        self.full_evictions = False

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        self.active_evictions = False
        self.full_evictions = False
        await super().set_prompt(dynprompt, node_ids, is_changed_cache)

    def clean_unused(self):
        self._clean_subcaches()

    async def set(self, node_id, value):
        self.timestamps[self.cache_key_set.get_data_key(node_id)] = time.time()
        await super().set(node_id, value)

    async def get(self, node_id):
        self.timestamps[self.cache_key_set.get_data_key(node_id)] = time.time()
        return await super().get(node_id)

    def set_local(self, node_id, value):
        self.timestamps[self.cache_key_set.get_data_key(node_id)] = time.time()
        super().set_local(node_id, value)

    def ram_release(self, target, free_active=False, min_entry_size=0):
        if psutil.virtual_memory().available >= target:
            return 0

        clean_list = []

        for key, cache_entry in self.cache.items():
            if not free_active and self.used_generation[key] == self.generation:
                continue

            if all_outputs_dynamic(cache_entry.outputs) and self.used_generation[key] == self.generation:
                continue

            oom_score = RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER ** (self.generation - self.used_generation[key])

            ram_usage = RAM_CACHE_DEFAULT_RAM_USAGE
            oom_ram_usage = ram_usage
            def scan_list_for_ram_usage(outputs):
                nonlocal ram_usage, oom_ram_usage
                if outputs is None:
                    return
                for output in outputs:
                    if isinstance(output, (list, tuple)):
                        scan_list_for_ram_usage(output)
                    elif isinstance(output, torch.Tensor) and output.device.type == 'cpu':
                        ram_usage += output.numel() * output.element_size()
                        oom_ram_usage += output.numel() * output.element_size()
                    elif is_model_patcher_output(output) and self.used_generation[key] != self.generation:
                        #old ModelPatchers are the first to go
                        oom_ram_usage = 1e30
            scan_list_for_ram_usage(cache_entry.outputs)

            if ram_usage < min_entry_size:
                continue

            oom_score *= oom_ram_usage
            #In the case where we have no information on the node ram usage at all,
            #break OOM score ties on the last touch timestamp (pure LRU)
            bisect.insort(clean_list, (oom_score, self.timestamps[key], key, ram_usage))

        freed = 0
        while psutil.virtual_memory().available < target and clean_list:
            _, _, key, ram_usage = clean_list.pop()
            del self.cache[key]
            self.used_generation.pop(key, None)
            self.timestamps.pop(key, None)
            self.children.pop(key, None)
            freed += ram_usage
        if freed and free_active:
            self.active_evictions = True
            if min_entry_size == 0:
                self.full_evictions = True
        return freed

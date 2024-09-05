import itertools
from typing import Sequence, Mapping
from comfy_execution.graph import DynamicPrompt

import nodes

from comfy_execution.graph_utils import is_link

class CacheKeySet:
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        self.keys = {}
        self.subcache_keys = {}

    def add_keys(self, node_ids):
        raise NotImplementedError()

    def all_node_ids(self):
        return set(self.keys.keys())

    def get_used_keys(self):
        return self.keys.values()

    def get_used_subcache_keys(self):
        return self.subcache_keys.values()

    def get_data_key(self, node_id):
        return self.keys.get(node_id, None)

    def get_subcache_key(self, node_id):
        return self.subcache_keys.get(node_id, None)

class Unhashable:
    def __init__(self):
        self.value = float("NaN")

def to_hashable(obj):
    # So that we don't infinitely recurse since frozenset and tuples
    # are Sequences.
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, Mapping):
        return frozenset([(to_hashable(k), to_hashable(v)) for k, v in sorted(obj.items())])
    elif isinstance(obj, Sequence):
        return frozenset(zip(itertools.count(), [to_hashable(i) for i in obj]))
    else:
        # TODO - Support other objects like tensors?
        return Unhashable()

class CacheKeySetID(CacheKeySet):
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        super().__init__(dynprompt, node_ids, is_changed_cache)
        self.dynprompt = dynprompt
        self.add_keys(node_ids)

    def add_keys(self, node_ids):
        for node_id in node_ids:
            if node_id in self.keys:
                continue
            if not self.dynprompt.has_node(node_id):
                continue
            node = self.dynprompt.get_node(node_id)
            self.keys[node_id] = (node_id, node["class_type"])
            self.subcache_keys[node_id] = (node_id, node["class_type"])

class CacheKeySetInputSignature(CacheKeySet):
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        super().__init__(dynprompt, node_ids, is_changed_cache)
        self.dynprompt = dynprompt
        self.is_changed_cache = is_changed_cache
        self.add_keys(node_ids)

    def include_node_id_in_input(self) -> bool:
        return False

    def add_keys(self, node_ids):
        for node_id in node_ids:
            if node_id in self.keys:
                continue
            if not self.dynprompt.has_node(node_id):
                continue
            node = self.dynprompt.get_node(node_id)
            self.keys[node_id] = self.get_node_signature(self.dynprompt, node_id)
            self.subcache_keys[node_id] = (node_id, node["class_type"])

    def get_node_signature(self, dynprompt, node_id):
        signature = []
        ancestors, order_mapping = self.get_ordered_ancestry(dynprompt, node_id)
        signature.append(self.get_immediate_node_signature(dynprompt, node_id, order_mapping))
        for ancestor_id in ancestors:
            signature.append(self.get_immediate_node_signature(dynprompt, ancestor_id, order_mapping))
        return to_hashable(signature)

    def get_immediate_node_signature(self, dynprompt, node_id, ancestor_order_mapping):
        if not dynprompt.has_node(node_id):
            # This node doesn't exist -- we can't cache it.
            return [float("NaN")]
        node = dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        signature = [class_type, self.is_changed_cache.get(node_id)]
        if self.include_node_id_in_input() or (hasattr(class_def, "NOT_IDEMPOTENT") and class_def.NOT_IDEMPOTENT) or "UNIQUE_ID" in class_def.INPUT_TYPES().get("hidden", {}).values():
            signature.append(node_id)
        inputs = node["inputs"]
        for key in sorted(inputs.keys()):
            if is_link(inputs[key]):
                (ancestor_id, ancestor_socket) = inputs[key]
                ancestor_index = ancestor_order_mapping[ancestor_id]
                signature.append((key,("ANCESTOR", ancestor_index, ancestor_socket)))
            else:
                signature.append((key, inputs[key]))
        return signature

    # This function returns a list of all ancestors of the given node. The order of the list is
    # deterministic based on which specific inputs the ancestor is connected by.
    def get_ordered_ancestry(self, dynprompt, node_id):
        ancestors = []
        order_mapping = {}
        self.get_ordered_ancestry_internal(dynprompt, node_id, ancestors, order_mapping)
        return ancestors, order_mapping

    def get_ordered_ancestry_internal(self, dynprompt, node_id, ancestors, order_mapping):
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
    def __init__(self, key_class):
        self.key_class = key_class
        self.initialized = False
        self.dynprompt: DynamicPrompt
        self.cache_key_set: CacheKeySet
        self.cache = {}
        self.subcaches = {}

    def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        self.dynprompt = dynprompt
        self.cache_key_set = self.key_class(dynprompt, node_ids, is_changed_cache)
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

    def _set_immediate(self, node_id, value):
        assert self.initialized
        cache_key = self.cache_key_set.get_data_key(node_id)
        self.cache[cache_key] = value

    def _get_immediate(self, node_id):
        if not self.initialized:
            return None
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key in self.cache:
            return self.cache[cache_key]
        else:
            return None

    def _ensure_subcache(self, node_id, children_ids):
        subcache_key = self.cache_key_set.get_subcache_key(node_id)
        subcache = self.subcaches.get(subcache_key, None)
        if subcache is None:
            subcache = BasicCache(self.key_class)
            self.subcaches[subcache_key] = subcache
        subcache.set_prompt(self.dynprompt, children_ids, self.is_changed_cache)
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
    def __init__(self, key_class):
        super().__init__(key_class)

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

    def get(self, node_id):
        cache = self._get_cache_for(node_id)
        if cache is None:
            return None
        return cache._get_immediate(node_id)

    def set(self, node_id, value):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        cache._set_immediate(node_id, value)

    def ensure_subcache_for(self, node_id, children_ids):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        return cache._ensure_subcache(node_id, children_ids)

class LRUCache(BasicCache):
    def __init__(self, key_class, max_size=100):
        super().__init__(key_class)
        self.max_size = max_size
        self.min_generation = 0
        self.generation = 0
        self.used_generation = {}
        self.children = {}

    def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        super().set_prompt(dynprompt, node_ids, is_changed_cache)
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

    def get(self, node_id):
        self._mark_used(node_id)
        return self._get_immediate(node_id)

    def _mark_used(self, node_id):
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key is not None:
            self.used_generation[cache_key] = self.generation

    def set(self, node_id, value):
        self._mark_used(node_id)
        return self._set_immediate(node_id, value)

    def ensure_subcache_for(self, node_id, children_ids):
        # Just uses subcaches for tracking 'live' nodes
        super()._ensure_subcache(node_id, children_ids)

        self.cache_key_set.add_keys(children_ids)
        self._mark_used(node_id)
        cache_key = self.cache_key_set.get_data_key(node_id)
        self.children[cache_key] = []
        for child_id in children_ids:
            self._mark_used(child_id)
            self.children[cache_key].append(self.cache_key_set.get_data_key(child_id))
        return self


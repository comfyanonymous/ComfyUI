import bisect
import gc
import itertools
import psutil
import time
import torch
from typing import Sequence, Mapping, Dict
from comfy_execution.graph import DynamicPrompt
from abc import ABC, abstractmethod

import nodes

from comfy_execution.graph_utils import is_link

NODE_CLASS_CONTAINS_UNIQUE_ID: Dict[str, bool] = {}


def include_unique_id_in_input(class_type: str) -> bool:
    if class_type in NODE_CLASS_CONTAINS_UNIQUE_ID:
        return NODE_CLASS_CONTAINS_UNIQUE_ID[class_type]
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    NODE_CLASS_CONTAINS_UNIQUE_ID[class_type] = "UNIQUE_ID" in class_def.INPUT_TYPES().get("hidden", {}).values()
    return NODE_CLASS_CONTAINS_UNIQUE_ID[class_type]

class CacheKeySet(ABC):
    def __init__(self, dynprompt, node_ids, is_changed):
        self.keys = {}
        self.subcache_keys = {}

    @abstractmethod
    async def add_keys(self, node_ids):
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
    
    async def update_cache_key(self, node_id) -> None:
        pass

    def is_key_updated(self, node_id) -> bool:
        return True

class Unhashable:
    def __init__(self):
        self.value = float("NaN")

def to_hashable(obj):
    # So that we don't infinitely recurse since frozenset and tuples
    # are Sequences.
    if isinstance(obj, (int, float, str, bool, bytes, type(None))):
        return obj
    elif isinstance(obj, Mapping):
        return frozenset([(to_hashable(k), to_hashable(v)) for k, v in sorted(obj.items())])
    elif isinstance(obj, Sequence):
        return frozenset(zip(itertools.count(), [to_hashable(i) for i in obj]))
    else:
        # TODO - Support other objects like tensors?
        return Unhashable()
    
def throw_on_unhashable(obj):
    # Same as to_hashable except throwing for unhashables instead.
    if isinstance(obj, (int, float, str, bool, bytes, type(None))):
        return obj
    elif isinstance(obj, Mapping):
        return frozenset([(throw_on_unhashable(k), throw_on_unhashable(v)) for k, v in sorted(obj.items())])
    elif isinstance(obj, Sequence):
        return frozenset(zip(itertools.count(), [throw_on_unhashable(i) for i in obj]))
    else:
        raise Exception("Object unhashable.")

class CacheKeySetID(CacheKeySet):
    def __init__(self, dynprompt, node_ids, is_changed):
        super().__init__(dynprompt, node_ids, is_changed)
        self.dynprompt = dynprompt

    async def add_keys(self, node_ids):
        for node_id in node_ids:
            if node_id in self.keys:
                continue
            if not self.dynprompt.has_node(node_id):
                continue
            node = self.dynprompt.get_node(node_id)
            self.keys[node_id] = (node_id, node["class_type"])
            self.subcache_keys[node_id] = (node_id, node["class_type"])

class CacheKeySetInputSignature(CacheKeySet):
    def __init__(self, dynprompt, node_ids, is_changed):
        super().__init__(dynprompt, node_ids, is_changed)
        self.dynprompt = dynprompt
        self.is_changed = is_changed
        self.updated_node_ids = set()

    def include_node_id_in_input(self) -> bool:
        return False
    
    async def update_cache_key(self, node_id):
        if node_id in self.updated_node_ids:
            return
        if node_id not in self.keys:
            return
        self.updated_node_ids.add(node_id)
        self.keys[node_id] = await self.get_node_signature(node_id)

    def is_key_updated(self, node_id):
        return node_id in self.updated_node_ids

    async def add_keys(self, node_ids):
        for node_id in node_ids:
            if node_id in self.keys:
                continue
            if not self.dynprompt.has_node(node_id):
                continue
            node = self.dynprompt.get_node(node_id)
            self.keys[node_id] = None
            self.subcache_keys[node_id] = (node_id, node["class_type"])

    async def get_node_signature(self, node_id):
        signatures = []
        ancestors, order_mapping, node_inputs = self.get_ordered_ancestry(node_id)
        node = self.dynprompt.get_node(node_id)
        node["signature"] = to_hashable(await self.get_immediate_node_signature(node_id, order_mapping, node_inputs))
        signatures.append(node["signature"])
        for ancestor_id in ancestors:
            ancestor_node = self.dynprompt.get_node(ancestor_id)
            assert "signature" in ancestor_node
            signatures.append(ancestor_node["signature"])
        signatures = frozenset(zip(itertools.count(), signatures))
        return signatures

    async def get_immediate_node_signature(self, node_id, ancestor_order_mapping, inputs):
        if not self.dynprompt.has_node(node_id):
            # This node doesn't exist -- we can't cache it.
            return [float("NaN")]
        node = self.dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        signature = [class_type, await self.is_changed.get(node_id)]
        for key in sorted(inputs.keys()):
            if is_link(inputs[key]):
                (ancestor_id, ancestor_socket) = inputs[key]
                ancestor_index = ancestor_order_mapping[ancestor_id]
                signature.append((key,("ANCESTOR", ancestor_index, ancestor_socket)))
            else:
                signature.append((key, inputs[key]))
        if self.include_node_id_in_input() or (hasattr(class_def, "NOT_IDEMPOTENT") and class_def.NOT_IDEMPOTENT) or include_unique_id_in_input(class_type):
            signature.append(node_id)
        return signature
    
    def get_ordered_ancestry(self, node_id):
        def get_ancestors(ancestors, ret: list=[]):
            for ancestor_id in ancestors:
                if ancestor_id not in ret:
                    ret.append(ancestor_id)
                ancestor_node = self.dynprompt.get_node(ancestor_id)
                get_ancestors(ancestor_node["ancestors"], ret)
            return ret
        
        ancestors, node_inputs = self.get_ordered_ancestry_internal(node_id)
        ancestors = get_ancestors(ancestors)

        order_mapping = {}
        for i, ancestor_id in enumerate(ancestors):
            order_mapping[ancestor_id] = i
        
        return ancestors, order_mapping, node_inputs

    def get_ordered_ancestry_internal(self, node_id):
        def get_hashable(obj):
            try:
                return throw_on_unhashable(obj)
            except:
                return Unhashable
        
        ancestors = []
        node_inputs = {}

        if not self.dynprompt.has_node(node_id):
            return ancestors, node_inputs
        
        node = self.dynprompt.get_node(node_id)
        if "ancestors" in node:
            return node["ancestors"], node_inputs
        
        input_data_all, _, _ = self.is_changed.get_input_data(node_id)
        inputs = self.dynprompt.get_node(node_id)["inputs"]
        for key in sorted(inputs.keys()):
            if key in input_data_all:
                if is_link(inputs[key]):
                    ancestor_id = inputs[key][0]
                    hashable = get_hashable(input_data_all[key])
                    if hashable is Unhashable or (input_data_all[key] and is_link(input_data_all[key][0])):
                        # Link still needed
                        node_inputs[key] = inputs[key]
                        if ancestor_id not in ancestors:
                            ancestors.append(ancestor_id)
                    else:
                        # Replace link
                        node_inputs[key] = input_data_all[key]
                else:
                    hashable = get_hashable(inputs[key])
                    if hashable is Unhashable:
                        node_inputs[key] = Unhashable()
                    else:
                        node_inputs[key] = [inputs[key]]
        
        node["ancestors"] = ancestors
        return node["ancestors"], node_inputs

class BasicCache:
    def __init__(self, key_class):
        self.key_class = key_class
        self.initialized = False
        self.dynprompt: DynamicPrompt
        self.cache_key_set: CacheKeySet
        self.cache = {}
        self.subcaches = {}

    async def set_prompt(self, dynprompt, node_ids, is_changed):
        self.dynprompt = dynprompt
        self.cache_key_set = self.key_class(dynprompt, node_ids, is_changed)
        await self.cache_key_set.add_keys(node_ids)
        self.is_changed = is_changed
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
            else:
                self.subcaches[key].clean_unused()
        for key in to_remove:
            del self.subcaches[key]

    def clean_unused(self):
        assert self.initialized
        self._clean_cache()
        self._clean_subcaches()

    def poll(self, **kwargs):
        pass

    async def _update_cache_key_immediate(self, node_id):
        await self.cache_key_set.update_cache_key(node_id)
    
    def _is_key_updated_immediate(self, node_id):
        return self.cache_key_set.is_key_updated(node_id)

    def _set_immediate(self, node_id, value):
        assert self.initialized
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key is not None:
            self.cache[cache_key] = value

    def _get_immediate(self, node_id):
        if not self.initialized:
            return None
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]
        else:
            return None

    async def _ensure_subcache(self, node_id, children_ids):
        subcache_key = self.cache_key_set.get_subcache_key(node_id)
        subcache = self.subcaches.get(subcache_key, None)
        if subcache is None:
            subcache = BasicCache(self.key_class)
            self.subcaches[subcache_key] = subcache
        await subcache.set_prompt(self.dynprompt, children_ids, self.is_changed)
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

    async def ensure_subcache_for(self, node_id, children_ids):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        return await cache._ensure_subcache(node_id, children_ids)
    
    async def update_cache_key(self, node_id):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        await cache._update_cache_key_immediate(node_id)
        
    def is_key_updated(self, node_id):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        return cache._is_key_updated_immediate(node_id)

class NullCache:

    async def set_prompt(self, dynprompt, node_ids, is_changed):
        pass

    def all_node_ids(self):
        return []

    def clean_unused(self):
        pass

    def poll(self, **kwargs):
        pass

    def get(self, node_id):
        return None

    def set(self, node_id, value):
        pass

    async def ensure_subcache_for(self, node_id, children_ids):
        return self
    
    async def update_cache_key(self, node_id):
        pass
        
    def is_key_updated(self, node_id):
        return True

class LRUCache(BasicCache):
    def __init__(self, key_class, max_size=100):
        super().__init__(key_class)
        self.max_size = max_size
        self.min_generation = 0
        self.generation = 0
        self.used_generation = {}
        self.children = {}

    async def set_prompt(self, dynprompt, node_ids, is_changed):
        await super().set_prompt(dynprompt, node_ids, is_changed)
        self.generation += 1
        for node_id in node_ids:
            self._mark_used(node_id)

    def clean_unused(self):
        while len(self.cache) > self.max_size and self.min_generation < self.generation:
            self.min_generation += 1
            to_remove = [key for key in self.cache if key not in self.used_generation or self.used_generation[key] < self.min_generation]
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
    
    async def update_cache_key(self, node_id):
        await self._update_cache_key_immediate(node_id)
        self._mark_used(node_id)
        
    def is_key_updated(self, node_id):
        self._mark_used(node_id)
        return self._is_key_updated_immediate(node_id)


#Iterating the cache for usage analysis might be expensive, so if we trigger make sure
#to take a chunk out to give breathing space on high-node / low-ram-per-node flows.

RAM_CACHE_HYSTERESIS = 1.1

#This is kinda in GB but not really. It needs to be non-zero for the below heuristic
#and as long as Multi GB models dwarf this it will approximate OOM scoring OK

RAM_CACHE_DEFAULT_RAM_USAGE = 0.1

#Exponential bias towards evicting older workflows so garbage will be taken out
#in constantly changing setups.

RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER = 1.3

class RAMPressureCache(LRUCache):

    def __init__(self, key_class):
        super().__init__(key_class, 0)
        self.timestamps = {}

    def clean_unused(self):
        self._clean_subcaches()

    def set(self, node_id, value):
        self.timestamps[self.cache_key_set.get_data_key(node_id)] = time.time()
        super().set(node_id, value)

    def get(self, node_id):
        self.timestamps[self.cache_key_set.get_data_key(node_id)] = time.time()
        return super().get(node_id)

    def poll(self, ram_headroom):
        def _ram_gb():
            return psutil.virtual_memory().available / (1024**3)

        if _ram_gb() > ram_headroom:
            return
        gc.collect()
        if _ram_gb() > ram_headroom:
            return

        clean_list = []

        for key, (outputs, _), in self.cache.items():
            oom_score =  RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER ** (self.generation - self.used_generation[key])

            ram_usage = RAM_CACHE_DEFAULT_RAM_USAGE
            def scan_list_for_ram_usage(outputs):
                nonlocal ram_usage
                if outputs is None:
                    return
                for output in outputs:
                    if isinstance(output, list):
                        scan_list_for_ram_usage(output)
                    elif isinstance(output, torch.Tensor) and output.device.type == 'cpu':
                        #score Tensors at a 50% discount for RAM usage as they are likely to
                        #be high value intermediates
                        ram_usage += (output.numel() * output.element_size()) * 0.5
                    elif hasattr(output, "get_ram_usage"):
                        ram_usage += output.get_ram_usage()
            scan_list_for_ram_usage(outputs)

            oom_score *= ram_usage
            #In the case where we have no information on the node ram usage at all,
            #break OOM score ties on the last touch timestamp (pure LRU)
            bisect.insort(clean_list, (oom_score, self.timestamps[key], key))

        while _ram_gb() < ram_headroom * RAM_CACHE_HYSTERESIS and clean_list:
            _, _, key = clean_list.pop()
            del self.cache[key]
            gc.collect()

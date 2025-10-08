import itertools
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
    def __init__(self, dynprompt, node_ids, is_changed_cache):
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
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        super().__init__(dynprompt, node_ids, is_changed_cache)
        self.dynprompt = dynprompt
        self.is_changed_cache = is_changed_cache

    def include_node_id_in_input(self) -> bool:
        return False

    async def add_keys(self, node_ids):
        for node_id in node_ids:
            if node_id in self.keys:
                continue
            if not self.dynprompt.has_node(node_id):
                continue
            node = self.dynprompt.get_node(node_id)
            self.keys[node_id] = await self.get_node_signature(self.dynprompt, node_id)
            self.subcache_keys[node_id] = (node_id, node["class_type"])

    async def get_node_signature(self, dynprompt, node_id):
        signature = []
        ancestors, order_mapping = self.get_ordered_ancestry(dynprompt, node_id)
        signature.append(await self.get_immediate_node_signature(dynprompt, node_id, order_mapping))
        for ancestor_id in ancestors:
            signature.append(await self.get_immediate_node_signature(dynprompt, ancestor_id, order_mapping))
        return to_hashable(signature)

    async def get_immediate_node_signature(self, dynprompt, node_id, ancestor_order_mapping):
        if not dynprompt.has_node(node_id):
            # This node doesn't exist -- we can't cache it.
            return [float("NaN")]
        node = dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        signature = [class_type, await self.is_changed_cache.get(node_id)]
        if self.include_node_id_in_input() or (hasattr(class_def, "NOT_IDEMPOTENT") and class_def.NOT_IDEMPOTENT) or include_unique_id_in_input(class_type):
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

class LRUCache(BasicCache):
    def __init__(self, key_class, max_size=100):
        super().__init__(key_class)
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


class DependencyAwareCache(BasicCache):
    """
    A cache implementation that tracks dependencies between nodes and manages
    their execution and caching accordingly. It extends the BasicCache class.
    Nodes are removed from this cache once all of their descendants have been
    executed.
    """

    def __init__(self, key_class):
        """
        Initialize the DependencyAwareCache.

        Args:
            key_class: The class used for generating cache keys.
        """
        super().__init__(key_class)
        self.descendants = {}  # Maps node_id -> set of descendant node_ids
        self.ancestors = {}    # Maps node_id -> set of ancestor node_ids
        self.executed_nodes = set()  # Tracks nodes that have been executed

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        """
        Clear the entire cache and rebuild the dependency graph.

        Args:
            dynprompt: The dynamic prompt object containing node information.
            node_ids: List of node IDs to initialize the cache for.
            is_changed_cache: Flag indicating if the cache has changed.
        """
        # Clear all existing cache data
        self.cache.clear()
        self.subcaches.clear()
        self.descendants.clear()
        self.ancestors.clear()
        self.executed_nodes.clear()

        # Call the parent method to initialize the cache with the new prompt
        await super().set_prompt(dynprompt, node_ids, is_changed_cache)

        # Rebuild the dependency graph
        self._build_dependency_graph(dynprompt, node_ids)

    def _build_dependency_graph(self, dynprompt, node_ids):
        """
        Build the dependency graph for all nodes.

        Args:
            dynprompt: The dynamic prompt object containing node information.
            node_ids: List of node IDs to build the graph for.
        """
        self.descendants.clear()
        self.ancestors.clear()
        for node_id in node_ids:
            self.descendants[node_id] = set()
            self.ancestors[node_id] = set()

        for node_id in node_ids:
            inputs = dynprompt.get_node(node_id)["inputs"]
            for input_data in inputs.values():
                if is_link(input_data):  # Check if the input is a link to another node
                    ancestor_id = input_data[0]
                    self.descendants[ancestor_id].add(node_id)
                    self.ancestors[node_id].add(ancestor_id)

    def set(self, node_id, value):
        """
        Mark a node as executed and store its value in the cache.

        Args:
            node_id: The ID of the node to store.
            value: The value to store for the node.
        """
        self._set_immediate(node_id, value)
        self.executed_nodes.add(node_id)
        self._cleanup_ancestors(node_id)

    def get(self, node_id):
        """
        Retrieve the cached value for a node.

        Args:
            node_id: The ID of the node to retrieve.

        Returns:
            The cached value for the node.
        """
        return self._get_immediate(node_id)

    async def ensure_subcache_for(self, node_id, children_ids):
        """
        Ensure a subcache exists for a node and update dependencies.

        Args:
            node_id: The ID of the parent node.
            children_ids: List of child node IDs to associate with the parent node.

        Returns:
            The subcache object for the node.
        """
        subcache = await super()._ensure_subcache(node_id, children_ids)
        for child_id in children_ids:
            self.descendants[node_id].add(child_id)
            self.ancestors[child_id].add(node_id)
        return subcache

    def _cleanup_ancestors(self, node_id):
        """
        Check if ancestors of a node can be removed from the cache.

        Args:
            node_id: The ID of the node whose ancestors are to be checked.
        """
        for ancestor_id in self.ancestors.get(node_id, []):
            if ancestor_id in self.executed_nodes:
                # Remove ancestor if all its descendants have been executed
                if all(descendant in self.executed_nodes for descendant in self.descendants[ancestor_id]):
                    self._remove_node(ancestor_id)

    def _remove_node(self, node_id):
        """
        Remove a node from the cache.

        Args:
            node_id: The ID of the node to remove.
        """
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key in self.cache:
            del self.cache[cache_key]
        subcache_key = self.cache_key_set.get_subcache_key(node_id)
        if subcache_key in self.subcaches:
            del self.subcaches[subcache_key]

    def clean_unused(self):
        """
        Clean up unused nodes. This is a no-op for this cache implementation.
        """
        pass

    def recursive_debug_dump(self):
        """
        Dump the cache and dependency graph for debugging.

        Returns:
            A list containing the cache state and dependency graph.
        """
        result = super().recursive_debug_dump()
        result.append({
            "descendants": self.descendants,
            "ancestors": self.ancestors,
            "executed_nodes": list(self.executed_nodes),
        })
        return result

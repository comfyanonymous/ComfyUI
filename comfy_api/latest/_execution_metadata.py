"""Prompt-owned observations and explicitly scoped shared PNG metadata writes."""
import asyncio
import contextvars
import math
import threading
import json
import uuid
import weakref


MAX_METADATA_BYTES = 1024 * 1024
_active = contextvars.ContextVar("comfy_execution_metadata", default=None)


def json_copy(value):
    parts = []
    total = 0
    for part in json.JSONEncoder(separators=(",", ":")).iterencode(value):
        total += len(part.encode("utf-8"))
        if total > MAX_METADATA_BYTES:
            raise ValueError("execution metadata exceeds 1 MiB")
        parts.append(part)
    return json.loads("".join(parts))


def current(prompt_id):
    owner = _active.get()
    return owner if owner is not None and owner.prompt_id == str(prompt_id) else None


class ExecutionMetadataScope:
    def __init__(self, prompt_id, dynamic_prompt, extra_pnginfo=None, *, observe=False):
        self.prompt_id = str(prompt_id)
        self.dynamic_prompt = dynamic_prompt
        self.extra_pnginfo = extra_pnginfo
        self.observe = observe
        self.closed = False
        self._write_lock = threading.RLock()
        self.execution_order = []
        self.observations = {}
        self._record_sizes = {}
        self._record_bytes = 0
        self._identities = {}
        self._namespace = uuid.uuid4().hex
        self._next_identity = 0
        self._error = None
        self._token = None
        self._projection_bytes = 0

    def __enter__(self):
        self._token = _active.set(self)
        return self

    def __exit__(self, *_exc):
        self.close()
        _active.reset(self._token)

    def close(self):
        with self._write_lock:
            self.closed = True
        self._identities.clear()
        self.observations.clear()
        self.execution_order.clear()
        self._record_sizes.clear()
        self.dynamic_prompt = None
        self.extra_pnginfo = None

    def _identity(self, value, *, conditioning=False):
        key = id(value)
        prior = self._identities.get(key)
        if prior is not None:
            token, reference, borrowed = prior
            if (reference if borrowed else reference()) is value:
                return token
        self._next_identity += 1
        token = f"{self._namespace}:{self._next_identity}"
        if conditioning:
            # Native prompt tracing compares conditioning-container identity.
            # Borrow only these containers, for this prompt, without tensor copies.
            self._identities[key] = token, value, True
        else:
            try:
                self._identities[key] = token, weakref.ref(value), False
            except TypeError:
                return None
        return token

    def _project(self, value, *, hint=None, depth=0):
        if depth > 32:
            raise ValueError("execution observation nesting exceeds 32 levels")
        exact = type(value)
        if value is None or exact in {bool, int, float, str}:
            if exact is str and len(value) > MAX_METADATA_BYTES:
                raise ValueError("execution metadata string exceeds 1 MiB")
            self._spend(len(json.dumps(value).encode("utf-8")))
            return value
        if exact in {list, tuple, dict}:
            conditioning = hint == "CONDITIONING"
            if exact is list and value and type(value[0]) in {list, tuple} and len(value[0]) == 2:
                import torch

                conditioning = conditioning or (
                    isinstance(value[0][0], torch.Tensor) and type(value[0][1]) is dict)
            result = {"kind": exact.__name__}
            if conditioning:
                result["identity"] = self._identity(value, conditioning=True)
            self._spend(len(json.dumps(result).encode("utf-8")) + 12 + len(value))
            if exact is dict:
                result["items"] = [
                    [self._project(key, depth=depth + 1), self._project(item, depth=depth + 1)]
                    for key, item in value.items()
                ]
            else:
                result["items"] = [self._project(item, depth=depth + 1) for item in value]
            return result

        import torch
        from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise

        if isinstance(value, torch.Tensor):
            result = {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
            self._spend(len(json.dumps(result).encode("utf-8")))
            return result
        if exact in {Noise_EmptyNoise, Noise_RandomNoise}:
            seed = object.__getattribute__(value, "__dict__")["seed"]
            result = {"kind": "noise", "identity": self._identity(value), "seed": self._project(seed)}
        else:
            result = {"kind": "opaque", "declared_type": hint, "identity": self._identity(value)}
        self._spend(len(json.dumps(result).encode("utf-8")))
        return result

    def _spend(self, size):
        self._projection_bytes += size
        if self._projection_bytes > MAX_METADATA_BYTES:
            raise ValueError("execution observation exceeds 1 MiB")

    def _store(self, node_id, record):
        detached = json_copy(record)
        size = len(json.dumps(detached, separators=(",", ":")).encode("utf-8"))
        total = self._record_bytes - self._record_sizes.get(node_id, 0) + size
        if total > MAX_METADATA_BYTES:
            raise ValueError("execution observations exceed 1 MiB")
        self.observations[node_id] = detached
        self._record_sizes[node_id] = size
        self._record_bytes = total

    def _fail(self, error):
        self._error = f"{type(error).__name__}: {error}"
        self._identities.clear()
        self.observations.clear()
        self._record_sizes.clear()
        self._record_bytes = 0

    def _record(self, node_id, class_def, inputs, outputs=None, *, cached=False):
        if not self.observe or self.closed or self._error:
            return
        try:
            self._projection_bytes = 0
            node = self.dynamic_prompt.get_node(node_id)
            schema = class_def.INPUT_TYPES()
            from comfy_api.internal import _ComfyNodeInternal
            from . import _io

            if issubclass(class_def, _ComfyNodeInternal):
                schema, _hidden, _v3_data = _io.get_finalized_class_inputs(schema, node.get("inputs", {}))
            public = {**schema.get("required", {}), **schema.get("optional", {})}
            return_types = tuple(getattr(class_def, "RETURN_TYPES", ()) or ())
            record = self.observations.get(node_id, {}).copy()
            record.update({
                "class_type": node["class_type"],
                "class_name": class_def.__name__,
                "return_types": [str(item) for item in return_types],
                "cached": cached,
                "parent_node_id": self.dynamic_prompt.get_parent_node_id(node_id),
            })
            if inputs is not None:
                record["inputs"] = {
                    name: [self._project(item, hint=str(public[name][0])) for item in values]
                    for name, values in inputs.items() if name in public
                }
            if outputs is not None:
                record["outputs"] = [
                    [self._project(item, hint=str(return_types[index]) if index < len(return_types) else None)
                     for item in values]
                    for index, values in enumerate(outputs)
                ]
            self._store(node_id, record)
        except Exception as error:
            # Observation failure does not change unrelated node computation.
            # A consumer requesting these observations receives the failure.
            self._fail(error)

    def record_start(self, node_id, class_def, inputs):
        if not self.observe or self.closed:
            return
        if node_id not in self.execution_order:
            self.execution_order.append(node_id)
        self._record(node_id, class_def, inputs)

    def record_outputs(self, node_id, class_def, outputs):
        self._record(node_id, class_def, None, outputs)

    def record_cached(self, node_id, class_def, inputs, outputs):
        self._record(node_id, class_def, inputs, outputs, cached=True)

    def snapshot(self, node_id, *, include_observations=False):
        if self.closed:
            raise RuntimeError("execution metadata owner is not current")
        if type(include_observations) is not bool:
            raise TypeError("include_observations must be a boolean")
        self.dynamic_prompt.get_node(node_id)
        if include_observations and (not self.observe or self._error):
            raise RuntimeError(self._error or "execution observations were not collected")
        result = {
            "prompt_id": self.prompt_id,
            "node_id": node_id,
            "prompt": self.dynamic_prompt.original_prompt,
            "nodes": {key: self.dynamic_prompt.get_node(key) for key in self.dynamic_prompt.all_node_ids()},
            "extra_pnginfo": self.extra_pnginfo,
            "parent_node_id": self.dynamic_prompt.get_parent_node_id(node_id),
        }
        if include_observations:
            result["execution_order"] = self.execution_order
            result["observations"] = self.observations
        return json_copy(result)


def _bounded_pnginfo_json(value):
    remaining = 4096
    def check(item, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 32:
            raise ValueError("PNG metadata exceeds 4096 values or 32 nesting levels")
        exact = type(item)
        if item is None or exact in {bool, int, str}:
            if exact is str and len(item) > MAX_METADATA_BYTES:
                raise ValueError("PNG metadata string exceeds 1 MiB")
            return
        if exact is float and math.isfinite(item):
            return
        if exact is dict:
            for key, child in item.items():
                if type(key) is not str or len(key) > 1024:
                    raise TypeError("PNG metadata keys must be strings of at most 1024 characters")
                check(child, depth + 1)
            return
        if exact is list:
            for child in item:
                check(child, depth + 1)
            return
        raise TypeError("PNG metadata accepts only finite JSON values")
    check(value)
    parts = []
    size = 0
    for part in json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":")).iterencode(value):
        size += len(part.encode("utf-8"))
        if size > MAX_METADATA_BYTES:
            raise ValueError("PNG metadata exceeds 1 MiB")
        parts.append(part)
    return "".join(parts)


def encode_pnginfo_update(updates, expected=None):
    if type(updates) is not dict or (expected is not None and type(expected) is not dict):
        raise TypeError("PNG metadata update and expected value must be JSON objects")
    return _bounded_pnginfo_json({"updates": updates, "expected": expected})


def decode_pnginfo_update(payload):
    if type(payload) is not str or len(payload.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError("PNG metadata update must be a JSON string of at most 1 MiB")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("PNG metadata update contains duplicate JSON keys")
            result[key] = value
        return result
    try:
        value = json.loads(payload, object_pairs_hook=unique_object)
    except RecursionError as error:
        raise ValueError("PNG metadata nesting is too deep") from error
    if type(value) is not dict or set(value) != {"updates", "expected"}:
        raise ValueError("PNG metadata update requires only updates and expected")
    encode_pnginfo_update(value["updates"], value["expected"])
    return value["updates"], value["expected"]


def _same_pnginfo_json(left, right):
    # JSON booleans must never compare equal to numbers. Keep numeric types
    # exact too, while object key insertion order has no semantic effect.
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _same_pnginfo_json(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _same_pnginfo_json(a, b) for a, b in zip(left, right))
    return left == right


class ExecutionMetadataWriteScope:
    """Host-only authority for one actual execute invocation, never a Guest ref."""
    def __init__(self, owner, node_id, guard):
        if type(owner) is not ExecutionMetadataScope:
            raise TypeError("PNG metadata write requires the native execution owner")
        self.owner = owner
        self.node_id = str(node_id)
        self.prompt_id = owner.prompt_id
        self.token = uuid.uuid4().hex
        self._guard = guard
        self._active = False
        self._closed = False
        self._task = None
        self._dispatch_task = None
        self._context = None

    def activate(self):
        with self.owner._write_lock:
            if self._closed or self._active or current(self.prompt_id) is not self.owner:
                raise PermissionError("PNG metadata write scope is not current")
            self.owner.dynamic_prompt.get_node(self.node_id)
            self._task = asyncio.current_task()
            self._dispatch_task = self._task
            self._context = contextvars.copy_context()
            self._active = True
            self.check(self.prompt_id, self.node_id)

    def bind_native_task(self, prompt_id, node_id):
        """Core binds its actual async callback, in addition to the dispatch."""
        with self.owner._write_lock:
            self.check(prompt_id, node_id)
            task = asyncio.current_task()
            if (self._task is not self._dispatch_task or task is self._dispatch_task
                    or task is None or task.done() or task.cancelling()
                    or current(self.prompt_id) is not self.owner):
                raise PermissionError("PNG metadata native callback is not current or is already bound")
            self._task = task
            self._context = contextvars.copy_context()

    def check(self, prompt_id, node_id):
        if (not self._active or self._closed or self.owner.closed
                or self.prompt_id != str(prompt_id) or self.node_id != str(node_id)
                or self._dispatch_task is None or self._dispatch_task.done() or self._dispatch_task.cancelling()
                or self._task is None or self._task.done() or self._task.cancelling()
                or self._context is None):
            raise PermissionError("PNG metadata write scope is closed, cancelled, or belongs to another invocation")
        self._context.copy().run(self._guard)

    def close(self):
        with self.owner._write_lock:
            self._active = False
            self._closed = True
            self._task = None
            self._dispatch_task = None
            self._context = None

    def update(self, updates, expected=None):
        # Validation and the final check precede one synchronous in-place commit.
        updates, expected = decode_pnginfo_update(encode_pnginfo_update(updates, expected))
        with self.owner._write_lock:
            self.check(self.prompt_id, self.node_id)
            target = self.owner.extra_pnginfo
            if type(target) is not dict:
                raise RuntimeError("no shared EXTRA_PNGINFO object is bound")
            if expected is not None and not _same_pnginfo_json(target, expected):
                return False
            merged = {**target, **updates}
            _bounded_pnginfo_json(merged)
            self.check(self.prompt_id, self.node_id)
            target.update(updates)
            return True

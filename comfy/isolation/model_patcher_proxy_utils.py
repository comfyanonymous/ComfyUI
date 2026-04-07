# pylint: disable=import-outside-toplevel,logging-fstring-interpolation,protected-access
# Isolation utilities and serializers for ModelPatcherProxy
from __future__ import annotations

import logging
import os
from typing import Any

from comfy.cli_args import args

logger = logging.getLogger(__name__)


def maybe_wrap_model_for_isolation(model_patcher: Any) -> Any:
    from comfy.isolation.model_patcher_proxy_registry import ModelPatcherRegistry
    from comfy.isolation.model_patcher_proxy import ModelPatcherProxy

    is_child = os.environ.get("PYISOLATE_CHILD") == "1"
    isolation_active = args.use_process_isolation or is_child

    if not isolation_active:
        return model_patcher
    if is_child:
        return model_patcher
    if isinstance(model_patcher, ModelPatcherProxy):
        return model_patcher

    registry = ModelPatcherRegistry()
    model_id = registry.register(model_patcher)
    logger.debug(f"Isolated ModelPatcher: {model_id}")
    return ModelPatcherProxy(model_id, registry, manage_lifecycle=True)


def register_hooks_serializers(registry=None):
    from pyisolate._internal.serialization_registry import SerializerRegistry
    import comfy.hooks

    if registry is None:
        registry = SerializerRegistry.get_instance()

    def serialize_enum(obj):
        return {"__enum__": f"{type(obj).__name__}.{obj.name}"}

    def deserialize_enum(data):
        cls_name, val_name = data["__enum__"].split(".")
        cls = getattr(comfy.hooks, cls_name)
        return cls[val_name]

    registry.register("EnumHookType", serialize_enum, deserialize_enum)
    registry.register("EnumHookScope", serialize_enum, deserialize_enum)
    registry.register("EnumHookMode", serialize_enum, deserialize_enum)
    registry.register("EnumWeightTarget", serialize_enum, deserialize_enum)

    def serialize_hook_group(obj):
        return {"__type__": "HookGroup", "hooks": obj.hooks}

    def deserialize_hook_group(data):
        hg = comfy.hooks.HookGroup()
        for h in data["hooks"]:
            hg.add(h)
        return hg

    registry.register("HookGroup", serialize_hook_group, deserialize_hook_group)

    def serialize_dict_state(obj):
        d = obj.__dict__.copy()
        d["__type__"] = type(obj).__name__
        if "custom_should_register" in d:
            del d["custom_should_register"]
        return d

    def deserialize_dict_state_generic(cls):
        def _deserialize(data):
            h = cls()
            h.__dict__.update(data)
            return h

        return _deserialize

    def deserialize_hook_keyframe(data):
        h = comfy.hooks.HookKeyframe(strength=data.get("strength", 1.0))
        h.__dict__.update(data)
        return h

    registry.register("HookKeyframe", serialize_dict_state, deserialize_hook_keyframe)

    def deserialize_hook_keyframe_group(data):
        h = comfy.hooks.HookKeyframeGroup()
        h.__dict__.update(data)
        return h

    registry.register(
        "HookKeyframeGroup", serialize_dict_state, deserialize_hook_keyframe_group
    )

    def deserialize_hook(data):
        h = comfy.hooks.Hook()
        h.__dict__.update(data)
        return h

    registry.register("Hook", serialize_dict_state, deserialize_hook)

    def deserialize_weight_hook(data):
        h = comfy.hooks.WeightHook()
        h.__dict__.update(data)
        return h

    registry.register("WeightHook", serialize_dict_state, deserialize_weight_hook)

    def serialize_set(obj):
        return {"__set__": list(obj)}

    def deserialize_set(data):
        return set(data["__set__"])

    registry.register("set", serialize_set, deserialize_set)

    try:
        from comfy.weight_adapter.lora import LoRAAdapter

        def serialize_lora(obj):
            return {"weights": {}, "loaded_keys": list(obj.loaded_keys)}

        def deserialize_lora(data):
            return LoRAAdapter(set(data["loaded_keys"]), data["weights"])

        registry.register("LoRAAdapter", serialize_lora, deserialize_lora)
    except Exception:
        pass

    try:
        from comfy.hooks import _HookRef
        import uuid

        def serialize_hook_ref(obj):
            return {
                "__hook_ref__": True,
                "id": getattr(obj, "_pyisolate_id", str(uuid.uuid4())),
            }

        def deserialize_hook_ref(data):
            h = _HookRef()
            h._pyisolate_id = data.get("id", str(uuid.uuid4()))
            return h

        registry.register("_HookRef", serialize_hook_ref, deserialize_hook_ref)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to register _HookRef: {e}")


try:
    register_hooks_serializers()
except Exception as e:
    logger.error(f"Failed to initialize hook serializers: {e}")

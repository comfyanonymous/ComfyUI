import asyncio
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
pyisolate_root = repo_root.parent / "pyisolate"
if pyisolate_root.exists():
    sys.path.insert(0, str(pyisolate_root))

from comfy.isolation.adapter import ComfyUIAdapter
from comfy.isolation.runtime_helpers import _wrap_remote_handles_as_host_proxies
from pyisolate._internal.model_serialization import deserialize_from_isolation
from pyisolate._internal.remote_handle import RemoteObjectHandle
from pyisolate._internal.serialization_registry import SerializerRegistry


def test_shared_model_ksampler_contract():
    registry = SerializerRegistry.get_instance()
    registry.clear()
    ComfyUIAdapter().register_serializers(registry)

    handle = RemoteObjectHandle("model_0", "ModelPatcher")

    class FakeExtension:
        async def call_remote_object_method(self, object_id, method_name, *args, **kwargs):
            assert object_id == "model_0"
            assert method_name == "get_model_object"
            assert args == ("latent_format",)
            assert kwargs == {}
            return "resolved:latent_format"

    wrapped = (handle,)
    assert isinstance(wrapped, tuple)
    assert isinstance(wrapped[0], RemoteObjectHandle)

    deserialized = asyncio.run(deserialize_from_isolation(wrapped))
    proxied = _wrap_remote_handles_as_host_proxies(deserialized, FakeExtension())
    model_for_host = proxied[0]

    assert not isinstance(model_for_host, RemoteObjectHandle)
    assert hasattr(model_for_host, "get_model_object")
    assert model_for_host.get_model_object("latent_format") == "resolved:latent_format"

    registry.clear()

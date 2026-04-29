import logging
import socket
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
pyisolate_root = repo_root.parent / "pyisolate"
if pyisolate_root.exists():
    sys.path.insert(0, str(pyisolate_root))

from comfy.isolation.adapter import ComfyUIAdapter
from comfy_api.latest._io import FolderType
from comfy_api.latest._ui import SavedImages, SavedResult
from pyisolate._internal.rpc_transports import JSONSocketTransport
from pyisolate._internal.serialization_registry import SerializerRegistry


def test_savedimages_roundtrip(caplog):
    registry = SerializerRegistry.get_instance()
    registry.clear()
    ComfyUIAdapter().register_serializers(registry)

    payload = SavedImages(
        results=[SavedResult("issue82.png", "slice2", FolderType.output)],
        is_animated=True,
    )

    a, b = socket.socketpair()
    sender = JSONSocketTransport(a)
    receiver = JSONSocketTransport(b)
    try:
        with caplog.at_level(logging.WARNING, logger="pyisolate._internal.rpc_transports"):
            sender.send({"ui": payload})
            result = receiver.recv()
    finally:
        sender.close()
        receiver.close()
        registry.clear()

    ui = result["ui"]
    assert isinstance(ui, SavedImages)
    assert ui.is_animated is True
    assert len(ui.results) == 1
    assert isinstance(ui.results[0], SavedResult)
    assert ui.results[0].filename == "issue82.png"
    assert ui.results[0].subfolder == "slice2"
    assert ui.results[0].type == FolderType.output
    assert ui.as_dict() == {
        "images": [SavedResult("issue82.png", "slice2", FolderType.output)],
        "animated": (True,),
    }
    assert not any("GENERIC SERIALIZER USED" in record.message for record in caplog.records)
    assert not any("GENERIC DESERIALIZER USED" in record.message for record in caplog.records)

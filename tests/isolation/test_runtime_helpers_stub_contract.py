"""Generic runtime-helper stub contract tests."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from comfy.isolation import runtime_helpers
from comfy_api.latest import io as latest_io
from tests.isolation.stage_internal_probe_node import PROBE_NODE_NAME, staged_probe_node


class _DummyExtension:
    def __init__(self, *, name: str, module_path: str):
        self.name = name
        self.module_path = module_path

    async def execute_node(self, _node_name: str, **inputs):
        return {
            "__node_output__": True,
            "args": (inputs,),
            "ui": {"status": "ok"},
            "expand": False,
            "block_execution": False,
        }


def _install_model_serialization_stub(monkeypatch):
    async def deserialize_from_isolation(payload, _extension):
        return payload

    monkeypatch.setitem(
        sys.modules,
        "pyisolate._internal.model_serialization",
        SimpleNamespace(
            serialize_for_isolation=lambda payload: payload,
            deserialize_from_isolation=deserialize_from_isolation,
        ),
    )


def test_stub_sets_relative_python_module(monkeypatch):
    _install_model_serialization_stub(monkeypatch)
    monkeypatch.setattr(runtime_helpers, "scan_shm_forensics", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_helpers, "_relieve_host_vram_pressure", lambda *args, **kwargs: None)

    extension = _DummyExtension(name="internal_probe", module_path=os.getcwd())
    stub = cast(Any, runtime_helpers.build_stub_class(
        "ProbeNode",
        {
            "is_v3": True,
            "schema_v1": {},
            "input_types": {},
        },
        extension,
        {},
        logging.getLogger("test"),
    ))

    info = getattr(stub, "GET_NODE_INFO_V1")()
    assert info["python_module"] == "custom_nodes.internal_probe"


def test_stub_ui_dispatch_roundtrip(monkeypatch):
    _install_model_serialization_stub(monkeypatch)
    monkeypatch.setattr(runtime_helpers, "scan_shm_forensics", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_helpers, "_relieve_host_vram_pressure", lambda *args, **kwargs: None)

    extension = _DummyExtension(name="internal_probe", module_path=os.getcwd())
    stub = runtime_helpers.build_stub_class(
        "ProbeNode",
        {
            "is_v3": True,
            "schema_v1": {"python_module": "custom_nodes.internal_probe"},
            "input_types": {},
        },
        extension,
        {},
        logging.getLogger("test"),
    )

    result = asyncio.run(getattr(stub, "_pyisolate_execute")(SimpleNamespace(), token="value"))

    assert isinstance(result, latest_io.NodeOutput)
    assert result.ui == {"status": "ok"}


def test_stub_class_types_align_with_extension():
    extension = SimpleNamespace(name="internal_probe", module_path="/sandbox/probe")
    running_extensions = {"internal_probe": extension}

    specs = [
        SimpleNamespace(module_path=Path("/sandbox/probe"), node_name="ProbeImage"),
        SimpleNamespace(module_path=Path("/sandbox/probe"), node_name="ProbeAudio"),
        SimpleNamespace(module_path=Path("/sandbox/other"), node_name="OtherNode"),
    ]

    class_types = runtime_helpers.get_class_types_for_extension(
        "internal_probe", running_extensions, specs
    )

    assert class_types == {"ProbeImage", "ProbeAudio"}


def test_probe_stage_requires_explicit_root():
    script = Path(__file__).resolve().parent / "stage_internal_probe_node.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "--target-root" in result.stderr


def test_probe_stage_cleans_up_context():
    with staged_probe_node() as module_path:
        staged_root = module_path.parents[1]
        assert module_path.name == PROBE_NODE_NAME
        assert staged_root.exists()

    assert not staged_root.exists()

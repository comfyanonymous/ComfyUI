"""Test that _InnerModelProxy exposes state_dict for LoRA loading."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

repo_root = Path(__file__).resolve().parents[2]
pyisolate_root = repo_root.parent / "pyisolate"
if pyisolate_root.exists():
    sys.path.insert(0, str(pyisolate_root))

from comfy.isolation.model_patcher_proxy import ModelPatcherProxy


def test_inner_model_proxy_state_dict_returns_keys():
    """_InnerModelProxy.state_dict() delegates to parent.model_state_dict()."""
    proxy = object.__new__(ModelPatcherProxy)
    proxy._model_id = "test_model"
    proxy._rpc = MagicMock()
    proxy._model_type_name = "SDXL"
    proxy._inner_model_channels = None

    fake_keys = ["diffusion_model.input.weight", "diffusion_model.output.weight"]
    proxy._call_rpc = MagicMock(return_value=fake_keys)

    inner = proxy.model
    sd = inner.state_dict()

    assert isinstance(sd, dict)
    assert "diffusion_model.input.weight" in sd
    assert "diffusion_model.output.weight" in sd
    proxy._call_rpc.assert_called_with("model_state_dict", None)


def test_inner_model_proxy_state_dict_callable():
    """state_dict is a callable, not a property — matches torch.nn.Module interface."""
    proxy = object.__new__(ModelPatcherProxy)
    proxy._model_id = "test_model"
    proxy._rpc = MagicMock()
    proxy._model_type_name = "SDXL"
    proxy._inner_model_channels = None

    proxy._call_rpc = MagicMock(return_value=[])

    inner = proxy.model
    assert callable(inner.state_dict)

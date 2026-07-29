import torch

from comfy_api.torch_helpers import torch_compile


def test_windows_rocm_default_mode_drops_injected_guard_options(monkeypatch):
    monkeypatch.setattr(torch_compile.sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", "7.15", raising=False)

    result = torch_compile.normalize_torch_compile_kwargs(
        {
            "backend": "inductor",
            "mode": "default",
            "options": {"guard_filter_fn": object()},
        }
    )

    assert result["mode"] is None
    assert result["options"] is None


def test_windows_rocm_custom_options_disable_cudagraphs(monkeypatch):
    monkeypatch.setattr(torch_compile.sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", "7.15", raising=False)

    result = torch_compile.normalize_torch_compile_kwargs(
        {
            "backend": "inductor",
            "mode": "default",
            "options": {"max_autotune": True},
        }
    )

    assert result["mode"] is None
    assert result["options"] == {
        "max_autotune": True,
        "triton.cudagraphs": False,
        "triton.cudagraph_trees": False,
    }


def test_non_rocm_compile_options_are_unchanged(monkeypatch):
    monkeypatch.setattr(torch_compile.sys, "platform", "linux")
    compile_kwargs = {
        "backend": "inductor",
        "mode": "default",
        "options": {"max_autotune": True},
    }

    assert torch_compile.normalize_torch_compile_kwargs(compile_kwargs) == compile_kwargs

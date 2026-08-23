from types import SimpleNamespace

import torch

import comfy.utils
from comfy.controlnet import ControlNet


class FakeControlModel:
    dtype = torch.float32

    def __call__(self, **kw):
        self.last_hint = kw["hint"]
        return torch.zeros((1, 4))


def make_cn():
    cn = ControlNet()
    cn.control_model = FakeControlModel()
    cn.model_sampling_current = SimpleNamespace(
        timestep=lambda t: t,
        calculate_input=lambda t, x: x,
    )
    # focus these tests on hint preparation, not output merging
    cn.control_merge = lambda control, control_prev, output_dtype=None: control
    cn.cond_hint_original = torch.zeros((1, 3, 16, 16))
    return cn


def call(cn, h, w, batch=1):
    x = torch.zeros((batch, 4, h, w))
    cond = {"c_crossattn": torch.zeros((1, 1, 1))}
    return cn.get_control(x, torch.tensor([999.0]), cond, 1, {})


def test_alternating_area_sizes_prepare_hint_once_per_size(monkeypatch):
    prep_calls = []
    real_upscale = comfy.utils.common_upscale

    def spy_upscale(samples, width, height, *a, **k):
        prep_calls.append((width, height))
        return real_upscale(samples, width, height, *a, **k)

    monkeypatch.setattr(comfy.utils, "common_upscale", spy_upscale)
    cn = make_cn()

    call(cn, 8, 8)
    call(cn, 4, 4)
    call(cn, 8, 8)

    # Area composition alternates two sizes every step; each size should be
    # prepared once instead of re-upscaling (and re-VAE-encoding) per step.
    assert len(prep_calls) == 2, f"hint prepared {len(prep_calls)} times"


def test_cached_hints_are_bounded():
    cn = make_cn()
    call(cn, 8, 8)
    call(cn, 4, 4)
    call(cn, 6, 6)

    assert len(cn.cond_hints) <= 2


def test_cleanup_releases_cached_hints():
    cn = make_cn()
    call(cn, 8, 8)

    cn.cleanup()

    assert len(cn.cond_hints) == 0


def test_batch_broadcast_does_not_pollute_cache(monkeypatch):
    prep_calls = []
    real_upscale = comfy.utils.common_upscale

    def spy_upscale(samples, width, height, *a, **k):
        prep_calls.append((width, height))
        return real_upscale(samples, width, height, *a, **k)

    monkeypatch.setattr(comfy.utils, "common_upscale", spy_upscale)
    cn = make_cn()
    call(cn, 8, 8, batch=1)
    call(cn, 8, 8, batch=2)

    assert len(prep_calls) == 1
    # the cached copy keeps its original pre-broadcast batch size so later
    # calls with any batch count can reuse it
    assert cn.cond_hints[(8, 8)].shape[0] == cn.cond_hint_original.shape[0]

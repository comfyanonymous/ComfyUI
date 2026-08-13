import math

import torch

import comfy.model_base
import comfy.model_management


class _StubModel:
    """Minimal stand-in for BaseModel: just the attributes memory_required reads."""
    memory_usage_factor_conds = ()
    memory_usage_shape_process = {}
    memory_usage_factor = 2.0

    def get_dtype_inference(self):
        return torch.bfloat16


INPUT_SHAPE = (1, 16, 1, 180, 320)
AREA = INPUT_SHAPE[0] * math.prod(INPUT_SHAPE[2:])
DTYPE_SIZE = 2  # bf16
EFFICIENT = AREA * DTYPE_SIZE * 0.01 * _StubModel.memory_usage_factor * (1024 * 1024)
CONSERVATIVE = AREA * 0.15 * _StubModel.memory_usage_factor * (1024 * 1024)


def _patch_attention(monkeypatch, xformers=False, pytorch_flash=False, flash=False):
    monkeypatch.setattr(comfy.model_management, "xformers_enabled", lambda: xformers)
    monkeypatch.setattr(comfy.model_management, "pytorch_attention_flash_attention", lambda: pytorch_flash)
    monkeypatch.setattr(comfy.model_management, "flash_attention_enabled", lambda: flash)


def _estimate():
    return comfy.model_base.BaseModel.memory_required(_StubModel(), INPUT_SHAPE)


def test_no_efficient_attention_uses_conservative_estimate(monkeypatch):
    _patch_attention(monkeypatch)
    assert _estimate() == CONSERVATIVE


def test_pytorch_flash_attention_uses_efficient_estimate(monkeypatch):
    _patch_attention(monkeypatch, pytorch_flash=True)
    assert _estimate() == EFFICIENT


def test_xformers_uses_efficient_estimate(monkeypatch):
    _patch_attention(monkeypatch, xformers=True)
    assert _estimate() == EFFICIENT


def test_flash_attention_flag_uses_efficient_estimate(monkeypatch):
    # --use-flash-attention must select the efficient estimate even when
    # pytorch attention was not auto enabled (e.g. torch builds without
    # working aotriton), otherwise the estimate is 7.5x too large.
    _patch_attention(monkeypatch, flash=True)
    assert _estimate() == EFFICIENT


def test_conservative_estimate_is_7_5x_efficient_at_bf16(monkeypatch):
    # Documents the size of the gap between the two formulas: at bf16 the
    # conservative path asks for 7.5x more working memory than the efficient
    # one for the same shapes. If either formula is retuned, this ratio (and
    # the impact of picking the wrong branch) changes; update it consciously.
    _patch_attention(monkeypatch, flash=True)
    efficient = _estimate()
    _patch_attention(monkeypatch)
    conservative = _estimate()
    assert conservative == 7.5 * efficient

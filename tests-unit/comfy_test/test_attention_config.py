import gc
import json
import weakref

import pytest
import torch

import comfy.ldm.modules.attention as attention
from comfy.ldm.minimax.model import PackedLayout


def make_module(config):
    encoded = torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)
    return attention.ComfyAttention().with_config(encoded)


@pytest.fixture
def sol_calls(monkeypatch):
    """Stub comfy_kitchen SOL so configs load on any device and record what reaches the kernel."""
    calls = []

    def sol_attn(q, k, v, **kwargs):
        calls.append(kwargs)
        return torch.zeros_like(q)

    monkeypatch.setattr(attention.comfy_kitchen, "sol_attn_is_available", lambda device: True)
    monkeypatch.setattr(attention.comfy_kitchen, "sol_attn", sol_attn)
    return calls


def run_sol(function, seq_len=1000, transformer_options=None):
    q = k = v = torch.zeros(1, 2, seq_len, 8)
    return function(q, k, v, 2, skip_reshape=True, transformer_options=transformer_options or {})


def test_single_config_is_unchanged(sol_calls):
    module = make_module({"attention": "comfy_kitchen_sol", "tau": 0.5})
    assert module.function.keywords == {"tau": 0.5}
    run_sol(module.function)
    assert sol_calls[0]["tau"] == 0.5
    assert sol_calls[0]["sink_blocks"] is None and sol_calls[0]["sink_q"] is None


def test_sol_options_and_sinks_reach_kernel(sol_calls):
    module = make_module({"attention": "comfy_kitchen_sol", "tau": 1.5, "topk_ratio": 0.25, "tail": False,
                          "sink_keys": "prefix", "sink_queries": "audio", "sink_extra": "ignored"})
    spans = {"prefix": (0, 300), "audio": (200, 300)}
    run_sol(module.function, transformer_options={"attention_spans": spans})
    call = sol_calls[0]
    assert (call["tau"], call["topk_ratio"], call["tail"]) == (1.5, 0.25, False)
    # 64-token blocks covering the rows: prefix rows [0, 300) -> blocks [0, 5); audio rows [200, 300) -> [3, 5)
    assert call["sink_blocks"] == [0, 5] and call["sink_q"] == [3, 5]


def test_sinks_ignore_missing_and_out_of_range_spans(sol_calls):
    module = make_module({"attention": "comfy_kitchen_sol", "sink_keys": "prefix", "sink_queries": "audio"})
    run_sol(module.function)  # model publishes no spans
    run_sol(module.function, seq_len=128, transformer_options={"attention_spans": {"prefix": (0, 300), "audio": (200, 300)}})
    assert sol_calls[0]["sink_blocks"] is None and sol_calls[0]["sink_q"] is None
    assert sol_calls[1]["sink_blocks"] == [0, 2] and sol_calls[1]["sink_q"] is None  # clamped / beyond a short sequence


def test_per_step_config_dispatches_on_sigma(sol_calls, monkeypatch):
    dense_calls = []
    monkeypatch.setattr(attention, "optimized_attention", lambda *args, **kwargs: dense_calls.append(1) or torch.ones(1))
    module = make_module([
        {"attention": "dense", "sigma_range": [0.8, 1.0]},
        {"attention": "comfy_kitchen_sol", "tau": 1.0, "sigma_range": [0.2, 0.8]},
        {"attention": "comfy_kitchen_sol", "tau": 2.0},  # fallback
    ])
    for sigma in (0.9, 0.5, 0.1):
        run_sol(module.function, transformer_options={"sigmas": torch.tensor([sigma])})
    assert len(dense_calls) == 1
    assert [call["tau"] for call in sol_calls] == [1.0, 2.0]
    run_sol(module.function)  # no sigmas -> fallback
    assert sol_calls[-1]["tau"] == 2.0


def test_explicit_dense_stops_preference_list(sol_calls, monkeypatch):
    monkeypatch.setattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False)
    assert make_module([{"attention": "dense"}, {"attention": "comfy_kitchen_sol"}]).function is None
    # an unavailable kernel is skipped, also when choosing the per-step fallback
    assert make_module([{"attention": "comfy_kitchen_int8"}, {"attention": "comfy_kitchen_sol", "tau": 2.0}]).function.keywords == {"tau": 2.0}
    per_step = make_module([{"attention": "comfy_kitchen_sol", "tau": 1.0, "sigma_range": [0.5, 1.0]},
                            {"attention": "comfy_kitchen_int8"}, {"attention": "comfy_kitchen_sol", "tau": 2.0}])
    run_sol(per_step.function, transformer_options={"sigmas": torch.tensor([0.1])})
    assert sol_calls[-1]["tau"] == 2.0


def test_per_step_cache_does_not_keep_sigmas_alive(sol_calls):
    module = make_module([{"attention": "comfy_kitchen_sol", "sigma_range": [0.0, 1.0]}])
    sigmas = torch.tensor([0.5])
    ref = weakref.ref(sigmas)
    run_sol(module.function, transformer_options={"sigmas": sigmas})
    del sigmas
    gc.collect()
    assert ref() is None


def test_per_step_config_rejects_bad_range(sol_calls):
    with pytest.raises(ValueError, match="sigma_range"):
        make_module([{"attention": "comfy_kitchen_sol", "sigma_range": [0.5]}])


def test_minimax_layout_attention_spans():
    layout = PackedLayout(10, 1, 4, 6, 3)
    starts = {kind: (a, b) for a, b, kind in layout.segments}
    assert layout.attention_spans == {"prefix": (0, starts["video"][0]), "text": starts["text"], "audio": starts["audio"]}

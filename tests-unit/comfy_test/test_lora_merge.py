import logging

import torch

import comfy.lora
import comfy.sd


class _FakeModel:
    def __init__(self, sd):
        self._sd = sd

    def state_dict(self):
        return self._sd


class _FakePatcher:
    def __init__(self, patches=None):
        self.patches = patches or {}


def _merge(sd, patcher, loaded, strength=1.0):
    return comfy.sd._merge_minimax_h3_adaln_patches(
        patcher, loaded, sd, strength
    )


def _apply_set_patch(value, base):
    patches = [(1.0, ("set", (value,)), 1.0, None, None)]
    return comfy.lora.calculate_weight(
        patches,
        base.clone(),
        "diffusion_model.blocks.0.adaln_proj.linear.weight",
    )


def test_stacked_pruned_adaln_merges_linearly():
    torch.manual_seed(0)
    table = torch.randn(4, 4)
    base_w = torch.randn(6, 4)
    base_b = torch.randn(6)
    w1 = base_w + torch.randn(6, 4) * 0.1
    w2 = base_w + torch.randn(6, 4) * 0.1

    wk = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    bk = "diffusion_model.blocks.0.adaln_proj.linear.bias"
    tk = "diffusion_model.adaln_t_table"
    sd = {tk: table, wk: base_w, bk: base_b}
    patcher = _FakePatcher({wk: [(1.0, ("set", (w1,)), 1.0, None, None)]})
    loaded = {
        tk: ("set", (table,)),
        wk: ("set", (w2,)),
        bk: ("set", (base_b + torch.randn(6) * 0.05,)),
    }

    merged = dict(_merge(sd, patcher, loaded, strength=0.5))
    assert wk not in loaded
    merged_w = merged[wk][1][0]
    expected = w1 + 0.5 * (w2 - base_w)
    assert torch.allclose(merged_w.float(), expected.float())


def test_sequential_weight_application_is_linear():
    torch.manual_seed(2)
    table = torch.randn(4, 4)
    base_w = torch.randn(6, 4)
    d1 = torch.randn(6, 4) * 0.05
    d2 = torch.randn(6, 4) * 0.05
    d3 = torch.randn(6, 4) * 0.05

    wk = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    tk = "diffusion_model.adaln_t_table"
    sd = {tk: table, wk: base_w}
    patcher = _FakePatcher()

    for delta in (d1, d2, d3):
        loaded = {
            tk: ("set", (table,)),
            wk: ("set", (base_w + delta,)),
        }
        merged = dict(_merge(sd, patcher, loaded, strength=1.0))
        patcher.patches[wk] = [
            (1.0, merged[wk], 1.0, None, None)
        ]

    applied = _apply_set_patch(patcher.patches[wk][0][1][1][0], base_w)
    expected = base_w + d1 + d2 + d3
    assert torch.allclose(applied.float(), expected.float())


def test_strength_is_applied_to_adaln_delta():
    torch.manual_seed(3)
    table = torch.randn(4, 4)
    base_w = torch.randn(6, 4)
    delta = torch.randn(6, 4) * 0.05

    wk = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    tk = "diffusion_model.adaln_t_table"
    sd = {tk: table, wk: base_w}
    patcher = _FakePatcher()
    loaded = {tk: ("set", (table,)), wk: ("set", (base_w + delta,))}
    merged = dict(_merge(sd, patcher, loaded, strength=0.5))

    applied = _apply_set_patch(merged[wk][1][0], base_w)
    assert torch.allclose(applied.float(), (base_w + 0.5 * delta).float())


def test_table_branch_retains_base_tensor():
    torch.manual_seed(4)
    table = torch.randn(4, 4)
    other_table = torch.randn(4, 4)
    base_w = torch.randn(6, 4)

    tk = "diffusion_model.adaln_t_table"
    wk = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    sd = {tk: table, wk: base_w}
    patcher = _FakePatcher()
    loaded = {tk: ("set", (other_table,)), wk: ("set", (base_w,))}

    merged = dict(_merge(sd, patcher, loaded))
    assert dict(merged)[tk][1][0] is table


def test_shape_mismatch_removes_incompatible_patch(caplog):
    torch.manual_seed(5)
    table = torch.randn(4, 4)
    base_w = torch.randn(6, 4)
    bad_w = torch.randn(7, 4)

    tk = "diffusion_model.adaln_t_table"
    wk = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    sd = {tk: table, wk: base_w}
    patcher = _FakePatcher()
    loaded = {tk: ("set", (table,)), wk: ("set", (bad_w,))}

    with caplog.at_level(logging.WARNING):
        _merge(sd, patcher, loaded)
    assert wk not in loaded
    assert any("shape mismatch" in r.message for r in caplog.records)


def test_table_mismatch_warns_but_merges(caplog):
    torch.manual_seed(1)
    table = torch.randn(4, 4)
    other_table = table + torch.randn(4, 4) * 0.1
    base_w = torch.randn(6, 4)
    w1 = base_w + torch.randn(6, 4) * 0.1
    w2 = base_w + torch.randn(6, 4) * 0.1

    wk = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    tk = "diffusion_model.adaln_t_table"
    sd = {tk: table, wk: base_w}
    patcher = _FakePatcher({wk: [(1.0, ("set", (w1,)), 1.0, None, None)]})
    loaded = {tk: ("set", (other_table,)), wk: ("set", (w2,))}

    with caplog.at_level(logging.WARNING):
        merged = _merge(sd, patcher, loaded)
    assert wk not in loaded
    assert any("adaln_t_table differs" in r.message for r in caplog.records)
    expected = w1 + (w2 - base_w)
    applied = _apply_set_patch(dict(merged)[wk][1][0], base_w)
    assert torch.allclose(applied.float(), expected.float())

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.modules.attention import _comfy_kitchen_int8_inputs


def test_skip_reshape_inputs_are_made_contiguous():
    # mirrors comfy/ldm/minimax/model.py's Attention.forward, which hands the
    # int8 kernel transposed views (q/k/v shape (s, heads, head_dim) -> (1, heads, s, head_dim))
    s, heads, head_dim = 8, 4, 16
    q = torch.randn(s, heads, head_dim).transpose(0, 1).unsqueeze(0)
    k = torch.randn(s, heads, head_dim).transpose(0, 1).unsqueeze(0)
    v = torch.randn(s, heads, head_dim).transpose(0, 1).unsqueeze(0)
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()

    q_out, k_out, v_out, mask_out, b, dim_head = _comfy_kitchen_int8_inputs(
        q, k, v, heads, None, skip_reshape=True, enable_gqa=False
    )

    assert q_out.is_contiguous()
    assert k_out.is_contiguous()
    assert v_out.is_contiguous()
    assert torch.equal(q_out, q)
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)
    assert mask_out is None
    assert b == 1
    assert dim_head == head_dim

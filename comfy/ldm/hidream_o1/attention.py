"""HiDream-O1 two-pass attention: tokens [0, ar_len) are causal, [ar_len, T)
attend full K/V. Splitting Q at the boundary avoids the (B, 1, T, T) additive
mask the general-purpose path would build (~500 MB at T~16K) and lets the
gen half hit the user's preferred backend via optimized_attention.
"""

import torch

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention


def make_two_pass_attention(ar_len: int):
    """Build a two-pass attention callable. AR pass uses SDPA-causal directly, gen pass routes through optimized_attention.
    """
    def two_pass_attention(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if skip_reshape:
            B, H, T, D = q.shape
        else:
            B, T, total_dim = q.shape
            D = total_dim // heads
            H = heads
            q = q.view(B, T, H, D).transpose(1, 2)
            k = k.view(B, T, H, D).transpose(1, 2)
            v = v.view(B, T, H, D).transpose(1, 2)

        if ar_len >= T:
            out = comfy.ops.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        elif ar_len <= 0:
            out = optimized_attention(q, k, v, heads, mask=None, skip_reshape=True, skip_output_reshape=True)
        else:
            out_ar = comfy.ops.scaled_dot_product_attention(
                q[:, :, :ar_len], k[:, :, :ar_len], v[:, :, :ar_len],
                attn_mask=None, dropout_p=0.0, is_causal=True,
            )
            out_gen = optimized_attention(
                q[:, :, ar_len:], k, v, heads,
                mask=None, skip_reshape=True, skip_output_reshape=True,
            )
            out = torch.cat([out_ar, out_gen], dim=2)

        if skip_output_reshape:
            return out
        return out.transpose(1, 2).reshape(B, T, H * D)

    return two_pass_attention

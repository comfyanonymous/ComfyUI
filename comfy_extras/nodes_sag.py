import torch
from torch import einsum
from einops import rearrange, repeat
import os
from comfy.ldm.modules.attention import optimized_attention, _ATTN_PRECISION

# from comfy/ldm/modules/attention.py
# but modified to return attention scores as well as output
def attention_basic(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    scale = dim_head ** -0.5

    h = heads
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    # force cast to fp32 to avoid overflowing
    if _ATTN_PRECISION =="fp32":
        with torch.autocast(enabled=False, device_type = 'cuda'):
            q, k = q.float(), k.float()
            sim = einsum('b i d, b j d -> b i j', q, k) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return (out, sim)

class SagNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "scale": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 30.0}),
                             "blur_sigma": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, scale, blur_sigma):
        m = model.clone()
        # set extra options on the model
        m.model.extra_options["sag"] = True
        m.model.extra_options["sag_scale"] = scale
        m.model.extra_options["sag_sigma"] = blur_sigma
        
        attn_scores = None
        m.model.get_attn_scores = lambda: attn_scores

        def attn_and_record(q, k, v, extra_options):
            nonlocal attn_scores
            # if uncond, save the attention scores
            cond_or_uncond = extra_options["cond_or_uncond"]
            if 1 in cond_or_uncond:
                uncond_index = cond_or_uncond.index(1)
                # do the entire attention operation, but save the attention scores to attn_scores
                (out, sim) = attention_basic(q, k, v, heads=extra_options["n_heads"])
                attn_scores = sim[uncond_index]
                return out
            else:
                return optimized_attention(q, k, v, heads = extra_options["n_heads"])

        # from diffusers:
        # unet.mid_block.attentions[0].transformer_blocks[0].attn1.patch
        # we might have to patch at different locations depending on sd1.5/2.1 vs sdXL
        m.set_model_patch_replace(attn_and_record, "attn1", "middle", 0)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "Self-Attention Guidance": SagNode,
}

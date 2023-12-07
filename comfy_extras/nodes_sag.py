import torch
from torch import einsum
from einops import rearrange, repeat
import os
from comfy.ldm.modules.attention import optimized_attention, _ATTN_PRECISION

# from comfy/ldm/modules/attention.py
# but modified to return attention scores as well as output
def attention_basic_with_sim(q, k, v, heads, mask=None):
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
                             "scale": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 5.0, "step": 0.1}),
                             "blur_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, scale, blur_sigma):
        m = model.clone()
        # set extra options on the model
        m.model_options["sag"] = True
        m.model_options["sag_scale"] = scale
        m.model_options["sag_sigma"] = blur_sigma
        
        attn_scores = None
        mid_block_shape = None
        m.model.get_attn_scores = lambda: attn_scores
        m.model.get_mid_block_shape = lambda: mid_block_shape

        # TODO: make this work properly with chunked batches
        #       currently, we can only save the attn from one UNet call
        def attn_and_record(q, k, v, extra_options):
            nonlocal attn_scores
            # if uncond, save the attention scores
            heads = extra_options["n_heads"]
            cond_or_uncond = extra_options["cond_or_uncond"]
            b = q.shape[0] // len(cond_or_uncond)
            if 1 in cond_or_uncond:
                uncond_index = cond_or_uncond.index(1)
                # do the entire attention operation, but save the attention scores to attn_scores
                (out, sim) = attention_basic_with_sim(q, k, v, heads=heads)
                # when using a higher batch size, I BELIEVE the result batch dimension is [uc1, ... ucn, c1, ... cn]
                n_slices = heads * b
                attn_scores = sim[n_slices * uncond_index:n_slices * (uncond_index+1)]
                return out
            else:
                return optimized_attention(q, k, v, heads=heads)

        # from diffusers:
        # unet.mid_block.attentions[0].transformer_blocks[0].attn1.patch
        def set_model_patch_replace(patch, name, key):
            to = m.model_options["transformer_options"]
            if "patches_replace" not in to:
                to["patches_replace"] = {}
            if name not in to["patches_replace"]:
                to["patches_replace"][name] = {}
            to["patches_replace"][name][key] = patch
        set_model_patch_replace(attn_and_record, "attn1", ("middle", 0, 0))
        # from diffusers:
        # unet.mid_block.attentions[0].register_forward_hook()
        def forward_hook(m, inp, out):
            nonlocal mid_block_shape
            mid_block_shape = out[0].shape[-2:]
        m.model.diffusion_model.middle_block[0].register_forward_hook(forward_hook)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "Self-Attention Guidance": SagNode,
}

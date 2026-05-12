"""Translate MoGe checkpoint keys to the layouts our nn.Modules use.

MoGe checkpoints embed DINOv2 with the original Meta naming
(``backbone.blocks.{i}.attn.qkv.weight``, ``ls1.gamma``, ``mlp.w12``, ...).
The shared ``comfy.image_encoders.dino2.Dinov2Model`` uses HF naming
(``encoder.layer.{i}.attention.attention.{query,key,value}.weight``,
``layer_scale1.lambda1``, ``mlp.weights_in``, ...).  We rewrite keys at load
time and split the fused ``qkv`` weight into separate Q/K/V tensors.
"""

from __future__ import annotations

import re


_DINOV2_TOPLEVEL_RENAMES = {
    "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
    "patch_embed.proj.bias":   "embeddings.patch_embeddings.projection.bias",
    "cls_token":               "embeddings.cls_token",
    "pos_embed":               "embeddings.position_embeddings",
    "register_tokens":         "embeddings.register_tokens",
    "mask_token":              "embeddings.mask_token",
    "norm.weight":             "layernorm.weight",
    "norm.bias":               "layernorm.bias",
}

_BLOCK_SUFFIX_RENAMES = [
    ("ls1.gamma",       "layer_scale1.lambda1"),
    ("ls2.gamma",       "layer_scale2.lambda1"),
    ("attn.proj.",      "attention.output.dense."),
    ("mlp.w12.",        "mlp.weights_in."),
    ("mlp.w3.",         "mlp.weights_out."),
]

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


def remap_dinov2_keys(sd: dict, src_prefix: str = "") -> dict:
    """Rewrite Meta-style DINOv2 keys under ``src_prefix`` to comfy/HF naming.

    Splits each fused ``attn.qkv.{weight,bias}`` into separate
    ``attention.attention.{query,key,value}.{weight,bias}`` tensors using a
    chunk along the leading dim.

    Keys that do not start with ``src_prefix`` are returned unchanged.
    """
    out: dict = {}
    for k, v in sd.items():
        if not k.startswith(src_prefix):
            out[k] = v
            continue
        rel = k[len(src_prefix):]

        # Top-level (cls token, pos embed, patch embed, mask token, register tokens, final norm).
        if rel in _DINOV2_TOPLEVEL_RENAMES:
            out[src_prefix + _DINOV2_TOPLEVEL_RENAMES[rel]] = v
            continue

        m = _BLOCK_RE.match(rel)
        if not m:
            out[k] = v
            continue

        i, sub = m.group(1), m.group(2)

        # Split fused qkv into separate q / k / v tensors.
        if sub == "attn.qkv.weight" or sub == "attn.qkv.bias":
            q, kw, vw = v.chunk(3, dim=0)
            tail = sub.rsplit(".", 1)[1]  # weight / bias
            base = "{}encoder.layer.{}.attention.attention".format(src_prefix, i)
            out["{}.query.{}".format(base, tail)] = q
            out["{}.key.{}".format(base, tail)] = kw
            out["{}.value.{}".format(base, tail)] = vw
            continue

        for old, new in _BLOCK_SUFFIX_RENAMES:
            sub = sub.replace(old, new)
        out["{}encoder.layer.{}.{}".format(src_prefix, i, sub)] = v

    return out


def remap_moge_state_dict(sd: dict) -> dict:
    """Convert a full MoGe checkpoint state dict to the layout our modules expect.

    - v1 backbone lives under ``backbone.``    -> rewrite that subtree.
    - v2 backbone lives under ``encoder.backbone.`` -> rewrite that subtree.

    Everything else (heads, neck, projections, image_mean/std buffers) keeps
    its original key names and passes through unchanged.
    """
    if any(k.startswith("encoder.backbone.") for k in sd):
        return remap_dinov2_keys(sd, src_prefix="encoder.backbone.")
    return remap_dinov2_keys(sd, src_prefix="backbone.")

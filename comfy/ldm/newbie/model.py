from __future__ import annotations
from typing import Optional, Any, Dict
import torch
import torch.nn as nn
import comfy.ldm.common_dit as common_dit
from comfy.ldm.lumina.model import NextDiT as NextDiTBase
from .components import RMSNorm

#######################################################
#            Adds support for NewBie image            #
#######################################################

def _fallback_operations():
    try:
        import comfy.ops
        return comfy.ops.disable_weight_init
    except Exception:
        return None

def _pop_unexpected_kwargs(kwargs: Dict[str, Any]) -> None:
    for k in (
        "model_type",
        "operation_settings",
        "unet_dtype",
        "weight_dtype",
        "precision",
        "extra_model_config",
    ):
        kwargs.pop(k, None)

class NewBieNextDiT_CLIP(NextDiTBase):

    def __init__(
        self,
        *args,
        clip_text_dim: int = 1024,
        clip_img_dim: int = 1024,
        device=None,
        dtype=None,
        operations=None,
        **kwargs,
    ):
        _pop_unexpected_kwargs(kwargs)
        if operations is None:
            operations = _fallback_operations()
        super().__init__(*args, device=device, dtype=dtype, operations=operations, **kwargs)
        self._nb_device = device
        self._nb_dtype = dtype
        self._nb_ops = operations
        min_mod = min(int(getattr(self, "dim", 1024)), 1024)
        if operations is not None and hasattr(operations, "Linear"):
            Linear = operations.Linear
            Norm = getattr(operations, "RMSNorm", None)
        else:
            Linear = nn.Linear
            Norm = None
        if Norm is not None:
            self.clip_text_pooled_proj = nn.Sequential(
                Norm(clip_text_dim, eps=1e-5, elementwise_affine=True, device=device, dtype=dtype),
                Linear(clip_text_dim, clip_text_dim, bias=True, device=device, dtype=dtype),
            )
        else:
            self.clip_text_pooled_proj = nn.Sequential(
                RMSNorm(clip_text_dim),
                nn.Linear(clip_text_dim, clip_text_dim, bias=True),
            )
        nn.init.normal_(self.clip_text_pooled_proj[1].weight, std=0.01)
        nn.init.zeros_(self.clip_text_pooled_proj[1].bias)
        self.time_text_embed = nn.Sequential(
            nn.SiLU(),
            Linear(min_mod + clip_text_dim, min_mod, bias=True, device=device, dtype=dtype),
        )
        nn.init.zeros_(self.time_text_embed[1].weight)
        nn.init.zeros_(self.time_text_embed[1].bias)
        if Norm is not None:
            self.clip_img_pooled_embedder = nn.Sequential(
                Norm(clip_img_dim, eps=1e-5, elementwise_affine=True, device=device, dtype=dtype),
                Linear(clip_img_dim, min_mod, bias=True, device=device, dtype=dtype),
            )
        else:
            self.clip_img_pooled_embedder = nn.Sequential(
                RMSNorm(clip_img_dim),
                nn.Linear(clip_img_dim, min_mod, bias=True),
            )
        nn.init.normal_(self.clip_img_pooled_embedder[1].weight, std=0.01)
        nn.init.zeros_(self.clip_img_pooled_embedder[1].bias)

    @staticmethod
    def _get_clip_from_kwargs(transformer_options: dict, kwargs: dict, key: str):
        if key in kwargs:
            return kwargs.get(key)
        if transformer_options is not None and key in transformer_options:
            return transformer_options.get(key)
        extra = transformer_options.get("extra_cond", None) if transformer_options else None
        if isinstance(extra, dict) and key in extra:
            return extra.get(key)
        return None
    def _forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        num_tokens: int,
        attention_mask: Optional[torch.Tensor] = None,
        transformer_options: dict = {},
        **kwargs,
    ):
        t = timesteps
        cap_feats = context
        cap_mask = attention_mask
        bs, c, h, w = x.shape
        x = common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        t_emb = self.t_embedder(t, dtype=x.dtype)
        adaln_input = t_emb
        clip_text_pooled = self._get_clip_from_kwargs(transformer_options, kwargs, "clip_text_pooled")
        clip_img_pooled = self._get_clip_from_kwargs(transformer_options, kwargs, "clip_img_pooled")
        if clip_text_pooled is not None:
            if clip_text_pooled.dim() > 2:
                clip_text_pooled = clip_text_pooled.view(clip_text_pooled.shape[0], -1)
            clip_text_pooled = clip_text_pooled.to(device=t_emb.device, dtype=t_emb.dtype)
            clip_emb = self.clip_text_pooled_proj(clip_text_pooled)
            adaln_input = self.time_text_embed(torch.cat([t_emb, clip_emb], dim=-1))
        if clip_img_pooled is not None:
            if clip_img_pooled.dim() > 2:
                clip_img_pooled = clip_img_pooled.view(clip_img_pooled.shape[0], -1)
            clip_img_pooled = clip_img_pooled.to(device=t_emb.device, dtype=t_emb.dtype)
            adaln_input = adaln_input + self.clip_img_pooled_embedder(clip_img_pooled)
        if isinstance(cap_feats, torch.Tensor):
            try:
                target_dtype = next(self.cap_embedder.parameters()).dtype
            except StopIteration:
                target_dtype = cap_feats.dtype
            cap_feats = cap_feats.to(device=t_emb.device, dtype=target_dtype)
        cap_feats = self.cap_embedder(cap_feats)
        patches = transformer_options.get("patches", {})
        x_is_tensor = True
        img, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(
            x, cap_feats, cap_mask, adaln_input, num_tokens, transformer_options=transformer_options
        )
        freqs_cis = freqs_cis.to(img.device)
        for i, layer in enumerate(self.layers):
            img = layer(img, mask, freqs_cis, adaln_input, transformer_options=transformer_options)
            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p(
                        {
                            "img": img[:, cap_size[0] :],
                            "txt": img[:, : cap_size[0]],
                            "pe": freqs_cis[:, cap_size[0] :],
                            "vec": adaln_input,
                            "x": x,
                            "block_index": i,
                            "transformer_options": transformer_options,
                        }
                    )
                    if isinstance(out, dict):
                        if "img" in out:
                            img[:, cap_size[0] :] = out["img"]
                        if "txt" in out:
                            img[:, : cap_size[0]] = out["txt"]

        img = self.final_layer(img, adaln_input)
        img = self.unpatchify(img, img_size, cap_size, return_tensor=x_is_tensor)
        img = img[:, :, :h, :w]
        return img

def NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP(**kwargs):
    _pop_unexpected_kwargs(kwargs)
    kwargs.setdefault("patch_size", 2)
    kwargs.setdefault("in_channels", 16)
    kwargs.setdefault("dim", 2304)
    kwargs.setdefault("n_layers", 36)
    kwargs.setdefault("n_heads", 24)
    kwargs.setdefault("n_kv_heads", 8)
    kwargs.setdefault("axes_dims", [32, 32, 32])
    kwargs.setdefault("axes_lens", [1024, 512, 512])
    return NewBieNextDiT_CLIP(**kwargs)

def NewBieNextDiT(*, device=None, dtype=None, operations=None, **kwargs):
    _pop_unexpected_kwargs(kwargs)
    if operations is None:
        operations = _fallback_operations()
    if dtype is None:
        dev_str = str(device) if device is not None else ""
        if dev_str.startswith("cuda") and torch.cuda.is_available():
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            dtype = torch.float32
    model = NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP(
        device=device, dtype=dtype, operations=operations, **kwargs
    )
    return model
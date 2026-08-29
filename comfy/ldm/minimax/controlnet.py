"""MiniMax H3 Fun ControlNet-Union model patch."""

import torch
import torch.nn as nn

import comfy.ldm.common_dit
from .model import DiTBlock, patchify_video


class ControlDiTBlock(DiTBlock):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps, first_block=False,
                 apply_silu=True, adaln_dtype=None, dtype=None, device=None, operations=None):
        super().__init__(hidden, heads, head_dim, ffn, t_dim, eps, qk_eps, apply_silu=apply_silu,
                         adaln_dtype=adaln_dtype, dtype=dtype, device=device, operations=operations)
        if first_block:
            self.before_proj = operations.Linear(hidden, hidden, bias=True, dtype=dtype, device=device)
        self.after_proj = operations.Linear(hidden, hidden, bias=True, dtype=dtype, device=device)


class MiniMaxH3FunControl(torch.nn.Module):
    def __init__(self, control_in_dim=49, injection_layers=(0, 10, 20, 30, 40), hidden_size=5376,
                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
                 time_embed_dim=2688, patch_size=(1, 2, 2), norm_eps=1e-5, qk_norm_eps=1e-5,
                 use_adaln_curves=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.patch_size = tuple(patch_size)
        self.injection_layers = tuple(injection_layers)
        if not self.injection_layers or self.injection_layers[0] != 0:
            raise ValueError("MiniMax H3 Fun control injection layers must start at layer 0")
        if self.injection_layers != tuple(sorted(set(self.injection_layers))):
            raise ValueError("MiniMax H3 Fun control injection layers must be unique and increasing")
        self.control_in_dim = control_in_dim
        patch_dim = control_in_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        self.control_proj_in = operations.Linear(patch_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.control_blocks = nn.ModuleList([
            ControlDiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                            time_embed_dim, norm_eps, qk_norm_eps, first_block=(i == 0),
                            apply_silu=not use_adaln_curves,
                            adaln_dtype=torch.float32 if use_adaln_curves else dtype,
                            dtype=dtype, device=device, operations=operations)
            for i in range(len(self.injection_layers))])

    def init_stream(self, h, control_latent, layout, t_emb):
        if any(kind not in ("text", "audio", "video") for _, _, kind in layout.segments):
            raise ValueError("MiniMax H3 Fun ControlNet does not support keyframe or reference conditioning")
        adaln_in = self.control_blocks[0].adaln_proj.linear.in_features
        if t_emb.shape[-1] != adaln_in:
            raise RuntimeError(
                "MiniMax H3 controlnet adaln width {} does not match the base model's timestep embedding width {}: "
                "the controlnet and base checkpoint use different adaln forms (curve basis vs full), "
                "convert the controlnet to match the base model.".format(adaln_in, t_emb.shape[-1]))

        patch_dim = self.control_in_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        control_latent = comfy.ldm.common_dit.pad_to_patch_size(control_latent.to(torch.float32), self.patch_size)
        target_rows = patchify_video(control_latent, self.patch_size)
        if target_rows.shape[1] < patch_dim:
            target_rows = torch.nn.functional.pad(target_rows, (0, patch_dim - target_rows.shape[1]))
        elif target_rows.shape[1] > patch_dim:
            raise ValueError("MiniMax H3 control input has {} columns but the model patch expects {}".format(target_rows.shape[1], patch_dim))

        c = h.clone()
        c[layout.img_pos.to(h.device)] = self.control_proj_in(target_rows).to(h.dtype)
        return self.control_blocks[0].before_proj(c).add_(h)

    def step(self, index, c, t_emb, mod_segments, rope_freqs, transformer_options):
        block = self.control_blocks[index]
        c = DiTBlock.forward(block, c, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        return c, block.after_proj(c)


def is_minimax_h3_fun_state_dict(state_dict):
    common = (
        "control_proj_in.weight",
        "control_blocks.0.adaln_proj.linear.weight",
        "control_blocks.0.after_proj.weight",
        "control_blocks.0.before_proj.weight",
    )
    native = (
        "control_blocks.0.attn.qkv_proj.weight",
        "control_blocks.0.attn.q_norm.weight",
        "control_blocks.0.mlp.fc1.weight",
    )
    diffusers = (
        "control_blocks.0.attn.to_q.weight",
        "control_blocks.0.attn.to_k.weight",
        "control_blocks.0.attn.to_v.weight",
        "control_blocks.0.attn.norm_q.weight",
        "control_blocks.0.ff.net.0.proj.weight",
    )
    return all(key in state_dict for key in common) and (all(key in state_dict for key in native) or all(key in state_dict for key in diffusers))


def convert_minimax_h3_fun_state_dict(state_dict):
    if "control_blocks.0.attn.to_q.weight" not in state_dict:
        return state_dict

    converted = {}
    for key, value in state_dict.items():
        if key.endswith(".attn.to_q.weight"):
            base = key[:-len("to_q.weight")]
            converted[base + "qkv_proj.weight"] = torch.cat([
                state_dict[base + "to_q.weight"],
                state_dict[base + "to_k.weight"],
                state_dict[base + "to_v.weight"],
            ], dim=0)
        elif key.endswith(".attn.to_k.weight") or key.endswith(".attn.to_v.weight"):
            continue
        elif key.endswith(".ff.net.0.proj.weight"):
            half = value.shape[0] // 2
            converted[key.replace(".ff.net.0.proj.", ".mlp.fc1.")] = torch.cat([value[half:], value[:half]], dim=0)
        else:
            converted[key.replace(".attn.norm_q.", ".attn.q_norm.")
                         .replace(".attn.norm_k.", ".attn.k_norm.")
                         .replace(".attn.to_out.0.", ".attn.out_proj.")
                         .replace(".ff.net.2.", ".mlp.fc2.")] = value
    return converted

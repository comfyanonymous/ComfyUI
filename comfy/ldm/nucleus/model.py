# Based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_nucleusmoe_image.py
# Copyright 2025 Nucleus-Image Team, The HuggingFace Team. All rights reserved.
# Apache 2.0 License
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import repeat

from comfy.ldm.lightricks.model import TimestepEmbedding, Timesteps
from comfy.ldm.modules.attention import optimized_attention_masked
import comfy.ldm.common_dit
import comfy.ops
import comfy.patcher_extension
from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class
from comfy.ldm.flux.math import apply_rope1


class NucleusMoETimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.time_proj = Timesteps(num_channels=embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=embedding_dim,
            time_embed_dim=4 * embedding_dim,
            out_dim=embedding_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.norm = operations.RMSNorm(embedding_dim, eps=1e-6, dtype=dtype, device=device)

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
        return self.norm(timesteps_emb)


class NucleusMoEEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list, scale_rope=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1

        self.register_buffer(
            "pos_freqs",
            torch.cat(
                [
                    self._rope_params(pos_index, self.axes_dim[0], self.theta),
                    self._rope_params(pos_index, self.axes_dim[1], self.theta),
                    self._rope_params(pos_index, self.axes_dim[2], self.theta),
                ],
                dim=1,
            ),
        )
        self.register_buffer(
            "neg_freqs",
            torch.cat(
                [
                    self._rope_params(neg_index, self.axes_dim[0], self.theta),
                    self._rope_params(neg_index, self.axes_dim[1], self.theta),
                    self._rope_params(neg_index, self.axes_dim[2], self.theta),
                ],
                dim=1,
            ),
        )

    @staticmethod
    def _rope_params(index, dim, theta=10000):
        assert dim % 2 == 0
        freqs = torch.outer(
            index.float(),
            1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)),
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def _compute_video_freqs(self, frame, height, width, idx=0, device=None):
        pos_freqs = self.pos_freqs.to(device)
        neg_freqs = self.neg_freqs.to(device)

        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)

        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(frame * height * width, -1)
        return freqs.clone().contiguous()

    def forward(self, video_fhw, device=None, max_txt_seq_len=None):
        if max_txt_seq_len is None:
            raise ValueError("max_txt_seq_len must be provided")

        if isinstance(video_fhw, list) and len(video_fhw) > 0:
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = None
        max_txt_seq_len_int = int(max_txt_seq_len)
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index_val = max(height // 2, width // 2)
            else:
                max_vid_index_val = max(height, width)
            if max_vid_index is None or max_vid_index_val > max_vid_index:
                max_vid_index = max_vid_index_val

        if max_vid_index is None:
            raise ValueError("video_fhw must contain at least one image shape")
        end_index = max_vid_index + max_txt_seq_len_int
        if end_index > self.pos_freqs.shape[0]:
            raise ValueError(
                f"Nucleus RoPE requires {end_index} positions, "
                f"but only {self.pos_freqs.shape[0]} are available."
            )

        txt_freqs = self.pos_freqs.to(device)[max_vid_index:end_index]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


def _apply_rotary_emb_nucleus(x, freqs_cis):
    """Apply rotary embeddings using complex multiplication.
    x: (B, S, H, D) tensor
    freqs_cis: (S, D/2) complex tensor
    """
    if x.shape[1] == 0:
        return x

    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)  # (S, 1, D/2)
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(-2)
    return x_out.type_as(x)


class NucleusMoEAttention(nn.Module):
    """Attention with GQA and text KV caching for Nucleus-Image.
    Image queries attend to concatenated image+text KV (cross-attention style).
    """

    def __init__(
        self,
        query_dim: int,
        dim_head: int = 128,
        heads: int = 16,
        num_kv_heads: int = 4,
        eps: float = 1e-5,
        bias: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.inner_kv_dim = num_kv_heads * dim_head
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = heads // num_kv_heads

        # Image projections
        self.to_q = operations.Linear(query_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)
        self.to_k = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)
        self.to_v = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)

        # Text projections
        self.add_k_proj = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)
        self.add_v_proj = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)

        # QK norms
        self.norm_q = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)

        # Output
        self.to_out = nn.ModuleList([
            operations.Linear(self.inner_dim, query_dim, bias=False, dtype=dtype, device=device),
        ])

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        cached_txt_key=None,
        cached_txt_value=None,
        transformer_options={},
    ):
        batch_size = hidden_states.shape[0]
        seq_img = hidden_states.shape[1]

        # Image projections -> (B, S, H, D_head)
        img_query = self.to_q(hidden_states).view(batch_size, seq_img, self.heads, -1)
        img_key = self.to_k(hidden_states).view(batch_size, seq_img, self.num_kv_heads, -1)
        img_value = self.to_v(hidden_states).view(batch_size, seq_img, self.num_kv_heads, -1)

        # Normalize
        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)

        # Apply RoPE to image Q/K
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = _apply_rotary_emb_nucleus(img_query, img_freqs)
            img_key = _apply_rotary_emb_nucleus(img_key, img_freqs)

        # Text KV
        if cached_txt_key is not None and cached_txt_value is not None:
            txt_key, txt_value = cached_txt_key, cached_txt_value
            joint_key = torch.cat([img_key, txt_key], dim=1)
            joint_value = torch.cat([img_value, txt_value], dim=1)
        elif encoder_hidden_states is not None:
            seq_txt = encoder_hidden_states.shape[1]
            txt_key = self.add_k_proj(encoder_hidden_states).view(batch_size, seq_txt, self.num_kv_heads, -1)
            txt_value = self.add_v_proj(encoder_hidden_states).view(batch_size, seq_txt, self.num_kv_heads, -1)

            txt_key = self.norm_added_k(txt_key)

            if image_rotary_emb is not None:
                txt_key = _apply_rotary_emb_nucleus(txt_key, txt_freqs)

            joint_key = torch.cat([img_key, txt_key], dim=1)
            joint_value = torch.cat([img_value, txt_value], dim=1)
        else:
            joint_key = img_key
            joint_value = img_value

        # GQA: repeat KV heads to match query heads
        if self.num_kv_groups > 1:
            joint_key = joint_key.repeat_interleave(self.num_kv_groups, dim=2)
            joint_value = joint_value.repeat_interleave(self.num_kv_groups, dim=2)

        # Reshape for attention: (B, H, S, D)
        img_query = img_query.transpose(1, 2)
        joint_key = joint_key.transpose(1, 2)
        joint_value = joint_value.transpose(1, 2)

        # Build attention mask
        if attention_mask is not None:
            seq_txt = attention_mask.shape[-1]
            attn_mask = torch.zeros(
                (batch_size, 1, seq_txt + seq_img),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            attn_mask[:, 0, seq_img:] = attention_mask
        else:
            attn_mask = None

        # Attention
        hidden_states = optimized_attention_masked(
            img_query, joint_key, joint_value, self.heads,
            attn_mask, transformer_options=transformer_options,
            skip_reshape=True,
        )

        hidden_states = self.to_out[0](hidden_states)
        return hidden_states


class SwiGLUExperts(nn.Module):
    """SwiGLU feed-forward experts for MoE.

    HF checkpoints store packed expert weights and can use grouped GEMM. Split
    per-expert Linear modules are still supported for locally quantized files.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_dim: int,
        num_experts: int,
        use_grouped_mm: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.moe_intermediate_dim = moe_intermediate_dim
        self.hidden_size = hidden_size
        self.use_grouped_mm = use_grouped_mm
        self._grouped_mm_failed = False
        self._dtype = dtype
        self._device = device
        self._operations = operations

    def _has_packed_experts(self):
        return getattr(self, "weight", None) is not None and getattr(self, "bias", None) is not None

    def _register_packed_experts(self, gate_up, down):
        self.weight = nn.Parameter(gate_up, requires_grad=False)
        self.bias = nn.Parameter(down, requires_grad=False)
        self.comfy_cast_weights = True
        self.weight_function = []
        self.bias_function = []

    def _pop_quant_conf(self, state_dict, key):
        quant_conf = state_dict.pop(key, None)
        if quant_conf is None:
            return None
        return json.loads(quant_conf.cpu().numpy().tobytes())

    def _load_packed_quant_tensor(self, state_dict, key, tensor, quant_conf):
        if quant_conf is None:
            return tensor

        quant_format = quant_conf.get("format", None)
        if quant_format is None:
            raise ValueError(f"Missing quantization format for Nucleus packed expert tensor {key}")
        if quant_format not in QUANT_ALGOS:
            raise ValueError(f"Unsupported quantization format {quant_format} for Nucleus packed expert tensor {key}")

        qconfig = QUANT_ALGOS[quant_format]
        layout_type = qconfig["comfy_tensor_layout"]
        layout_cls = get_layout_class(layout_type)
        scale = state_dict.pop(f"{key}_scale", None)
        if scale is None:
            scale = state_dict.pop(f"{key}.weight_scale", None)
        if scale is None:
            raise ValueError(f"Missing quantization scale for Nucleus packed expert tensor {key}")

        orig_dtype = self._dtype or torch.bfloat16
        params = layout_cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=tuple(tensor.shape),
        )
        return QuantizedTensor(tensor.to(dtype=qconfig["storage_t"]), layout_type, params)

    def _dequantize_for_expert_mm(self, tensor, dtype):
        if isinstance(tensor, QuantizedTensor):
            tensor = tensor.dequantize()
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _build_split_experts(self):
        if hasattr(self, "gate_up_projs"):
            return
        operations = self._operations
        self.gate_up_projs = nn.ModuleList([
            operations.Linear(self.hidden_size, 2 * self.moe_intermediate_dim, bias=False, dtype=self._dtype, device=self._device)
            for _ in range(self.num_experts)
        ])
        self.down_projs = nn.ModuleList([
            operations.Linear(self.moe_intermediate_dim, self.hidden_size, bias=False, dtype=self._dtype, device=self._device)
            for _ in range(self.num_experts)
        ])

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        gate_up_key = f"{prefix}gate_up_proj"
        down_key = f"{prefix}down_proj"
        has_packed = gate_up_key in state_dict and down_key in state_dict
        internal_weight_key = f"{prefix}weight"
        internal_bias_key = f"{prefix}bias"
        has_internal_packed = internal_weight_key in state_dict and internal_bias_key in state_dict
        quant_conf_key = f"{prefix}comfy_quant"
        packed_quant_conf = self._pop_quant_conf(state_dict, quant_conf_key)
        split_prefixes = (f"{prefix}gate_up_projs.", f"{prefix}down_projs.")
        has_split = any(k.startswith(split_prefixes) for k in state_dict)

        if has_packed or has_internal_packed:
            packed_gate_up_key = gate_up_key if has_packed else internal_weight_key
            packed_down_key = down_key if has_packed else internal_bias_key
            gate_up = state_dict.pop(packed_gate_up_key)
            down = state_dict.pop(packed_down_key)
            gate_up = self._load_packed_quant_tensor(state_dict, packed_gate_up_key, gate_up, packed_quant_conf)
            down = self._load_packed_quant_tensor(state_dict, packed_down_key, down, packed_quant_conf)
            self._register_packed_experts(gate_up, down)
        elif has_split:
            self._build_split_experts()

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        if has_packed or has_internal_packed:
            for key in (gate_up_key, down_key, internal_weight_key, internal_bias_key, quant_conf_key):
                if key in missing_keys:
                    missing_keys.remove(key)

    def _run_experts_split(self, x, num_tokens_per_expert):
        output = torch.zeros_like(x)
        cumsum = 0
        tokens_list = num_tokens_per_expert.tolist()

        for i in range(self.num_experts):
            n_tokens = int(tokens_list[i])
            if n_tokens == 0:
                continue
            expert_input = x[cumsum : cumsum + n_tokens]
            cumsum += n_tokens

            gate_up = self.gate_up_projs[i](expert_input)
            gate, up = gate_up.chunk(2, dim=-1)
            mid = F.silu(gate) * up
            expert_output = self.down_projs[i](mid)

            output[cumsum - n_tokens : cumsum] = expert_output

        return output

    def _run_experts_packed_for_loop(self, x, num_tokens_per_expert, gate_up_proj, down_proj):
        tokens_list = num_tokens_per_expert.tolist()
        num_real_tokens = sum(tokens_list)
        num_padding = x.shape[0] - num_real_tokens
        x_per_expert = torch.split(x[:num_real_tokens], split_size_or_sections=tokens_list, dim=0)

        expert_outputs = []
        for expert_idx, x_expert in enumerate(x_per_expert):
            if x_expert.shape[0] == 0:
                continue
            gate_up = torch.matmul(x_expert, gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_outputs.append(torch.matmul(F.silu(gate) * up, down_proj[expert_idx]))

        if len(expert_outputs) > 0:
            out = torch.cat(expert_outputs, dim=0)
        else:
            out = x.new_zeros((0, self.hidden_size))
        if num_padding > 0:
            out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
        return out

    def _can_use_grouped_mm(self, x, gate_up_proj):
        return (
            self.use_grouped_mm
            and not self._grouped_mm_failed
            and hasattr(F, "grouped_mm")
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and gate_up_proj.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )

    def _run_experts_grouped_mm(self, x, num_tokens_per_expert, gate_up_proj, down_proj):
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        gate_up = F.grouped_mm(x, gate_up_proj, offs=offsets)
        gate, up = gate_up.chunk(2, dim=-1)
        out = F.grouped_mm(F.silu(gate) * up, down_proj, offs=offsets)
        return out.type_as(x)

    def forward(self, x, num_tokens_per_expert):
        """Run experts on pre-permuted tokens.
        x: (total_tokens, hidden_size)
        num_tokens_per_expert: (num_experts,)
        """
        if self._has_packed_experts():
            gate_up_proj, down_proj, offload_stream = comfy.ops.cast_bias_weight(self, x, offloadable=True, compute_dtype=x.dtype)
            try:
                gate_up_proj = self._dequantize_for_expert_mm(gate_up_proj, x.dtype)
                down_proj = self._dequantize_for_expert_mm(down_proj, x.dtype)
                if self._can_use_grouped_mm(x, gate_up_proj):
                    try:
                        return self._run_experts_grouped_mm(x, num_tokens_per_expert, gate_up_proj, down_proj)
                    except RuntimeError:
                        self._grouped_mm_failed = True
                return self._run_experts_packed_for_loop(x, num_tokens_per_expert, gate_up_proj, down_proj)
            finally:
                comfy.ops.uncast_bias_weight(self, gate_up_proj, down_proj, offload_stream)

        if hasattr(self, "gate_up_projs"):
            return self._run_experts_split(x, num_tokens_per_expert)

        raise RuntimeError("Nucleus MoE experts have not loaded packed or split weights.")


class NucleusMoELayer(nn.Module):
    """Expert-Choice routing with shared expert."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        moe_intermediate_dim: int,
        capacity_factor: float = 2.0,
        use_sigmoid: bool = False,
        route_scale: float = 2.5,
        use_grouped_mm: bool = False,
        timestep_dim: int = None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.use_sigmoid = use_sigmoid
        self.route_scale = route_scale

        if timestep_dim is None:
            timestep_dim = hidden_size

        # Router: takes concatenated [timestep; hidden_states]
        self.gate = operations.Linear(
            hidden_size + timestep_dim, num_experts, bias=False,
            dtype=dtype, device=device,
        )

        # Routed experts
        self.experts = SwiGLUExperts(
            hidden_size=hidden_size,
            moe_intermediate_dim=moe_intermediate_dim,
            num_experts=num_experts,
            use_grouped_mm=use_grouped_mm,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        # Shared expert (processes ALL tokens) - FeedForward structure for checkpoint compat
        self.shared_expert = FeedForward(hidden_size, dim_out=hidden_size, inner_dim=moe_intermediate_dim, dtype=dtype, device=device, operations=operations)

    def forward(self, hidden_states, hidden_states_unmodulated, timestep=None):
        """
        Expert-Choice routing.
        hidden_states: (B, S, D)
        hidden_states_unmodulated: (B, S, D)
        timestep: (B, D)
        """
        B, S, D = hidden_states.shape

        if timestep is not None:
            timestep_expanded = timestep.unsqueeze(1).expand(-1, S, -1)
            router_input = torch.cat([timestep_expanded, hidden_states_unmodulated], dim=-1)
        else:
            router_input = hidden_states_unmodulated

        # Compute routing scores
        logits = self.gate(router_input)  # (B, S, num_experts)
        if self.use_sigmoid:
            scores = torch.sigmoid(logits.float()).to(logits.dtype)
        else:
            scores = F.softmax(logits.float(), dim=-1).to(logits.dtype)

        # Expert-Choice: top-C selection per expert
        capacity = max(1, math.ceil(self.capacity_factor * S / self.num_experts))

        affinity = scores.transpose(1, 2)  # (B, num_experts, S)
        topk = torch.topk(affinity, k=capacity, dim=-1)
        top_indices = topk.indices
        gating = affinity.gather(dim=-1, index=top_indices)

        batch_offsets = torch.arange(B, device=hidden_states.device, dtype=torch.long).view(B, 1, 1) * S
        global_token_indices = (batch_offsets + top_indices).transpose(0, 1).reshape(self.num_experts, -1).reshape(-1)
        gating_flat = gating.transpose(0, 1).reshape(self.num_experts, -1).reshape(-1)

        token_score_sums = torch.zeros(B * S, device=hidden_states.device, dtype=gating_flat.dtype)
        token_score_sums.scatter_add_(0, global_token_indices, gating_flat)
        gating_flat = gating_flat / (token_score_sums[global_token_indices] + 1e-12)
        gating_flat = gating_flat * self.route_scale

        # Get hidden states for selected tokens
        hidden_flat = hidden_states.reshape(-1, D)
        selected_hidden = hidden_flat[global_token_indices]  # (total_selected, D)

        # Run experts
        tokens_per_expert = B * capacity
        num_tokens_per_expert = torch.full(
            (self.num_experts,),
            tokens_per_expert,
            dtype=torch.long,
            device=hidden_states.device,
        )
        expert_output = self.experts(selected_hidden, num_tokens_per_expert)

        # Apply routing scores
        expert_output = (expert_output.float() * gating_flat.unsqueeze(-1)).to(hidden_states.dtype)

        # Scatter back
        output = self.shared_expert(hidden_states).reshape(B * S, D)
        scatter_idx = global_token_indices.reshape(-1, 1).expand(-1, D)
        output = output.scatter_add(dim=0, index=scatter_idx, src=expert_output)
        output = output.view(B, S, D)

        return output


class SwiGLUProj(nn.Module):
    """SwiGLU projection layer with .proj attribute for checkpoint key compatibility."""

    def __init__(self, dim, inner_dim, bias=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dim, inner_dim * 2, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        gate_up = self.proj(x)
        hidden_states, gate = gate_up.chunk(2, dim=-1)
        return hidden_states * F.silu(gate)


class FeedForward(nn.Module):
    """Dense SwiGLU feed-forward for non-MoE blocks.
    Uses net ModuleList to match checkpoint keys: net.0.proj.weight, net.2.weight
    """

    def __init__(self, dim, dim_out=None, inner_dim=None, mult=4, dtype=None, device=None, operations=None):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.ModuleList([
            SwiGLUProj(dim, inner_dim, bias=False, dtype=dtype, device=device, operations=operations),
            nn.Identity(),  # dropout placeholder
            operations.Linear(inner_dim, dim_out, bias=False, dtype=dtype, device=device),
        ])

    def forward(self, hidden_states):
        return self.net[2](self.net[0](hidden_states))


class NucleusMoEImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_kv_heads: int,
        joint_attention_dim: int = 4096,
        is_moe: bool = False,
        num_experts: int = 64,
        moe_intermediate_dim: int = 1344,
        capacity_factor: float = 2.0,
        use_sigmoid: bool = False,
        route_scale: float = 2.5,
        use_grouped_mm: bool = False,
        eps: float = 1e-6,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dim = dim
        self.is_moe = is_moe

        # Modulation: produces 4 params (scale1, gate1, scale2, gate2)
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 4 * dim, bias=True, dtype=dtype, device=device),
        )

        # Norms
        self.pre_attn_norm = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.pre_mlp_norm = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)

        # Attention
        self.attn = NucleusMoEAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            eps=eps,
            bias=False,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        # Text projection
        self.encoder_proj = operations.Linear(joint_attention_dim, dim, bias=True, dtype=dtype, device=device)

        # FFN / MoE
        if is_moe:
            self.img_mlp = NucleusMoELayer(
                hidden_size=dim,
                num_experts=num_experts,
                moe_intermediate_dim=moe_intermediate_dim,
                capacity_factor=capacity_factor,
                use_sigmoid=use_sigmoid,
                route_scale=route_scale,
                use_grouped_mm=use_grouped_mm,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        else:
            # Dense FFN inner_dim = 4 * moe_intermediate_dim (matching original model)
            dense_inner_dim = moe_intermediate_dim * 4
            self.img_mlp = FeedForward(
                dim=dim, dim_out=dim, inner_dim=dense_inner_dim,
                dtype=dtype, device=device, operations=operations,
            )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb=None,
        attention_kwargs=None,
        transformer_options={},
    ):
        scale1, gate1, scale2, gate2 = self.img_mod(temb).unsqueeze(1).chunk(4, dim=-1)

        gate1 = gate1.clamp(min=-2.0, max=2.0)
        gate2 = gate2.clamp(min=-2.0, max=2.0)

        attn_kwargs = attention_kwargs or {}

        # Text projection (or use cached KV)
        context = None if attn_kwargs.get("cached_txt_key") is not None else self.encoder_proj(encoder_hidden_states)

        # Attention
        img_normed = self.pre_attn_norm(hidden_states)
        img_modulated = img_normed * (1 + scale1)

        img_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=context,
            attention_mask=attn_kwargs.get("attention_mask"),
            image_rotary_emb=image_rotary_emb,
            cached_txt_key=attn_kwargs.get("cached_txt_key"),
            cached_txt_value=attn_kwargs.get("cached_txt_value"),
            transformer_options=transformer_options,
        )

        hidden_states = hidden_states + gate1.tanh() * img_attn_output

        # FFN / MoE
        img_normed2 = self.pre_mlp_norm(hidden_states)
        img_modulated2 = img_normed2 * (1 + scale2)

        if self.is_moe:
            img_mlp_output = self.img_mlp(img_modulated2, img_normed2, timestep=temb)
        else:
            img_mlp_output = self.img_mlp(img_modulated2)

        hidden_states = hidden_states + gate2.tanh() * img_mlp_output

        if hidden_states.dtype == torch.float16:
            fp16_finfo = torch.finfo(torch.float16)
            hidden_states = hidden_states.clip(fp16_finfo.min, fp16_finfo.max)

        return hidden_states


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim, conditioning_embedding_dim, elementwise_affine=False, norm_eps=1e-6, dtype=None, device=None, operations=None):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = operations.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=True, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(embedding_dim, norm_eps, elementwise_affine, dtype=dtype, device=device)

    def forward(self, x, temb):
        emb = self.linear(self.silu(temb).to(x.dtype))
        scale, shift = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        return x


class NucleusMoEImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        joint_attention_dim: int = 4096,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        rope_theta: int = 10000,
        scale_rope: bool = True,
        dense_moe_strategy: str = "leave_first_three_blocks_dense",
        num_experts: int = 64,
        moe_intermediate_dim: int = 1344,
        capacity_factors: list = None,
        use_sigmoid: bool = False,
        route_scale: float = 2.5,
        use_grouped_mm: bool = False,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # Determine which layers are MoE
        if capacity_factors is not None:
            moe_layers = [cf > 0 for cf in capacity_factors]
        else:
            moe_layers = [self._is_moe_layer(dense_moe_strategy, i, num_layers) for i in range(num_layers)]
        self.moe_layers = moe_layers

        # RoPE
        self.pos_embed = NucleusMoEEmbedRope(
            theta=rope_theta,
            axes_dim=list(axes_dims_rope),
            scale_rope=scale_rope,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        # Timestep embedding
        self.time_text_embed = NucleusMoETimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        # Input projections
        self.txt_norm = operations.RMSNorm(joint_attention_dim, eps=1e-6, dtype=dtype, device=device)
        self.img_in = operations.Linear(in_channels * patch_size * patch_size, self.inner_dim, dtype=dtype, device=device)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            NucleusMoEImageTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                num_kv_heads=num_key_value_heads,
                joint_attention_dim=joint_attention_dim,
                is_moe=moe_layers[i],
                num_experts=num_experts,
                moe_intermediate_dim=moe_intermediate_dim,
                capacity_factor=capacity_factors[i] if capacity_factors is not None else (2.0 if moe_layers[i] else 0.0),
                use_sigmoid=use_sigmoid,
                route_scale=route_scale,
                use_grouped_mm=use_grouped_mm,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            for i in range(num_layers)
        ])

        # Output
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, norm_eps=1e-6, dtype=dtype, device=device, operations=operations)
        self.proj_out = operations.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False, dtype=dtype, device=device)

    @staticmethod
    def _is_moe_layer(strategy, layer_idx, num_layers):
        if strategy == "leave_first_three_and_last_block_dense":
            return layer_idx >= 3 and layer_idx < num_layers - 1
        elif strategy == "leave_first_three_blocks_dense":
            return layer_idx >= 3
        elif strategy == "leave_first_block_dense":
            return layer_idx >= 1
        elif strategy == "all_moe":
            return True
        elif strategy == "all_dense":
            return False
        return True

    @staticmethod
    def _normalize_attention_mask(attention_mask, dtype):
        if attention_mask is None:
            return None
        if attention_mask.ndim > 2:
            attention_mask = attention_mask.reshape(attention_mask.shape[0], -1)

        if not torch.is_floating_point(attention_mask):
            return (attention_mask.to(dtype) - 1) * torch.finfo(dtype).max

        if torch.all((attention_mask == 0) | (attention_mask == 1)):
            return (attention_mask.to(dtype) - 1) * torch.finfo(dtype).max

        return attention_mask.to(dtype)

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        bs, c, t, h, w = x.shape
        patch_size = self.patch_size
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (1, self.patch_size, self.patch_size))
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(
            orig_shape[0], orig_shape[1],
            orig_shape[-3], orig_shape[-2] // 2, 2,
            orig_shape[-1] // 2, 2,
        )
        hidden_states = hidden_states.permute(0, 2, 3, 5, 1, 4, 6)
        hidden_states = hidden_states.reshape(
            orig_shape[0],
            orig_shape[-3] * (orig_shape[-2] // 2) * (orig_shape[-1] // 2),
            orig_shape[1] * 4,
        )
        t_len = t
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        h_offset = ((h_offset + (patch_size // 2)) // patch_size)
        w_offset = ((w_offset + (patch_size // 2)) // patch_size)

        return hidden_states, (t_len, h_len, w_len), orig_shape

    def forward(self, x, timestep, context, attention_mask=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options
            ),
        ).execute(x, timestep, context, attention_mask, transformer_options, **kwargs)

    def _forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        transformer_options={},
        control=None,
        **kwargs,
    ):
        encoder_hidden_states = context
        encoder_hidden_states_mask = self._normalize_attention_mask(attention_mask, x.dtype)

        block_attention_kwargs = {}
        if encoder_hidden_states_mask is not None:
            block_attention_kwargs["attention_mask"] = encoder_hidden_states_mask

        hidden_states, (t_len, h_len, w_len), orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        # Compute RoPE
        img_freqs, txt_freqs = self.pos_embed(
            video_fhw=[(t_len, h_len, w_len)],
            device=x.device,
            max_txt_seq_len=encoder_hidden_states.shape[1],
        )
        image_rotary_emb = (img_freqs, txt_freqs)

        # Project inputs
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)

        temb = self.time_text_embed(timesteps, hidden_states)

        patches_replace = transformer_options.get("patches_replace", {})
        patches = transformer_options.get("patches", {})
        blocks_replace = patches_replace.get("dit", {})

        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({
                    "img": hidden_states,
                    "txt": encoder_hidden_states,
                    "transformer_options": transformer_options,
                })
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]

        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "single"

        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i

            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = block(
                        hidden_states=args["img"],
                        encoder_hidden_states=args["txt"],
                        temb=args["vec"],
                        image_rotary_emb=args["pe"],
                        attention_kwargs=block_attention_kwargs,
                        transformer_options=args["transformer_options"],
                    )
                    return {"img": out, "txt": args["txt"]}

                out = blocks_replace[("single_block", i)](
                    {"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb, "transformer_options": transformer_options},
                    {"original_block": block_wrap},
                )
                hidden_states = out["img"]
                encoder_hidden_states = out.get("txt", encoder_hidden_states)
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=block_attention_kwargs,
                    transformer_options=transformer_options,
                )

            if "single_block" in patches:
                for p in patches["single_block"]:
                    out = p({
                        "img": hidden_states,
                        "txt": encoder_hidden_states,
                        "x": x,
                        "block_index": i,
                        "transformer_options": transformer_options,
                    })
                    hidden_states = out["img"]
                    encoder_hidden_states = out.get("txt", encoder_hidden_states)

            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        hidden_states[:, :add.shape[1]] += add

        # Output
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # Unpack
        hidden_states = hidden_states[:, :num_embeds].view(
            orig_shape[0], orig_shape[-3],
            orig_shape[-2] // 2, orig_shape[-1] // 2,
            orig_shape[1], 2, 2,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 2, 5, 3, 6)
        # Diffusers negates Nucleus predictions before FlowMatchEulerDiscreteScheduler.step().
        return -hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]

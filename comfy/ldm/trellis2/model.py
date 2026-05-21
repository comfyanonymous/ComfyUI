import torch
import torch.nn.functional as F
import torch.nn as nn
from comfy.ldm.trellis2.vae import SparseTensor, SparseLinear, sparse_cat, VarLenTensor
from typing import Optional, Tuple, Literal, Union, List
from comfy.ldm.trellis2.attention import (
    sparse_windowed_scaled_dot_product_self_attention, sparse_scaled_dot_product_attention, scaled_dot_product_attention
)
from comfy.ldm.genmo.joint_model.layers import TimestepEmbedder
from comfy.ldm.flux.math import apply_rope, apply_rope1

class SparseGELU(nn.GELU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))

class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, device=None, dtype=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio), device=device, dtype=dtype, operations=operations),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), channels, device=device, dtype=dtype, operations=operations),
        )

    def forward(self, x: VarLenTensor) -> VarLenTensor:
        return self.mlp(x)

class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        o = super().forward(x)
        return o.to(dtype=x_dtype)


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int, device, dtype):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim, device=device, dtype=dtype))

    def forward(self, x: Union[VarLenTensor, torch.Tensor]) -> Union[VarLenTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, VarLenTensor):
            x = x.replace(F.normalize(x.feats, dim=-1) * self.gamma * self.scale)
        else:
            x = F.normalize(x, dim=-1) * self.gamma * self.scale
        return x.to(x_type)

class SparseRotaryPositionEmbedder(nn.Module):
    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        device=None
    ):
        super().__init__()
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32, device=device) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** (self.freqs))

    def _get_freqs_cis(self, coords: torch.Tensor) -> torch.Tensor:
        phases_list = []
        for i in range(self.dim):
            phases_list.append(torch.outer(coords[..., i], self.freqs.to(coords.device)))

        phases = torch.cat(phases_list, dim=-1)

        if phases.shape[-1] < self.head_dim // 2:
            padn = self.head_dim // 2 - phases.shape[-1]
            phases = torch.cat([phases, torch.zeros(*phases.shape[:-1], padn, device=phases.device)], dim=-1)

        cos = torch.cos(phases)
        sin = torch.sin(phases)

        f_cis_0 = torch.stack([cos, sin], dim=-1)
        f_cis_1 = torch.stack([-sin, cos], dim=-1)
        freqs_cis = torch.stack([f_cis_0, f_cis_1], dim=-1)

        return freqs_cis

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    def forward(self, q, k=None):
        cache_name = f'rope_cis_{self.dim}d_f{self.rope_freq[1]}_hd{self.head_dim}'
        freqs_cis = q.get_spatial_cache(cache_name)

        if freqs_cis is None:
            coords = q.coords[..., 1:].to(torch.float32)
            freqs_cis = self._get_freqs_cis(coords)
            q.register_spatial_cache(cache_name, freqs_cis)

        if q.feats.ndim == 3:
            f_cis = freqs_cis.unsqueeze(1)
        else:
            f_cis = freqs_cis

        if k is None:
            return q.replace(apply_rope1(q.feats, f_cis))

        q_feats, k_feats = apply_rope(q.feats, k.feats, f_cis)
        return q.replace(q_feats), k.replace(k_feats)

    @staticmethod
    def apply_rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases.unsqueeze(-2)
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed

class RotaryPositionEmbedder(SparseRotaryPositionEmbedder):
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if torch.is_complex(phases):
            phases = phases.to(torch.complex64)
        else:
            phases = phases.to(torch.float32)
        if phases.shape[-1] < self.head_dim // 2:
                padn = self.head_dim // 2 - phases.shape[-1]
                phases = torch.cat([phases, torch.polar(
                    torch.ones(*phases.shape[:-1], padn, device=phases.device, dtype=torch.float32),
                    torch.zeros(*phases.shape[:-1], padn, device=phases.device, dtype=torch.float32)
                )], dim=-1)
        return phases

class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed", "double_windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        device=None, dtype=None, operations=None
    ):
        super().__init__()

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = operations.Linear(channels, channels * 3, bias=qkv_bias, device=device, dtype=dtype)
        else:
            self.to_q = operations.Linear(channels, channels, bias=qkv_bias, device=device, dtype=dtype)
            self.to_kv = operations.Linear(self.ctx_channels, channels * 2, bias=qkv_bias, device=device, dtype=dtype)

        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads, device=device, dtype=dtype)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads, device=device, dtype=dtype)

        self.to_out = operations.Linear(channels, channels, device=device, dtype=dtype)

        if use_rope:
            self.rope = SparseRotaryPositionEmbedder(self.head_dim, rope_freq=rope_freq, device=device)

    @staticmethod
    def _linear(module: nn.Linear, x: Union[VarLenTensor, torch.Tensor]) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x: Union[VarLenTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[VarLenTensor, torch.Tensor], num_fused: int) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, VarLenTensor) else x_feats

    def forward(self, x: SparseTensor, context: Optional[Union[VarLenTensor, torch.Tensor]] = None) -> SparseTensor:
        if self._type == "self":
            dtype = next(self.to_qkv.parameters()).dtype
            x = x.to(dtype)
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.qk_rms_norm or self.use_rope:
                q, k, v = qkv.unbind(dim=-3)
                if self.qk_rms_norm:
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                if self.use_rope:
                    q, k = self.rope(q, k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
            elif self.attn_mode == "double_windowed":
                qkv0 = qkv.replace(qkv.feats[:, :, self.num_heads//2:])
                qkv1 = qkv.replace(qkv.feats[:, :, :self.num_heads//2])
                h0 = sparse_windowed_scaled_dot_product_self_attention(
                    qkv0, self.window_size, shift_window=(0, 0, 0)
                )
                h1 = sparse_windowed_scaled_dot_product_self_attention(
                    qkv1, self.window_size, shift_window=tuple([self.window_size//2] * 3)
                )
                h = qkv.replace(torch.cat([h0.feats, h1.feats], dim=1))
        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            dtype = next(self.to_kv.parameters()).dtype
            context = context.to(dtype)
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=-3)
                k = self.k_rms_norm(k)
                h = sparse_scaled_dot_product_attention(q, k, v)
            else:
                h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h

def _split_proj_context(context):
    if not isinstance(context, dict):
        return context, None
    global_ctx = context["global"]
    if "proj" in context:
        return global_ctx, context["proj"]
    if "proj_semantic" in context and "proj_color" in context:
        return global_ctx, (context["proj_semantic"], context["proj_color"])
    return global_ctx, None


class ProjectAttentionSparse(nn.Module):
    def __init__(self, cross_attn_block: nn.Module, channels: int, proj_in_channels: int,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = operations.Linear(proj_in_channels, channels, bias=True,
                                             device=device, dtype=dtype)

    def forward(self, x: SparseTensor, context) -> SparseTensor:
        global_ctx, proj_in = _split_proj_context(context)
        global_out = self.cross_attn_block(x, global_ctx)
        if isinstance(proj_in, tuple):
            proj_in = torch.cat([proj_in[0], proj_in[1]], dim=-1)
        proj_out = self.proj_linear(proj_in.to(self.proj_linear.weight.dtype))
        return global_out.replace(global_out.feats + proj_out.to(global_out.feats.dtype))


class ProjectAttentionDense(nn.Module):
    def __init__(self, cross_attn_block: nn.Module, channels: int, proj_in_channels: int,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = operations.Linear(proj_in_channels, channels, bias=True,
                                             device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, context) -> torch.Tensor:
        global_ctx, proj_in = _split_proj_context(context)
        global_out = self.cross_attn_block(x, global_ctx)
        if isinstance(proj_in, tuple):
            proj_in = torch.cat([proj_in[0], proj_in[1]], dim=-1)
        proj_out = self.proj_linear(proj_in.to(self.proj_linear.weight.dtype))
        return global_out + proj_out.to(global_out.dtype)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        image_attn_mode: Literal["global", "proj", "gated_proj"] = "global",
        proj_in_channels: Optional[int] = None,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.image_attn_mode = image_attn_mode
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, device=device)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6, device=device)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, device=device)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
            device=device, dtype=dtype, operations=operations
        )
        cross_inner = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            device=device, dtype=dtype, operations=operations
        )
        if image_attn_mode == "global":
            self.cross_attn = cross_inner
        else:
            if proj_in_channels is None:
                raise ValueError("proj_in_channels must be set when image_attn_mode != 'global'")
            self.cross_attn = ProjectAttentionSparse(
                cross_inner, channels, proj_in_channels,
                device=device, dtype=dtype, operations=operations,
            )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            device=device, dtype=dtype, operations=operations
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(channels, 6 * channels, bias=True, device=device, dtype=dtype)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels, device=device, dtype=dtype) / channels ** 0.5)

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        if self.image_attn_mode == "global":
            global_ctx, _ = _split_proj_context(context)
            h = self.cross_attn(h, global_ctx)
        else:
            h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context) -> SparseTensor:
        return self._forward(x, mod, context)


class SLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "rope",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = 'vanilla',
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        image_attn_mode: Literal["global", "proj", "gated_proj"] = "global",
        proj_in_channels: Optional[int] = None,
        dtype = None,
        device = None,
        operations = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.image_attn_mode = image_attn_mode
        self.proj_in_channels = proj_in_channels
        self.dtype = dtype

        self.t_embedder = TimestepEmbedder(model_channels, device=device, dtype=dtype, operations=operations)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(model_channels, 6 * model_channels, bias=True, device=device, dtype=dtype)
            )

        self.input_layer = SparseLinear(in_channels, model_channels, device=device, dtype=dtype, operations=operations)

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                image_attn_mode=image_attn_mode,
                proj_in_channels=proj_in_channels,
                device=device, dtype=dtype, operations=operations
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = SparseLinear(model_channels, out_channels, device=device, dtype=dtype, operations=operations)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        x: SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[SparseTensor] = None,
        **kwargs
    ) -> SparseTensor:
        if concat_cond is not None:
            x = sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = VarLenTensor.from_tensor_list(cond)

        dtype = next(self.input_layer.parameters()).dtype
        x = x.to(dtype)
        h = self.input_layer(x)
        t = t.to(dtype)
        t_embedder = self.t_embedder.to(dtype)
        t_emb = t_embedder(t, out_dtype = t.dtype)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h

class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, device=None, dtype=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(channels, int(channels * mlp_ratio), device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            operations.Linear(int(channels * mlp_ratio), channels, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int, device=None, dtype=None):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        device=None, dtype=None, operations=None
    ):
        super().__init__()

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = operations.Linear(channels, channels * 3, bias=qkv_bias, dtype=dtype, device=device)
        else:
            self.to_q = operations.Linear(channels, channels, bias=qkv_bias, device=device, dtype=dtype)
            self.to_kv = operations.Linear(self.ctx_channels, channels * 2, bias=qkv_bias, device=device, dtype=dtype)

        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads, device=device, dtype=dtype)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads, device=device, dtype=dtype)

        self.to_out = operations.Linear(channels, channels, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            x = x.to(next(self.to_qkv.parameters()).dtype)
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)

            if self.attn_mode == "full":
                if self.qk_rms_norm or self.use_rope:
                    q, k, v = qkv.unbind(dim=2)
                    if self.qk_rms_norm:
                        q = self.q_rms_norm(q)
                        k = self.k_rms_norm(k)
                    if self.use_rope:
                        assert phases is not None, "Phases must be provided for RoPE"
                        q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
                        k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            context = context.to(next(self.to_kv.parameters()).dtype)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h

class ModulatedTransformerCrossBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        image_attn_mode: Literal["global", "proj", "gated_proj"] = "global",
        proj_in_channels: Optional[int] = None,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.image_attn_mode = image_attn_mode
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, device=device)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6, device=device)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, device=device)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
            device=device, dtype=dtype, operations=operations
        )
        cross_inner = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            device=device, dtype=dtype, operations=operations
        )
        if image_attn_mode == "global":
            self.cross_attn = cross_inner
        else:
            if proj_in_channels is None:
                raise ValueError("proj_in_channels must be set when image_attn_mode != 'global'")
            self.cross_attn = ProjectAttentionDense(
                cross_inner, channels, proj_in_channels,
                device=device, dtype=dtype, operations=operations,
            )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            device=device, dtype=dtype, operations=operations
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels, device=device, dtype=dtype) / channels ** 0.5)

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h, phases=phases)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        if self.image_attn_mode == "global":
            global_ctx, _ = _split_proj_context(context)
            h = self.cross_attn(h, global_ctx)
        else:
            h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._forward(x, mod, context, phases)


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "rope",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = 'vanilla',
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        image_attn_mode: Literal["global", "proj", "gated_proj"] = "global",
        proj_in_channels: Optional[int] = None,
        operations=None,
        device = None,
        dtype = torch.float32,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.image_attn_mode = image_attn_mode
        self.proj_in_channels = proj_in_channels
        self.dtype = dtype
        self.device = device

        self.t_embedder = TimestepEmbedder(model_channels, dtype=dtype, device=device, operations=operations)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(model_channels, 6 * model_channels, bias=True, device=device, dtype=dtype)
            )

        pos_embedder = RotaryPositionEmbedder(self.model_channels // self.num_heads, 3, device=device)
        coords = torch.meshgrid(*[torch.arange(res, device=self.device, dtype=dtype) for res in [resolution] * 3], indexing='ij')
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        rope_phases = pos_embedder(coords)
        self.register_buffer("rope_phases", rope_phases, persistent=False)

        if pe_mode != "rope":
            self.rope_phases = None

        self.input_layer = operations.Linear(in_channels, model_channels, device=device, dtype=dtype)

        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                image_attn_mode=image_attn_mode,
                proj_in_channels=proj_in_channels,
                device=device, dtype=dtype, operations=operations
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = operations.Linear(model_channels, out_channels, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], self.in_channels, *[self.resolution] * 3)

        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = h.to(next(self.input_layer.parameters()).dtype)
        h = self.input_layer(h)
        t_emb = self.t_embedder(t, out_dtype = t.dtype)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        for block in self.blocks:
            h = block(h, t_emb, cond, self.rope_phases)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution] * 3).contiguous()

        return h

def timestep_reshift(t_shifted, old_shift=3.0, new_shift=5.0):
    t_shifted = t_shifted / 1000.0
    t_linear = t_shifted / (old_shift - t_shifted * (old_shift - 1))
    t_new = (new_shift * t_linear) / (1 + (new_shift - 1) * t_linear)
    t_new *= 1000.0
    return t_new


# Pixal3D ProjGrid math — port of upstream's ProjGrid + project_points_to_image_batch.
# World frame uses world Y as depth (Blender convention), camera looks along -Z local;
# transform_matrix is camera-to-world (inverted internally). Intrinsics: fx = 16 / tan(fov/2)
# with sensor_width = 32mm.

_PROJ_GRID_ROTATION = torch.tensor(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0],
     [0.0, 1.0, 0.0]]
)

_PROJ_FRONT_VIEW_TRANSFORM = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, -2.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
)


def _build_proj_transform_matrix(distance: torch.Tensor, batch_size: int,
                                 device, dtype=torch.float32) -> torch.Tensor:
    T = _PROJ_FRONT_VIEW_TRANSFORM.to(device=device, dtype=dtype)
    T = T.unsqueeze(0).expand(batch_size, -1, -1).clone()
    if distance.ndim == 0:
        distance = distance.expand(batch_size)
    T[:, 1, 3] = -distance.to(device=device, dtype=dtype)
    return T


def _project_points_to_image(points_world: torch.Tensor, transform_matrix: torch.Tensor,
                             camera_angle_x: torch.Tensor, resolution: int):
    B, N, _ = points_world.shape
    ones = torch.ones((B, N, 1), device=points_world.device, dtype=points_world.dtype)
    homo = torch.cat([points_world, ones], dim=-1)
    world_to_camera = torch.linalg.inv(transform_matrix.float()).to(transform_matrix.dtype)
    p_cam = torch.bmm(homo, world_to_camera.transpose(-2, -1))[..., :3]
    x_cam, y_cam, z_cam = p_cam.unbind(dim=-1)
    depth = -z_cam
    sensor_width = 32.0
    focal_length = 16.0 / torch.tan(camera_angle_x / 2.0)
    focal_px = focal_length * resolution / sensor_width
    focal_px = focal_px.to(p_cam.dtype).unsqueeze(1)
    denom = (-z_cam + 1e-8)
    x_pix = focal_px * x_cam / denom + resolution / 2.0
    y_pix = -focal_px * y_cam / denom + resolution / 2.0
    valid = ((x_pix >= 0) & (x_pix < resolution) &
             (y_pix >= 0) & (y_pix < resolution) & (depth > 0))
    return torch.stack([x_pix, y_pix], dim=-1), depth, valid


def _sample_features(feature_map: torch.Tensor, uv_ndc: torch.Tensor) -> torch.Tensor:
    B, C, _, _ = feature_map.shape
    grid = uv_ndc.view(B, -1, 1, 2).to(feature_map.dtype)
    feat = F.grid_sample(feature_map, grid, mode="bilinear",
                         padding_mode="border", align_corners=False)
    return feat.squeeze(-1)


def _coords_to_proj_world(coords: torch.Tensor, resolution: int, mesh_scale: torch.Tensor):
    if resolution < 1:
        raise ValueError(f"resolution must be positive, got {resolution}")
    batch_ids = coords[:, 0].long()
    if resolution == 1:
        norm = coords[:, 1:].to(torch.float32) * 0.0
    else:
        norm = coords[:, 1:].to(torch.float32) / (resolution - 1) * 2.0 - 1.0
    R = _PROJ_GRID_ROTATION.to(device=coords.device, dtype=torch.float32)
    rotated = norm @ R.T
    if mesh_scale.ndim == 0:
        scale_per_voxel = mesh_scale.expand(coords.shape[0])
    else:
        scale_per_voxel = mesh_scale.to(coords.device)[batch_ids]
    world = rotated / scale_per_voxel.unsqueeze(-1) / 2.0
    return world, batch_ids


def _dense_grid_proj_world(resolution: int, mesh_scale: torch.Tensor,
                           batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
    one = torch.linspace(-1.0, 1.0, resolution, device=device, dtype=dtype)
    x, y, z = torch.meshgrid(one, one, one, indexing="ij")
    grid = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    R_rot = _PROJ_GRID_ROTATION.to(device=device, dtype=dtype)
    grid = grid @ R_rot.T
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1).clone()
    if mesh_scale.ndim == 0:
        mesh_scale = mesh_scale.expand(batch_size)
    grid = grid / mesh_scale.to(device=device, dtype=dtype).view(-1, 1, 1) / 2.0
    return grid


def _back_project_to_tokens(
    coords_world: torch.Tensor,
    feature_map: torch.Tensor,
    transform_matrix: torch.Tensor,
    camera_angle_x: torch.Tensor,
    image_resolution: int,
    batch_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if coords_world.dim() == 2:
        assert batch_ids is not None
        B = transform_matrix.shape[0]
        out = torch.zeros((coords_world.shape[0], feature_map.shape[1]),
                          device=feature_map.device, dtype=feature_map.dtype)
        for b in range(B):
            mask = batch_ids == b
            if not mask.any():
                continue
            p = coords_world[mask].unsqueeze(0)
            uv, depth, valid = _project_points_to_image(
                p, transform_matrix[b:b+1], camera_angle_x[b:b+1], image_resolution)
            uv_ndc = (uv + 0.5) / image_resolution * 2.0 - 1.0
            # padding_mode='border' is load-bearing: masking out-of-frame voxels confuses
            # the SS DiT (~half the voxels go to zero, producing low poly + rotation drift).
            sampled = _sample_features(feature_map[b:b+1], uv_ndc)
            sampled = sampled.squeeze(0).transpose(0, 1)
            out[mask] = sampled
        return out
    else:
        uv, depth, valid = _project_points_to_image(
            coords_world, transform_matrix, camera_angle_x, image_resolution)
        uv_ndc = (uv + 0.5) / image_resolution * 2.0 - 1.0
        sampled = _sample_features(feature_map, uv_ndc)
        out = sampled.transpose(1, 2)
        return out


def _pack_per_voxel_scalar(proj_pack: Optional[dict], key: str, eval_batch: int, device) -> torch.Tensor:
    if proj_pack is None or key not in proj_pack:
        return torch.ones((eval_batch,), device=device, dtype=torch.float32)
    t = proj_pack[key].to(device=device, dtype=torch.float32)
    if t.ndim == 0:
        return t.expand(eval_batch).clone()
    return _expand_pack(t, eval_batch)


def _expand_pack(t: torch.Tensor, eval_batch: int) -> torch.Tensor:
    if eval_batch == t.shape[0]:
        return t
    if eval_batch % t.shape[0] != 0:
        raise ValueError(f"eval batch {eval_batch} is not a multiple of pack batch {t.shape[0]}")
    return t.repeat((eval_batch // t.shape[0],) + (1,) * (t.ndim - 1))


def _select_stage_entry(proj_pack: dict, stage: Optional[str]):
    """Returns (feature_map_lr, feature_map_hr_or_None, image_resolution)."""
    stages = proj_pack.get("stages")
    if stages is not None and stage is not None and stage in stages:
        entry = stages[stage]
        return entry["feature_map"], entry.get("feature_map_hr"), int(entry.get("image_resolution", 1024))
    if "feature_map" in proj_pack:
        return proj_pack["feature_map"], proj_pack.get("feature_map_hr"), int(proj_pack.get("image_resolution", 1024))
    raise ValueError(f"proj_feat_pack has no usable feature_map (stage={stage!r})")


def _build_proj_cond(global_cond: torch.Tensor, image_attn_mode: str, proj_pack: Optional[dict],
                     coords_world: torch.Tensor, batch_ids: Optional[torch.Tensor] = None,
                     eval_batch: Optional[int] = None,
                     proj_in_channels: Optional[int] = None,
                     stage: Optional[str] = None,
                     cond_or_uncond: Optional[list] = None):
    if image_attn_mode == "global":
        return global_cond
    if proj_pack is None:
        raise ValueError(f"image_attn_mode={image_attn_mode!r} but proj_feat_pack is missing")
    device = coords_world.device
    T = proj_pack["transform_matrix"].to(device)
    cam_angle = proj_pack["camera_angle_x"].to(device)
    feat_map_lr, feat_map_hr, image_resolution = _select_stage_entry(proj_pack, stage)
    feat_map_lr = feat_map_lr.to(device)
    if feat_map_hr is not None:
        feat_map_hr = feat_map_hr.to(device)
    if eval_batch is not None:
        T = _expand_pack(T, eval_batch)
        cam_angle = _expand_pack(cam_angle, eval_batch) if cam_angle.ndim >= 1 else cam_angle
        feat_map_lr = _expand_pack(feat_map_lr, eval_batch)
        if feat_map_hr is not None:
            feat_map_hr = _expand_pack(feat_map_hr, eval_batch)
    # Channel-count check against the trained proj_linear input. If HR is present, the
    # block expects (LR_channels + HR_channels) since we concat the sampled features.
    expected_channels = feat_map_lr.shape[1] + (feat_map_hr.shape[1] if feat_map_hr is not None else 0)
    if proj_in_channels is not None and expected_channels != proj_in_channels:
        hint = ""
        if feat_map_hr is None and expected_channels < proj_in_channels:
            hint = (" — feature_map_hr is missing for this stage. Connect a NAFModel "
                    "input to Pixal3DConditioning; the shape/texture stages of this "
                    "checkpoint need a NAF-upsampled HR feature map.")
        raise ValueError(
            f"proj_feat_pack[{stage!r}] has LR={feat_map_lr.shape[1]} "
            f"+ HR={feat_map_hr.shape[1] if feat_map_hr is not None else 0} "
            f"= {expected_channels} channels, sub-model expects {proj_in_channels}.{hint}"
        )
    proj_feats_lr = _back_project_to_tokens(coords_world, feat_map_lr, T, cam_angle,
                                            image_resolution=image_resolution,
                                            batch_ids=batch_ids)
    if feat_map_hr is not None:
        proj_feats_hr = _back_project_to_tokens(coords_world, feat_map_hr, T, cam_angle,
                                                image_resolution=image_resolution,
                                                batch_ids=batch_ids)
        proj_feats = torch.cat([proj_feats_lr, proj_feats_hr], dim=-1)
    else:
        proj_feats = proj_feats_lr
    # Mirror upstream's neg_cond by zeroing proj for any uncond batch slot.
    if cond_or_uncond is not None and eval_batch is not None:
        uncond_slots = [i for i, v in enumerate(cond_or_uncond) if v == 1]
        if uncond_slots:
            uncond_idx = torch.tensor(uncond_slots, device=proj_feats.device, dtype=torch.long)
            if batch_ids is None:
                proj_feats = proj_feats.clone()
                proj_feats[uncond_idx] = 0
            else:
                neg_mask = torch.isin(batch_ids, uncond_idx).unsqueeze(-1).to(proj_feats.dtype)
                proj_feats = proj_feats * (1.0 - neg_mask)
    return {"global": global_cond, "proj": proj_feats}

class Trellis2(nn.Module):
    def __init__(self, resolution,
                 in_channels = 32,
                 out_channels = 32,
                 model_channels = 1536,
                 cond_channels = 1024,
                 num_blocks = 30,
                 num_heads = 12,
                 mlp_ratio = 5.3334,
                 share_mod = True,
                 qk_rms_norm = True,
                 qk_rms_norm_cross = True,
                 init_txt_model=False, # for now
                 image_attn_mode_structure: str = "global",
                 proj_in_channels_structure: Optional[int] = None,
                 image_attn_mode_shape: str = "global",
                 proj_in_channels_shape: Optional[int] = None,
                 image_attn_mode_texture: str = "global",
                 proj_in_channels_texture: Optional[int] = None,
                 dtype=None, device=None, operations=None, **kwargs):

        super().__init__()
        self.dtype = dtype
        operations = operations or nn
        # for some reason it passes num_heads = -1
        if num_heads == -1:
            num_heads = 12
        args = {
            "out_channels":out_channels, "num_blocks":num_blocks, "cond_channels" :cond_channels,
            "model_channels":model_channels, "num_heads":num_heads, "mlp_ratio": mlp_ratio, "share_mod": share_mod,
            "qk_rms_norm": qk_rms_norm, "qk_rms_norm_cross": qk_rms_norm_cross, "device": device, "dtype": dtype, "operations": operations
        }
        self.image_attn_mode_structure = image_attn_mode_structure
        self.image_attn_mode_shape = image_attn_mode_shape
        self.image_attn_mode_texture = image_attn_mode_texture
        shape_proj_kwargs = {"image_attn_mode": image_attn_mode_shape, "proj_in_channels": proj_in_channels_shape}
        tex_proj_kwargs = {"image_attn_mode": image_attn_mode_texture, "proj_in_channels": proj_in_channels_texture}
        struct_proj_kwargs = {"image_attn_mode": image_attn_mode_structure, "proj_in_channels": proj_in_channels_structure}
        txt_only = kwargs.get("txt_only", False)
        if not txt_only:
            self.img2shape = SLatFlowModel(resolution=resolution, in_channels=in_channels, **shape_proj_kwargs, **args)
            self.shape2txt = None
            if init_txt_model:
                self.shape2txt = SLatFlowModel(resolution=resolution, in_channels=in_channels*2, **tex_proj_kwargs, **args)
            self.img2shape_512 = SLatFlowModel(resolution=32, in_channels=in_channels, **shape_proj_kwargs, **args)
            args.pop("out_channels")
            self.structure_model = SparseStructureFlowModel(resolution=16, in_channels=8, out_channels=8, **struct_proj_kwargs, **args)
        else:
            self.shape2txt = SLatFlowModel(resolution=resolution, in_channels=in_channels*2, **tex_proj_kwargs, **args)
        self.guidance_interval = [0.6, 1.0]
        self.guidance_interval_txt = [0.6, 0.9]

    def forward(self, x, timestep, context, **kwargs):
        transformer_options = kwargs.get("transformer_options", {})
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        model_options = {}
        if hasattr(self, "meta"):
            model_options = self.meta
        timestep = timestep.to(x.dtype)
        embeds = kwargs.get("embeds")
        if embeds is None:
            raise ValueError("Trellis2.forward requires 'embeds' in kwargs")

        is_1024 = True#self.img2shape.resolution == 1024
        coords = model_options.get("coords", None)
        coord_counts = model_options.get("coord_counts", None)
        mode = model_options.get("generation_mode", "structure_generation")
        proj_feat_pack = model_options.get("proj_feat_pack", None)
        coord_resolution = model_options.get("coord_resolution", None)

        is_512_run = False
        if mode == "shape_generation_512":
            is_512_run = True
            mode = "shape_generation"

        if coords is not None:
            if x.ndim == 4:
                x = x.squeeze(-1).transpose(1, 2)
            not_struct_mode = True
        else:
            mode = "structure_generation"
            not_struct_mode = False

        if x.size(-1) == 16 and x.size(-2) == 16:
            mode = "structure_generation"
            not_struct_mode = False

        if not not_struct_mode:
            bsz = x.size(0)
            x = x[:, :8]
            x = x.view(bsz, 8, 16, 16, 16)

        if is_1024 and not_struct_mode and not is_512_run:
            context = embeds

        sigmas = transformer_options.get("sigmas")[0].item()
        if sigmas < 1.00001:
            timestep *= 1000.0

        if context.size(0) > 1:
            cond = context.chunk(2)[1]
        else:
            cond = context

        shape_rule = sigmas < self.guidance_interval[0] or sigmas > self.guidance_interval[1]
        txt_rule = sigmas < self.guidance_interval_txt[0] or sigmas > self.guidance_interval_txt[1]

        if not_struct_mode:
            orig_bsz = x.shape[0]
            rule = txt_rule if mode == "texture_generation" else shape_rule

            # CFG Bypass Slicing
            if rule and orig_bsz > 1:
                half = orig_bsz // 2
                x_eval = x[half:]
                t_eval = timestep[half:] if timestep.shape[0] > 1 else timestep
                c_eval = cond
            else:
                x_eval = x
                t_eval = timestep
                c_eval = context

            B, N, C = x_eval.shape

            # Vectorized SparseTensor Construction
            if mode in ["shape_generation", "texture_generation"]:
                if coord_counts is not None:
                    logical_batch = coord_counts.shape[0]
                    # Duplicate coords if CFG is active
                    if B > logical_batch:
                        c_pos = coords.clone()
                        c_pos[:, 0] += logical_batch
                        batched_coords = torch.cat([coords, c_pos], dim=0)
                        counts_eval = torch.cat([coord_counts, coord_counts], dim=0)
                    else:
                        batched_coords = coords
                        counts_eval = coord_counts

                    # Create boolean mask [B, N] to drop the padded zeros instantly
                    mask = torch.arange(N, device=x.device).unsqueeze(0) < counts_eval.unsqueeze(1)
                    feats_flat = x_eval[mask]
                else:
                    feats_flat = x_eval.reshape(-1, C)
                    coords_list =[]
                    for i in range(B):
                        c = coords.clone()
                        c[:, 0] = i
                        coords_list.append(c)
                    batched_coords = torch.cat(coords_list, dim=0)
                    mask = None
            else:
                batched_coords = coords
                feats_flat = x_eval
                mask = None

            x_st = SparseTensor(feats=feats_flat, coords=batched_coords.to(torch.int32))

        if mode == "shape_generation":
            shape_attn = self.image_attn_mode_shape
            if shape_attn != "global":
                if coord_resolution is None:
                    raise ValueError("Pixal3D shape_generation requires coord_resolution in model_options; "
                                     "EmptyTrellis2ShapeLatent should set it from the input voxel.")
                mesh_scale = _pack_per_voxel_scalar(proj_feat_pack, "mesh_scale", B, batched_coords.device)
                xyz_world, batch_ids = _coords_to_proj_world(batched_coords, coord_resolution, mesh_scale)
                sub_model = self.img2shape_512 if is_512_run else self.img2shape
                stage_name = "shape_512" if is_512_run else "shape_1024"
                c_eval = _build_proj_cond(c_eval, shape_attn, proj_feat_pack, xyz_world, batch_ids,
                                          eval_batch=B,
                                          proj_in_channels=sub_model.proj_in_channels,
                                          stage=stage_name,
                                          cond_or_uncond=cond_or_uncond)
            if is_512_run:
                out = self.img2shape_512(x_st, t_eval, c_eval)
            else:
                out = self.img2shape(x_st, t_eval, c_eval)

        elif mode == "texture_generation":
            if self.shape2txt is None:
                raise ValueError("Checkpoint for Trellis2 doesn't include texture generation!")
            slat = model_options.get("shape_slat")
            if slat is None:
                raise ValueError("shape_slat can't be None")

            slat_feats = slat
            # Duplicate shape context if CFG is active
            if coord_counts is not None and B > coord_counts.shape[0]:
                slat_feats = torch.cat([slat_feats, slat_feats], dim=0)
            elif coord_counts is None:
                slat_feats = slat_feats[:N].repeat(B, 1)

            x_st = x_st.replace(feats=torch.cat([x_st.feats, slat_feats.to(x_st.feats.device)], dim=-1))
            tex_attn = self.image_attn_mode_texture
            if tex_attn != "global":
                if coord_resolution is None:
                    raise ValueError("Pixal3D texture_generation requires coord_resolution in model_options; "
                                     "EmptyTrellis2LatentTexture should set it from the input voxel.")
                mesh_scale = _pack_per_voxel_scalar(proj_feat_pack, "mesh_scale", B, batched_coords.device)
                xyz_world, batch_ids = _coords_to_proj_world(batched_coords, coord_resolution, mesh_scale)
                c_eval = _build_proj_cond(c_eval, tex_attn, proj_feat_pack, xyz_world, batch_ids,
                                          eval_batch=B,
                                          proj_in_channels=self.shape2txt.proj_in_channels,
                                          stage="tex_1024",
                                          cond_or_uncond=cond_or_uncond)
            out = self.shape2txt(x_st, t_eval, c_eval)

        else: # structure
            orig_bsz = x.shape[0]
            struct_attn = self.image_attn_mode_structure
            if shape_rule and orig_bsz > 1:
                half = orig_bsz // 2
                x_eval = x[half:]
                t_eval = timestep[half:] if timestep.shape[0] > 1 else timestep
                struct_cond = cond
                if struct_attn != "global":
                    mesh_scale = _pack_per_voxel_scalar(proj_feat_pack, "mesh_scale", half, x.device)
                    grid_xyz = _dense_grid_proj_world(16, mesh_scale, half, device=x.device)
                    struct_cond = _build_proj_cond(cond, struct_attn, proj_feat_pack, grid_xyz,
                                                   eval_batch=half,
                                                   proj_in_channels=self.structure_model.proj_in_channels,
                                                   stage="ss",
                                                   cond_or_uncond=cond_or_uncond)
                out = self.structure_model(x_eval, t_eval, struct_cond)
                out = out.repeat(2, 1, 1, 1, 1)
            else:
                struct_cond = context
                if struct_attn != "global":
                    mesh_scale = _pack_per_voxel_scalar(proj_feat_pack, "mesh_scale", orig_bsz, x.device)
                    grid_xyz = _dense_grid_proj_world(16, mesh_scale, orig_bsz, device=x.device)
                    struct_cond = _build_proj_cond(context, struct_attn, proj_feat_pack, grid_xyz,
                                                   eval_batch=orig_bsz,
                                                   proj_in_channels=self.structure_model.proj_in_channels,
                                                   stage="ss",
                                                   cond_or_uncond=cond_or_uncond)
                out = self.structure_model(x, timestep, struct_cond)

        if not_struct_mode:
            if mask is not None:
                # Instantly scatter the valid tokens back into a padded rectangular tensor
                padded_out = torch.zeros((B, N, out.feats.shape[-1]), device=x.device, dtype=out.feats.dtype)
                padded_out[mask] = out.feats
                out_tensor = padded_out.transpose(1, 2).unsqueeze(-1)
            else:
                out_tensor = out.feats.view(B, N, -1).transpose(1, 2).unsqueeze(-1)

            if rule and orig_bsz > 1:
                out_tensor = out_tensor.repeat(2, 1, 1, 1)
            return out_tensor
        else:
            out = torch.nn.functional.pad(out, (0, 0, 0, 0, 0, 0, 0, 24))

        return out

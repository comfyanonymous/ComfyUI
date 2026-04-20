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

def manual_cast(obj, dtype):
    return obj.to(dtype=dtype)

class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)


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
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
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
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            device=device, dtype=dtype, operations=operations
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

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor]) -> SparseTensor:
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
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor]) -> SparseTensor:
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
        h = manual_cast(h, self.dtype)
        t = t.to(dtype)
        t_embedder = self.t_embedder.to(dtype)
        t_emb = t_embedder(t, out_dtype = t.dtype)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond = manual_cast(cond, self.dtype)

        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = manual_cast(h, x.dtype)
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
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
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
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            device=device, dtype=dtype, operations=operations
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

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        t_emb = manual_cast(t_emb, self.dtype)
        h = manual_cast(h, self.dtype)
        cond = manual_cast(cond, self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond, self.rope_phases)
        h = manual_cast(h, x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = h.to(next(self.out_layer.parameters()).dtype)
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution] * 3).contiguous()

        return h

def timestep_reshift(t_shifted, old_shift=3.0, new_shift=5.0):
    t_shifted = t_shifted / 1000.0
    t_linear = t_shifted / (old_shift - t_shifted * (old_shift - 1))
    t_new = (new_shift * t_linear) / (1 + (new_shift - 1) * t_linear)
    t_new *= 1000.0
    return t_new

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
        self.img2shape = SLatFlowModel(resolution=resolution, in_channels=in_channels, **args)
        self.shape2txt = None
        if init_txt_model:
            self.shape2txt = SLatFlowModel(resolution=resolution, in_channels=in_channels*2, **args)
        self.img2shape_512 = SLatFlowModel(resolution=32, in_channels=in_channels, **args)
        args.pop("out_channels")
        self.structure_model = SparseStructureFlowModel(resolution=16, in_channels=8, out_channels=8, **args)
        self.guidance_interval = [0.6, 1.0]
        self.guidance_interval_txt = [0.6, 0.9]

    def forward(self, x, timestep, context, **kwargs):
        transformer_options = kwargs.get("transformer_options", {})
        embeds = kwargs.get("embeds")
        if embeds is None:
            raise ValueError("Trellis2.forward requires 'embeds' in kwargs")
        # img2shape.resolution is the latent-grid size, not the input pixel size:
        # 32 -> 512px path, 64 -> 1024px path.
        uses_1024_conditioning = self.img2shape.resolution == 64
        coords = transformer_options.get("coords", None)
        coord_counts = transformer_options.get("coord_counts")
        mode = transformer_options.get("generation_mode", "structure_generation")
        is_512_run = False
        timestep = timestep.to(self.dtype)
        if mode == "shape_generation_512":
            is_512_run = True
            mode = "shape_generation"
        if coords is not None:
            x = x.squeeze(-1).transpose(1, 2)
            not_struct_mode = True
        else:
            mode = "structure_generation"
            not_struct_mode = False

        if uses_1024_conditioning and not_struct_mode and not is_512_run:
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
        dense_out = None

        if not_struct_mode:
            orig_bsz = x.shape[0]
            rule = txt_rule if mode == "texture_generation" else shape_rule

            logical_batch = coord_counts.shape[0] if coord_counts is not None else 1
            if rule and orig_bsz > logical_batch:
                half = orig_bsz // 2
                x_eval = x[half:]
                t_eval = timestep[half:] if timestep.shape[0] > 1 else timestep
                c_eval = cond
            else:
                x_eval = x
                t_eval = timestep
                c_eval = context

            B, N, C = x_eval.shape

            if mode in ["shape_generation", "texture_generation"]:
                if coord_counts is not None:
                    logical_batch = coord_counts.shape[0]
                    if B % logical_batch != 0:
                        raise ValueError(
                            f"Trellis2 coord_counts batch {logical_batch} doesn't divide latent batch {B}"
                        )
                    repeat_factor = B // logical_batch
                    sparse_outs = []
                    active_coord_counts = []
                    if mode == "shape_generation" and repeat_factor > 1:
                        grouped_outs = []
                        grouped_counts = []
                        for i in range(logical_batch):
                            count = int(coord_counts[i].item())
                            coords_i = coords[coords[:, 0] == i].clone()
                            if coords_i.shape[0] != count:
                                raise ValueError(
                                    f"Trellis2 coords rows for batch {i} expected {count}, got {coords_i.shape[0]}"
                                )

                            feat_batches = []
                            coord_batches = []
                            index_batch = []
                            for rep in range(repeat_factor):
                                out_index = rep * logical_batch + i
                                feat_batches.append(x_eval[out_index, :count])
                                coords_rep = coords_i.clone()
                                coords_rep[:, 0] = rep
                                coord_batches.append(coords_rep)
                                index_batch.append(out_index)

                            x_st_i = SparseTensor(
                                feats=torch.cat(feat_batches, dim=0),
                                coords=torch.cat(coord_batches, dim=0).to(torch.int32),
                            )
                            index_tensor = torch.tensor(index_batch, device=x_eval.device, dtype=torch.long)
                            if t_eval.shape[0] > 1:
                                t_i = t_eval.index_select(0, index_tensor)
                            else:
                                t_i = t_eval
                            if c_eval.shape[0] > 1:
                                c_i = c_eval.index_select(0, index_tensor)
                            else:
                                c_i = c_eval

                            if is_512_run:
                                sparse_out = self.img2shape_512(x_st_i, t_i, c_i)
                            else:
                                sparse_out = self.img2shape(x_st_i, t_i, c_i)

                            feats_group, coords_group = sparse_out.to_tensor_list()
                            if len(feats_group) != repeat_factor:
                                raise ValueError(
                                    f"Trellis2 expected {repeat_factor} sparse output groups for batch {i}, got {len(feats_group)}"
                                )
                            for rep, (feats_rep, coords_rep) in enumerate(zip(feats_group, coords_group)):
                                if feats_rep.shape[0] != count:
                                    raise ValueError(
                                        f"Trellis2 sparse output rows for batch {i} rep {rep} expected {count}, got {feats_rep.shape[0]}"
                                    )
                                if coords_rep.shape[0] != count:
                                    raise ValueError(
                                        f"Trellis2 sparse output coords for batch {i} rep {rep} expected {count}, got {coords_rep.shape[0]}"
                                    )
                            grouped_outs.append(feats_group)
                            grouped_counts.append(count)

                        for rep in range(repeat_factor):
                            for i in range(logical_batch):
                                sparse_outs.append(grouped_outs[i][rep])
                                active_coord_counts.append(grouped_counts[i])
                    else:
                        for rep in range(repeat_factor):
                            for i in range(logical_batch):
                                out_index = rep * logical_batch + i
                                count = int(coord_counts[i].item())
                                coords_i = coords[coords[:, 0] == i].clone()
                                if coords_i.shape[0] != count:
                                    raise ValueError(
                                        f"Trellis2 coords rows for batch {i} expected {count}, got {coords_i.shape[0]}"
                                    )
                                coords_i[:, 0] = 0
                                feats_i = x_eval[out_index, :count]
                                x_st_i = SparseTensor(feats=feats_i, coords=coords_i.to(torch.int32))
                                t_i = t_eval[out_index].unsqueeze(0) if t_eval.shape[0] > 1 else t_eval
                                c_i = c_eval[out_index].unsqueeze(0) if c_eval.shape[0] > 1 else c_eval

                                if mode == "shape_generation":
                                    if is_512_run:
                                        sparse_out = self.img2shape_512(x_st_i, t_i, c_i)
                                    else:
                                        sparse_out = self.img2shape(x_st_i, t_i, c_i)
                                else:
                                    slat = transformer_options.get("shape_slat")
                                    if slat is None:
                                        raise ValueError("shape_slat can't be None")
                                    if slat.ndim == 3:
                                        if slat.shape[0] != logical_batch:
                                            raise ValueError(
                                                f"shape_slat batch {slat.shape[0]} doesn't match coord_counts batch {logical_batch}"
                                            )
                                        if slat.shape[1] < count:
                                            raise ValueError(
                                                f"shape_slat tokens {slat.shape[1]} can't cover coord count {count} for batch {i}"
                                            )
                                        slat_feats = slat[i, :count].to(x_st_i.device)
                                    else:
                                        slat_feats = slat[:count].to(x_st_i.device)
                                    x_st_i = x_st_i.replace(feats=torch.cat([x_st_i.feats, slat_feats], dim=-1))
                                    sparse_out = self.shape2txt(x_st_i, t_i, c_i)

                                sparse_outs.append(sparse_out.feats)
                                active_coord_counts.append(count)

                    out_channels = sparse_outs[0].shape[-1]
                    padded = sparse_outs[0].new_zeros((B, N, out_channels))
                    for out_index, (count, feats_i) in enumerate(zip(active_coord_counts, sparse_outs)):
                        padded[out_index, :count] = feats_i
                    dense_out = padded.transpose(1, 2).unsqueeze(-1)
                elif coords.shape[0] == N:
                    feats_flat = x_eval.reshape(-1, C)
                    coords_list = []
                    for i in range(B):
                        c = coords.clone()
                        c[:, 0] = i
                        coords_list.append(c)
                    batched_coords = torch.cat(coords_list, dim=0)
                elif coords.shape[0] == B * N:
                    feats_flat = x_eval.reshape(-1, C)
                    batched_coords = coords
                else:
                    raise ValueError(
                        f"Trellis2 expected coords rows {N} or {B * N}, got {coords.shape[0]}"
                    )
            else:
                batched_coords = coords
                feats_flat = x_eval

            if dense_out is None:
                x_st = SparseTensor(feats=feats_flat, coords=batched_coords.to(torch.int32))

        if dense_out is not None:
            out = dense_out
        elif mode == "shape_generation":
            if is_512_run:
                out = self.img2shape_512(x_st, t_eval, c_eval)
            else:
                out = self.img2shape(x_st, t_eval, c_eval)
        elif mode == "texture_generation":
            if self.shape2txt is None:
                raise ValueError("Checkpoint for Trellis2 doesn't include texture generation!")
            slat = transformer_options.get("shape_slat")
            if slat is None:
                raise ValueError("shape_slat can't be None")

            if slat.ndim == 3:
                if coord_counts is not None:
                    logical_batch = coord_counts.shape[0]
                    if slat.shape[0] != logical_batch:
                        raise ValueError(
                            f"shape_slat batch {slat.shape[0]} doesn't match coord_counts batch {logical_batch}"
                        )
                    if B % logical_batch != 0:
                        raise ValueError(
                            f"Trellis2 coord_counts batch {logical_batch} doesn't divide latent batch {B}"
                        )
                    repeat_factor = B // logical_batch
                    slat_list = []
                    for _ in range(repeat_factor):
                        for i in range(logical_batch):
                            count = int(coord_counts[i].item())
                            if slat.shape[1] < count:
                                raise ValueError(
                                    f"shape_slat tokens {slat.shape[1]} can't cover coord count {count} for batch {i}"
                                )
                            slat_list.append(slat[i, :count])
                    slat_feats_batched = torch.cat(slat_list, dim=0).to(x_st.device)
                else:
                    if slat.shape[0] != B:
                        raise ValueError(f"shape_slat batch {slat.shape[0]} doesn't match latent batch {B}")
                    if slat.shape[1] != N:
                        raise ValueError(f"shape_slat tokens {slat.shape[1]} doesn't match latent tokens {N}")
                    slat_feats_batched = slat.reshape(B * N, -1).to(x_st.device)
            else:
                base_slat_feats = slat[:N]
                slat_feats_batched = base_slat_feats.repeat(B, 1).to(x_st.device)
            x_st = x_st.replace(feats=torch.cat([x_st.feats, slat_feats_batched], dim=-1))
            out = self.shape2txt(x_st, t_eval, c_eval)
        else: # structure
            orig_bsz = x.shape[0]
            cond_or_uncond = transformer_options.get("cond_or_uncond") or []
            batch_groups = len(cond_or_uncond) if len(cond_or_uncond) > 0 and orig_bsz % len(cond_or_uncond) == 0 else 1
            logical_batch = orig_bsz // batch_groups
            if logical_batch > 1:
                x_groups = x.reshape(batch_groups, logical_batch, *x.shape[1:])
                if timestep.shape[0] > 1:
                    t_groups = timestep.reshape(batch_groups, logical_batch, *timestep.shape[1:])
                else:
                    t_groups = timestep
                c_groups = context.reshape(batch_groups, logical_batch, *context.shape[1:])

                if shape_rule and batch_groups > 1:
                    selected_group_indices = [batch_groups - 1]
                else:
                    selected_group_indices = list(range(batch_groups))

                out_groups = []
                for sample_index in range(logical_batch):
                    if shape_rule and batch_groups > 1:
                        half = orig_bsz // 2
                        x_i = x[half + sample_index].unsqueeze(0)
                        if timestep.shape[0] > 1:
                            t_i = timestep[half + sample_index].unsqueeze(0)
                        else:
                            t_i = timestep
                        if cond.shape[0] > 1:
                            c_i = cond[sample_index].unsqueeze(0)
                        else:
                            c_i = cond
                    else:
                        x_i = x_groups[selected_group_indices, sample_index]
                        if timestep.shape[0] > 1:
                            t_i = t_groups[selected_group_indices, sample_index]
                        else:
                            t_i = timestep
                        c_i = c_groups[selected_group_indices, sample_index]
                    out_groups.append(self.structure_model(x_i, t_i, c_i))

                out = out_groups[0].new_zeros((orig_bsz, *out_groups[0].shape[1:]))
                for sample_index, out_sample in enumerate(out_groups):
                    if shape_rule and batch_groups > 1:
                        repeated = out_sample[0]
                        for group_index in range(batch_groups):
                            out[group_index * logical_batch + sample_index] = repeated
                    else:
                        for local_group_index, group_index in enumerate(selected_group_indices):
                            out[group_index * logical_batch + sample_index] = out_sample[local_group_index]
            else:
                if shape_rule and orig_bsz > 1:
                    half = orig_bsz // 2
                    x = x[half:]
                    timestep = timestep[half:] if timestep.shape[0] > 1 else timestep
                out = self.structure_model(x, timestep, cond if shape_rule and orig_bsz > 1 else context)
                if shape_rule and orig_bsz > 1:
                    out = out.repeat(2, 1, 1, 1, 1)

        if not_struct_mode:
            if dense_out is None:
                out = out.feats
                out = out.view(B, N, -1).transpose(1, 2).unsqueeze(-1)
            if rule and orig_bsz > B:
                out = out.repeat(orig_bsz // B, 1, 1, 1)
        return out

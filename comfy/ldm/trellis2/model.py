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
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio)),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), channels),
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
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

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
        rope_freq: Tuple[float, float] = (1.0, 10000.0)
    ):
        super().__init__()
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
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
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = SparseRotaryPositionEmbedder(self.head_dim, rope_freq=rope_freq)

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

    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
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
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

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
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        self.input_layer = SparseLinear(in_channels, model_channels)

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
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = SparseLinear(model_channels, out_channels)

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
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

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
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

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
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
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
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

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

        self.t_embedder = TimestepEmbedder(model_channels, operations=operations)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        pos_embedder = RotaryPositionEmbedder(self.model_channels // self.num_heads, 3)
        coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution] * 3], indexing='ij')
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        rope_phases = pos_embedder(coords)
        self.register_buffer("rope_phases", rope_phases)

        if pe_mode != "rope":
            self.rope_phases = None

        self.input_layer = nn.Linear(in_channels, model_channels)

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
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels)

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
        # for some reason it passes num_heads = -1
        if num_heads == -1:
            num_heads = 12
        args = {
            "out_channels":out_channels, "num_blocks":num_blocks, "cond_channels" :cond_channels,
            "model_channels":model_channels, "num_heads":num_heads, "mlp_ratio": mlp_ratio, "share_mod": share_mod,
            "qk_rms_norm": qk_rms_norm, "qk_rms_norm_cross": qk_rms_norm_cross, "device": device, "dtype": dtype, "operations": operations
        }
        self.img2shape = SLatFlowModel(resolution=resolution, in_channels=in_channels, **args)
        if init_txt_model:
            self.shape2txt = SLatFlowModel(resolution=resolution, in_channels=in_channels*2, **args)
        args.pop("out_channels")
        self.structure_model = SparseStructureFlowModel(resolution=16, in_channels=8, out_channels=8, **args)
        self.guidance_interval = [0.6, 1.0]
        self.guidance_interval_txt = [0.6, 0.9]

    def forward(self, x, timestep, context, **kwargs):
        transformer_options = kwargs.get("transformer_options", {})
        embeds = kwargs.get("embeds")
        if embeds is None:
            raise ValueError("Trellis2.forward requires 'embeds' in kwargs")
        is_1024 = self.img2shape.resolution == 1024
        if is_1024:
            context = embeds
        coords = transformer_options.get("coords", None)
        mode = transformer_options.get("generation_mode", "structure_generation")
        if coords is not None:
            x = x.squeeze(-1).transpose(1, 2)
            not_struct_mode = True
        else:
            mode = "structure_generation"
            not_struct_mode = False
        sigmas = transformer_options.get("sigmas")[0].item()
        if sigmas < 1.00001:
            timestep *= 1000.0
        cond = context.chunk(2)[1]
        shape_rule = sigmas < self.guidance_interval[0] or sigmas > self.guidance_interval[1]
        txt_rule = sigmas < self.guidance_interval_txt[0] or sigmas > self.guidance_interval_txt[1]

        if not_struct_mode:
            B, N, C = x.shape

            if mode == "shape_generation":
                feats_flat = x.reshape(-1, C)

                # 3. inflate coords [N, 4] -> [B*N, 4]
                coords_list = []
                for i in range(B):
                    c = coords.clone()
                    c[:, 0] = i
                    coords_list.append(c)

                batched_coords = torch.cat(coords_list, dim=0)
            else: # TODO: texture
                # may remove the else if texture doesn't require special handling
                batched_coords = coords
                feats_flat = x
            x = SparseTensor(feats=feats_flat, coords=batched_coords.to(torch.int32))

        if mode == "shape_generation":
            # TODO
            out = self.img2shape(x, timestep, context)
        elif mode == "texture_generation":
            if self.shape2txt is None:
                raise ValueError("Checkpoint for Trellis2 doesn't include texture generation!")
            out = self.shape2txt(x, timestep, context if not txt_rule else cond)
        else: # structure
            timestep = timestep_reshift(timestep)
            orig_bsz = x.shape[0]
            if shape_rule:
                x = x[0].unsqueeze(0)
                timestep = timestep[0].unsqueeze(0)
            out = self.structure_model(x, timestep, context if not shape_rule else cond)
            if shape_rule:
                out = out.repeat(orig_bsz, 1, 1, 1, 1)

        if not_struct_mode:
            out = out.feats
            if mode == "shape_generation":
                out = out.view(B, N, -1).transpose(1, 2).unsqueeze(-1)
        return out

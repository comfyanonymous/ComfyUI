import torch
import torch.nn as nn
from typing import Tuple, Union, Optional

try:
    import flash_attn
    if hasattr(flash_attn, '__version__') and int(flash_attn.__version__[0]) == 2:
        from flash_attn.flash_attn_interface import flash_attn_kvpacked_func
        from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention
    else:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
        from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention
except Exception as e:
    print(f'flash_attn import failed: {e}')


def reshape_for_broadcast(freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]], x: torch.Tensor, head_first=False):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (x.shape[-2], x.shape[-1]), f'freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}'
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (x.shape[1], x.shape[-1]), f'freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}'
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (x.shape[-2], x.shape[-1]), f'freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}'
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}'
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: Optional[torch.Tensor],
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
        head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Precomputed frequency tensor for complex exponentials.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)    # [S, D]
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        if xk is not None:
            xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(xq.device)   # [S, D//2] --> [1, S, 1, D//2]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        if xk is not None:
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # [B, S, H, D//2]
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


class FlashSelfMHAModified(nn.Module):
    """
    Use QK Normalization.
    """
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=True,
                 qk_norm=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 device=None,
                 dtype=None,
                 norm_layer=nn.LayerNorm,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias, **factory_kwargs)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.inner_attn = FlashSelfAttention(attention_dropout=attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis_img=None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s, d = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)  # [b, s, 3, h, d]
        q, k, v = qkv.unbind(dim=2) # [b, s, h, d]
        q = self.q_norm(q).half()   # [b, s, h, d]
        k = self.k_norm(k).half()

        # Apply RoPE if needed
        if freqs_cis_img is not None:
            qq, kk = apply_rotary_emb(q, k, freqs_cis_img)
            assert qq.shape == q.shape and kk.shape == k.shape, f'qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}'
            q, k = qq, kk

        qkv = torch.stack([q, k, v], dim=2)     # [b, s, 3, h, d]
        context = self.inner_attn(qkv)
        out = self.out_proj(context.view(b, s, d))
        out = self.proj_drop(out)

        out_tuple = (out,)

        return out_tuple


class FlashCrossMHAModified(nn.Module):
    """
    Use QK Normalization.
    """
    def __init__(self,
                 qdim,
                 kdim,
                 num_heads,
                 qkv_bias=True,
                 qk_norm=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 device=None,
                 dtype=None,
                 norm_layer=nn.LayerNorm,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.kv_proj = nn.Linear(kdim, 2 * qdim, bias=qkv_bias, **factory_kwargs)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

        self.inner_attn = FlashCrossAttention(attention_dropout=attn_drop)
        self.out_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, freqs_cis_img=None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num_heads * head_dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // num_heads), RoPE for image
        """
        b, s1, _ = x.shape     # [b, s1, D]
        _, s2, _ = y.shape     # [b, s2, 1024]

        q = self.q_proj(x).view(b, s1, self.num_heads, self.head_dim)       # [b, s1, h, d]
        kv = self.kv_proj(y).view(b, s2, 2, self.num_heads, self.head_dim)  # [b, s2, 2, h, d]
        k, v = kv.unbind(dim=2)                 # [b, s2, h, d]
        q = self.q_norm(q).half()               # [b, s1, h, d]
        k = self.k_norm(k).half()               # [b, s2, h, d]

        # Apply RoPE if needed
        if freqs_cis_img is not None:
            qq, _ = apply_rotary_emb(q, None, freqs_cis_img)
            assert qq.shape == q.shape, f'qq: {qq.shape}, q: {q.shape}'
            q = qq                              # [b, s1, h, d]
        kv = torch.stack([k, v], dim=2)         # [b, s1, 2, h, d]
        context = self.inner_attn(q, kv)        # [b, s1, h, d]
        context = context.view(b, s1, -1)       # [b, s1, D]

        out = self.out_proj(context)
        out = self.proj_drop(out)

        out_tuple = (out,)

        return out_tuple


class CrossAttention(nn.Module):
    """
    Use QK Normalization.
    """
    def __init__(self,
                 qdim,
                 kdim,
                 num_heads,
                 qkv_bias=True,
                 qk_norm=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 device=None,
                 dtype=None,
                 norm_layer=nn.LayerNorm,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.kv_proj = nn.Linear(kdim, 2 * qdim, bias=qkv_bias, **factory_kwargs)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, freqs_cis_img=None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s1, c = x.shape     # [b, s1, D]
        _, s2, c = y.shape     # [b, s2, 1024]

        q = self.q_proj(x).view(b, s1, self.num_heads, self.head_dim)   # [b, s1, h, d]
        kv = self.kv_proj(y).view(b, s2, 2, self.num_heads, self.head_dim)    # [b, s2, 2, h, d]
        k, v = kv.unbind(dim=2) # [b, s, h, d]
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if needed
        if freqs_cis_img is not None:
            qq, _ = apply_rotary_emb(q, None, freqs_cis_img)
            assert qq.shape == q.shape, f'qq: {qq.shape}, q: {q.shape}'
            q = qq

        q = q * self.scale
        q = q.transpose(-2, -3).contiguous()        # q ->  B, L1, H, C - B, H, L1, C
        k = k.permute(0, 2, 3, 1).contiguous()      # k ->  B, L2, H, C - B, H, C, L2
        attn = q @ k                                # attn -> B, H, L1, L2
        attn = attn.softmax(dim=-1)                 # attn -> B, H, L1, L2
        attn = self.attn_drop(attn)
        x = attn @ v.transpose(-2, -3)              # v -> B, L2, H, C - B, H, L2, C    x-> B, H, L1, C
        context = x.transpose(1, 2)                 # context -> B, H, L1, C - B, L1, H, C

        context = context.contiguous().view(b, s1, -1)

        out = self.out_proj(context)  # context.reshape - B, L1, -1
        out = self.proj_drop(out)

        out_tuple = (out,)

        return out_tuple


class Attention(nn.Module):
    """
    We rename some layer names to align with flash attention
    """
    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=False, attn_drop=0., proj_drop=0.,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // num_heads
        # This assertion is aligned with flash attention
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        # qkv --> Wqkv
        self.Wqkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis_img=None):
        B, N, C = x.shape
        qkv = self.Wqkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)   # [3, b, h, s, d]
        q, k, v = qkv.unbind(0)     # [b, h, s, d]
        q = self.q_norm(q)          # [b, h, s, d]
        k = self.k_norm(k)          # [b, h, s, d]

        # Apply RoPE if needed
        if freqs_cis_img is not None:
            qq, kk = apply_rotary_emb(q, k, freqs_cis_img, head_first=True)
            assert qq.shape == q.shape and kk.shape == k.shape, \
                f'qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}'
            q, k = qq, kk

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)              # [b, h, s, d] @ [b, h, d, s]
        attn = attn.softmax(dim=-1)                 # [b, h, s, s]
        attn = self.attn_drop(attn)
        x = attn @ v                                # [b, h, s, d]

        x = x.transpose(1, 2).reshape(B, N, C)      # [b, s, h, d]
        x = self.out_proj(x)
        x = self.proj_drop(x)

        out_tuple = (x,)

        return out_tuple

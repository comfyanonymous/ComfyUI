import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets

        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        return x * random_tensor

class MLP(nn.Module):
    def __init__(self, width: int, ratio: int = 4, drop_path_rate: float = 0):
        super().__init__()
        self.gelu = nn.GELU()
        self.c_fc = nn.Linear(width, width * ratio)
        self.c_proj = nn.Linear(width * ratio, width)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class QKVMultiheadAttention(nn.Module):
    def __init__(
        self,
        heads: int,
        n_ctx: int,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool,
        norm_layer = nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
 
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)

        self.attention = QKVMultiheadAttention(
            heads = heads,
            n_ctx = n_ctx,
            width = width,
            norm_layer = norm_layer,
            qk_norm = qk_norm
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.drop_path(self.c_proj(x))
        return x


class ResAttnBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)
        self.ln_2 = norm_layer(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, n_ctx: int, heads: int, width: int, depth: int,
                 qkv_bias: bool = True, qk_norm: bool = False, drop_path_rate: float = 0.0):
        super().__init__()

        self.resblocks = nn.ModuleList([
            ResAttnBlock(n_ctx = n_ctx,
                         heads = heads,
                         width = width,
                         qkv_bias = qkv_bias,
                         qk_norm = qk_norm,
                         drop_path_rate = drop_path_rate)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for resnet in self.resblocks:
            x = resnet(x)

        return x

if __name__ == "__main__":
    torch.manual_seed(2025)
    model = Transformer(512, 8, 224, 3)
    outputs = model(x =  torch.randn(1, 512, 224))
    print(outputs)

import torch
import torch.nn as nn
from comfy.ldm.modules.attention import optimized_attention_masked


class LayerNormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.conv = operations.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias, device=device, dtype=dtype)
        self.layer_norm = operations.LayerNorm(out_channels, elementwise_affine=True, device=device, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        return torch.nn.functional.gelu(self.layer_norm(x.transpose(-2, -1)).transpose(-2, -1))

class LayerGroupNormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.conv = operations.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias, device=device, dtype=dtype)
        self.layer_norm = operations.GroupNorm(num_groups=out_channels, num_channels=out_channels, affine=True, device=device, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        return torch.nn.functional.gelu(self.layer_norm(x))

class ConvNoNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.conv = operations.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        return torch.nn.functional.gelu(x)


class ConvFeatureEncoder(nn.Module):
    def __init__(self, conv_dim, conv_bias=False, conv_norm=True, dtype=None, device=None, operations=None):
        super().__init__()
        if conv_norm:
            self.conv_layers = nn.ModuleList([
                LayerNormConv(1, conv_dim, kernel_size=10, stride=5, bias=True, device=device, dtype=dtype, operations=operations),
                LayerNormConv(conv_dim, conv_dim, kernel_size=3, stride=2, bias=conv_bias, device=device, dtype=dtype, operations=operations),
                LayerNormConv(conv_dim, conv_dim, kernel_size=3, stride=2, bias=conv_bias, device=device, dtype=dtype, operations=operations),
                LayerNormConv(conv_dim, conv_dim, kernel_size=3, stride=2, bias=conv_bias, device=device, dtype=dtype, operations=operations),
                LayerNormConv(conv_dim, conv_dim, kernel_size=3, stride=2, bias=conv_bias, device=device, dtype=dtype, operations=operations),
                LayerNormConv(conv_dim, conv_dim, kernel_size=2, stride=2, bias=conv_bias, device=device, dtype=dtype, operations=operations),
                LayerNormConv(conv_dim, conv_dim, kernel_size=2, stride=2, bias=conv_bias, device=device, dtype=dtype, operations=operations),
            ])
        else:
            self.conv_layers = nn.ModuleList([
                LayerGroupNormConv(1, conv_dim, kernel_size=10, stride=5, bias=conv_bias, device=device, dtype=dtype, operations=operations),
                ConvNoNorm(conv_dim, conv_dim, kernel_size=3, stride=2, bias=conv_bias, device=device, dtype=dtype, operations=operations),
                ConvNoNorm(conv_dim, conv_dim, kernel_size=3, stride=2, bias=conv_bias, device=device, dtype=dtype, operations=operations),
                ConvNoNorm(conv_dim, conv_dim, kernel_size=3, stride=2, bias=conv_bias, device=device, dtype=dtype, operations=operations),
                ConvNoNorm(conv_dim, conv_dim, kernel_size=3, stride=2, bias=conv_bias, device=device, dtype=dtype, operations=operations),
                ConvNoNorm(conv_dim, conv_dim, kernel_size=2, stride=2, bias=conv_bias, device=device, dtype=dtype, operations=operations),
                ConvNoNorm(conv_dim, conv_dim, kernel_size=2, stride=2, bias=conv_bias, device=device, dtype=dtype, operations=operations),
            ])

    def forward(self, x):
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x.transpose(1, 2)


class FeatureProjection(nn.Module):
    def __init__(self, conv_dim, embed_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.layer_norm = operations.LayerNorm(conv_dim, eps=1e-05, device=device, dtype=dtype)
        self.projection = operations.Linear(conv_dim, embed_dim, device=device, dtype=dtype)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.projection(x)
        return x


class PositionalConvEmbedding(nn.Module):
    def __init__(self, embed_dim=768, kernel_size=128, groups=16):
        super().__init__()
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.conv = torch.nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        self.activation = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, :-1]
        x = self.activation(x)
        x = x.transpose(1, 2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4.0,
        do_stable_layer_norm=True,
        dtype=None, device=None, operations=None
    ):
        super().__init__()

        self.pos_conv_embed = PositionalConvEmbedding(embed_dim=embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                do_stable_layer_norm=do_stable_layer_norm,
                device=device, dtype=dtype, operations=operations
            )
            for _ in range(num_layers)
        ])

        self.layer_norm = operations.LayerNorm(embed_dim, eps=1e-05, device=device, dtype=dtype)
        self.do_stable_layer_norm = do_stable_layer_norm

    def forward(self, x, mask=None):
        x = x + self.pos_conv_embed(x)
        all_x = ()
        if not self.do_stable_layer_norm:
            x = self.layer_norm(x)
        for layer in self.layers:
            all_x += (x,)
            x = layer(x, mask)
        if self.do_stable_layer_norm:
            x = self.layer_norm(x)
        all_x += (x,)
        return x, all_x


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.k_proj = operations.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.v_proj = operations.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.q_proj = operations.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.out_proj = operations.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)

    def forward(self, x, mask=None):
        assert (mask is None)  # TODO?
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        out = optimized_attention_masked(q, k, v, self.num_heads)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, dtype=None, device=None, operations=None):
        super().__init__()
        self.intermediate_dense = operations.Linear(embed_dim, int(embed_dim * mlp_ratio), device=device, dtype=dtype)
        self.output_dense = operations.Linear(int(embed_dim * mlp_ratio), embed_dim, device=device, dtype=dtype)

    def forward(self, x):
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.output_dense(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        do_stable_layer_norm=True,
        dtype=None, device=None, operations=None
    ):
        super().__init__()

        self.attention = Attention(embed_dim, num_heads, device=device, dtype=dtype, operations=operations)

        self.layer_norm = operations.LayerNorm(embed_dim, device=device, dtype=dtype)
        self.feed_forward = FeedForward(embed_dim, mlp_ratio, device=device, dtype=dtype, operations=operations)
        self.final_layer_norm = operations.LayerNorm(embed_dim, device=device, dtype=dtype)
        self.do_stable_layer_norm = do_stable_layer_norm

    def forward(self, x, mask=None):
        residual = x
        if self.do_stable_layer_norm:
            x = self.layer_norm(x)
        x = self.attention(x, mask=mask)
        x = residual + x
        if not self.do_stable_layer_norm:
            x = self.layer_norm(x)
            return self.final_layer_norm(x + self.feed_forward(x))
        else:
            return x + self.feed_forward(self.final_layer_norm(x))


class Wav2Vec2Model(nn.Module):
    """Complete Wav2Vec 2.0 model."""

    def __init__(
        self,
        embed_dim=1024,
        final_dim=256,
        num_heads=16,
        num_layers=24,
        conv_norm=True,
        conv_bias=True,
        do_normalize=True,
        do_stable_layer_norm=True,
        dtype=None, device=None, operations=None
    ):
        super().__init__()

        conv_dim = 512
        self.feature_extractor = ConvFeatureEncoder(conv_dim, conv_norm=conv_norm, conv_bias=conv_bias, device=device, dtype=dtype, operations=operations)
        self.feature_projection = FeatureProjection(conv_dim, embed_dim, device=device, dtype=dtype, operations=operations)

        self.masked_spec_embed = nn.Parameter(torch.empty(embed_dim, device=device, dtype=dtype))
        self.do_normalize = do_normalize

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            do_stable_layer_norm=do_stable_layer_norm,
            device=device, dtype=dtype, operations=operations
        )

    def forward(self, x, mask_time_indices=None, return_dict=False):
        x = torch.mean(x, dim=1)

        if self.do_normalize:
            x = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)

        features = self.feature_extractor(x)
        features = self.feature_projection(features)
        batch_size, seq_len, _ = features.shape

        x, all_x = self.encoder(features)
        return x, all_x

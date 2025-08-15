import torch
from comfy.text_encoders.bert import BertAttention
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention_for_device


class Dino2AttentionOutput(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.dense = operations.Linear(input_dim, output_dim, dtype=dtype, device=device)

    def forward(self, x):
        return self.dense(x)


class Dino2AttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, heads, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.attention = BertAttention(embed_dim, heads, dtype, device, operations)
        self.output = Dino2AttentionOutput(embed_dim, embed_dim, layer_norm_eps, dtype, device, operations)

    def forward(self, x, mask, optimized_attention):
        return self.output(self.attention(x, mask, optimized_attention))


class LayerScale(torch.nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        self.lambda1 = torch.nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x):
        return x * comfy.model_management.cast_to_device(self.lambda1, x.device, x.dtype)


class SwiGLUFFN(torch.nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        in_features = out_features = dim
        hidden_features = int(dim * 4)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = operations.Linear(in_features, 2 * hidden_features, bias=True, device=device, dtype=dtype)
        self.weights_out = operations.Linear(hidden_features, out_features, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        x = self.weights_in(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = torch.nn.functional.silu(x1) * x2
        return self.weights_out(x)


class Dino2Block(torch.nn.Module):
    def __init__(self, dim, num_heads, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.attention = Dino2AttentionBlock(dim, num_heads, layer_norm_eps, dtype, device, operations)
        self.layer_scale1 = LayerScale(dim, dtype, device, operations)
        self.layer_scale2 = LayerScale(dim, dtype, device, operations)
        self.mlp = SwiGLUFFN(dim, dtype, device, operations)
        self.norm1 = operations.LayerNorm(dim, eps=layer_norm_eps, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(dim, eps=layer_norm_eps, dtype=dtype, device=device)

    def forward(self, x, optimized_attention):
        x = x + self.layer_scale1(self.attention(self.norm1(x), None, optimized_attention))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class Dino2Encoder(torch.nn.Module):
    def __init__(self, dim, num_heads, layer_norm_eps, num_layers, dtype, device, operations):
        super().__init__()
        self.layer = torch.nn.ModuleList([Dino2Block(dim, num_heads, layer_norm_eps, dtype, device, operations) for _ in range(num_layers)])

    def forward(self, x, intermediate_output=None):
        optimized_attention = optimized_attention_for_device(x.device, False, small_input=True)

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layer) + intermediate_output

        intermediate = None
        for i, l in enumerate(self.layer):
            x = l(x, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class Dino2PatchEmbeddings(torch.nn.Module):
    def __init__(self, dim, num_channels=3, patch_size=14, image_size=518, dtype=None, device=None, operations=None):
        super().__init__()
        self.projection = operations.Conv2d(
            in_channels=num_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
            dtype=dtype,
            device=device
        )

    def forward(self, pixel_values):
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class Dino2Embeddings(torch.nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        patch_size = 14
        image_size = 518

        self.patch_embeddings = Dino2PatchEmbeddings(dim, patch_size=patch_size, image_size=image_size, dtype=dtype, device=device, operations=operations)
        self.position_embeddings = torch.nn.Parameter(torch.empty(1, (image_size // patch_size) ** 2 + 1, dim, dtype=dtype, device=device))
        self.cls_token = torch.nn.Parameter(torch.empty(1, 1, dim, dtype=dtype, device=device))
        self.mask_token = torch.nn.Parameter(torch.empty(1, dim, dtype=dtype, device=device))

    def forward(self, pixel_values):
        x = self.patch_embeddings(pixel_values)
        # TODO: mask_token?
        x = torch.cat((self.cls_token.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1), x), dim=1)
        x = x + comfy.model_management.cast_to_device(self.position_embeddings, x.device, x.dtype)
        return x


class Dinov2Model(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        num_layers = config_dict["num_hidden_layers"]
        dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        layer_norm_eps = config_dict["layer_norm_eps"]

        self.embeddings = Dino2Embeddings(dim, dtype, device, operations)
        self.encoder = Dino2Encoder(dim, heads, layer_norm_eps, num_layers, dtype, device, operations)
        self.layernorm = operations.LayerNorm(dim, eps=layer_norm_eps, dtype=dtype, device=device)

    def forward(self, pixel_values, attention_mask=None, intermediate_output=None):
        x = self.embeddings(pixel_values)
        x, i = self.encoder(x, intermediate_output=intermediate_output)
        x = self.layernorm(x)
        pooled_output = x[:, 0, :]
        return x, i, pooled_output, None

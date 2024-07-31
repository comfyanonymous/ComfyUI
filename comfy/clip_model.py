import torch
from comfy.ldm.modules.attention import optimized_attention_for_device
import comfy.ops

class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device, operations):
        super().__init__()

        self.heads = heads
        self.q_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

        self.out_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None, optimized_attention=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        out = optimized_attention(q, k, v, self.heads, mask)
        return self.out_proj(out)

ACTIVATIONS = {"quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
               "gelu": torch.nn.functional.gelu,
}

class CLIPMLP(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_size, activation, dtype, device, operations):
        super().__init__()
        self.fc1 = operations.Linear(embed_dim, intermediate_size, bias=True, dtype=dtype, device=device)
        self.activation = ACTIVATIONS[activation]
        self.fc2 = operations.Linear(intermediate_size, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class CLIPLayer(torch.nn.Module):
    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations):
        super().__init__()
        self.layer_norm1 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device, operations)
        self.layer_norm2 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, device, operations)

    def forward(self, x, mask=None, optimized_attention=None):
        x += self.self_attn(self.layer_norm1(x), mask, optimized_attention)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations):
        super().__init__()
        self.layers = torch.nn.ModuleList([CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations) for i in range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate

class CLIPEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None, operations=None):
        super().__init__()
        self.token_embedding = operations.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)
        self.position_embedding = operations.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, dtype=torch.float32):
        return self.token_embedding(input_tokens, out_dtype=dtype) + comfy.ops.cast_to(self.position_embedding.weight, dtype=dtype, device=input_tokens.device)


class CLIPTextModel_(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        self.eos_token_id = config_dict["eos_token_id"]

        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, dtype=dtype, device=device, operations=operations)
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations)
        self.final_layer_norm = operations.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, attention_mask=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=torch.float32):
        x = self.embeddings(input_tokens, dtype=dtype)
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), float("-inf"))

        causal_mask = torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device).fill_(float("-inf")).triu_(1)
        if mask is not None:
            mask += causal_mask
        else:
            mask = causal_mask

        x, i = self.encoder(x, mask=mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)

        pooled_output = x[torch.arange(x.shape[0], device=x.device), (torch.round(input_tokens).to(dtype=torch.int, device=x.device) == self.eos_token_id).int().argmax(dim=-1),]
        return x, i, pooled_output

class CLIPTextModel(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device, operations)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = operations.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


class CLIPVisionEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim, num_channels=3, patch_size=14, image_size=224, dtype=None, device=None, operations=None):
        super().__init__()
        self.class_embedding = torch.nn.Parameter(torch.empty(embed_dim, dtype=dtype, device=device))

        self.patch_embedding = operations.Conv2d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
            dtype=dtype,
            device=device
        )

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_patches + 1
        self.position_embedding = operations.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, pixel_values):
        embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return torch.cat([comfy.ops.cast_to_input(self.class_embedding, embeds).expand(pixel_values.shape[0], 1, -1), embeds], dim=1) + comfy.ops.cast_to_input(self.position_embedding.weight, embeds)


class CLIPVision(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]

        self.embeddings = CLIPVisionEmbeddings(embed_dim, config_dict["num_channels"], config_dict["patch_size"], config_dict["image_size"], dtype=dtype, device=device, operations=operations)
        self.pre_layrnorm = operations.LayerNorm(embed_dim)
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations)
        self.post_layernorm = operations.LayerNorm(embed_dim)

    def forward(self, pixel_values, attention_mask=None, intermediate_output=None):
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)
        #TODO: attention_mask?
        x, i = self.encoder(x, mask=None, intermediate_output=intermediate_output)
        pooled_output = self.post_layernorm(x[:, 0, :])
        return x, i, pooled_output

class CLIPVisionModelProjection(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.vision_model = CLIPVision(config_dict, dtype, device, operations)
        self.visual_projection = operations.Linear(config_dict["hidden_size"], config_dict["projection_dim"], bias=False)

    def forward(self, *args, **kwargs):
        x = self.vision_model(*args, **kwargs)
        out = self.visual_projection(x[2])
        return (x[0], x[1], out)

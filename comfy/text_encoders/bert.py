import torch
from comfy.ldm.modules.attention import optimized_attention_for_device
import comfy.ops

class BertAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device, operations):
        super().__init__()

        self.heads = heads
        self.query = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.key = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.value = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)


    def forward(self, x, mask=None, optimized_attention=None):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        out = optimized_attention(q, k, v, self.heads, mask)
        return out

class BertOutput(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.dense = operations.Linear(input_dim, output_dim, dtype=dtype, device=device)
        self.LayerNorm = operations.LayerNorm(output_dim, eps=layer_norm_eps, dtype=dtype, device=device)
        # self.dropout = nn.Dropout(0.0)

    def forward(self, x, y):
        x = self.dense(x)
        # hidden_states = self.dropout(hidden_states)
        x = self.LayerNorm(x + y)
        return x

class BertAttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, heads, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.self = BertAttention(embed_dim, heads, dtype, device, operations)
        self.output = BertOutput(embed_dim, embed_dim, layer_norm_eps, dtype, device, operations)

    def forward(self, x, mask, optimized_attention):
        y = self.self(x, mask, optimized_attention)
        return self.output(y, x)

class BertIntermediate(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_dim, dtype, device, operations):
        super().__init__()
        self.dense = operations.Linear(embed_dim, intermediate_dim, dtype=dtype, device=device)

    def forward(self, x):
        x = self.dense(x)
        return torch.nn.functional.gelu(x)


class BertBlock(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_dim, heads, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.attention = BertAttentionBlock(embed_dim, heads, layer_norm_eps, dtype, device, operations)
        self.intermediate = BertIntermediate(embed_dim, intermediate_dim, dtype, device, operations)
        self.output = BertOutput(intermediate_dim, embed_dim, layer_norm_eps, dtype, device, operations)

    def forward(self, x, mask, optimized_attention):
        x = self.attention(x, mask, optimized_attention)
        y = self.intermediate(x)
        return self.output(y, x)

class BertEncoder(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, intermediate_dim, heads, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.layer = torch.nn.ModuleList([BertBlock(embed_dim, intermediate_dim, heads, layer_norm_eps, dtype, device, operations) for i in range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layer) + intermediate_output

        intermediate = None
        for i, l in enumerate(self.layer):
            x = l(x, mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate

class BertEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, max_position_embeddings, type_vocab_size, pad_token_id, embed_dim, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.word_embeddings = operations.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id, dtype=dtype, device=device)
        self.position_embeddings = operations.Embedding(max_position_embeddings, embed_dim, dtype=dtype, device=device)
        self.token_type_embeddings = operations.Embedding(type_vocab_size, embed_dim, dtype=dtype, device=device)

        self.LayerNorm = operations.LayerNorm(embed_dim, eps=layer_norm_eps, dtype=dtype, device=device)

    def forward(self, input_tokens, token_type_ids=None, dtype=None):
        x = self.word_embeddings(input_tokens, out_dtype=dtype)
        x += comfy.ops.cast_to_input(self.position_embeddings.weight[:x.shape[1]], x)
        if token_type_ids is not None:
            x += self.token_type_embeddings(token_type_ids, out_dtype=x.dtype)
        else:
            x += comfy.ops.cast_to_input(self.token_type_embeddings.weight[0], x)
        x = self.LayerNorm(x)
        return x


class BertModel_(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        embed_dim = config_dict["hidden_size"]
        layer_norm_eps = config_dict["layer_norm_eps"]

        self.embeddings = BertEmbeddings(config_dict["vocab_size"], config_dict["max_position_embeddings"], config_dict["type_vocab_size"], config_dict["pad_token_id"], embed_dim, layer_norm_eps, dtype, device, operations)
        self.encoder = BertEncoder(config_dict["num_hidden_layers"], embed_dim, config_dict["intermediate_size"], config_dict["num_attention_heads"], layer_norm_eps, dtype, device, operations)

    def forward(self, input_tokens, attention_mask=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None):
        x = self.embeddings(input_tokens, dtype=dtype)
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), float("-inf"))

        x, i = self.encoder(x, mask, intermediate_output)
        return x, i


class BertModel(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.bert = BertModel_(config_dict, dtype, device, operations)
        self.num_layers = config_dict["num_hidden_layers"]

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, embeddings):
        self.bert.embeddings.word_embeddings = embeddings

    def forward(self, *args, **kwargs):
        return self.bert(*args, **kwargs)

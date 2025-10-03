from typing import Optional, Callable

import torch
from torch import nn

from comfy.text_encoders.bert import (
    BertIntermediate as ClapTextIntermediate,
    BertOutput as ClapTextOutput,
)
from comfy.ldm.modules.attention import optimized_attention

import os
from comfy import sd1_clip
from transformers import AutoTokenizer

def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor],
    chunk_size: int,
    chunk_dim: int,
    *input_tensors,
) -> torch.Tensor:

    if chunk_size > 0:
        # chunk into tuples and apply forward_fn to each element in the tuple
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)

        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)

# from modeling_roberta
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

class ClapProjectionLayer(nn.Module):
    def __init__(self, hidden_size, projection_dim, device, dtype, operations):
        super().__init__()
        self.linear1 = operations.Linear(hidden_size, projection_dim, device=device, dtype=dtype)
        self.activation = torch.nn.ReLU()
        self.linear2 = operations.Linear(projection_dim, projection_dim, device=device, dtype=dtype)

    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


# same as RobertaEmbeddings
class ClapTextEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id, max_position_embeddings, type_vocab_size, layer_norm_eps, device, dtype, operations):
        super().__init__()
        self.word_embeddings = operations.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id, device=device, dtype=dtype)
        self.position_embeddings = operations.Embedding(max_position_embeddings, hidden_size, device=device, dtype=dtype)
        self.token_type_embeddings = operations.Embedding(type_vocab_size, hidden_size, device=device, dtype=dtype)

        self.LayerNorm = operations.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings, device=device, dtype=dtype).expand((1, -1)), persistent=True
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long, device=device), persistent=True
        )

        self.padding_idx = pad_token_id
        self.position_embeddings = operations.Embedding(
            max_position_embeddings, hidden_size, padding_idx=self.padding_idx, device=device, dtype=dtype
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):

        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

# same as AlignTextSelfAttention
class ClapTextSelfAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, device, dtype, operations):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = operations.Linear(hidden_size, self.all_head_size, device=device, dtype=dtype)
        self.key = operations.Linear(hidden_size, self.all_head_size, device=device, dtype=dtype)
        self.value = operations.Linear(hidden_size, self.all_head_size, device=device, dtype=dtype)

        self.scaling = self.attention_head_size**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states, value_states = [t.contiguous() for t in (query_states, key_states, value_states)]
        attention_mask = attention_mask.to(query_states.dtype)
        attn_output = optimized_attention(query_states, key_states, value_states, self.num_attention_heads, mask = attention_mask, skip_output_reshape=True, skip_reshape=True)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output.reshape(*input_shape, -1).contiguous()

class ClapTextSelfOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps, device, dtype, operations):
        super().__init__()
        self.dense = operations.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.LayerNorm = operations.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# same as AlignTextAttention
class ClapTextAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, layer_norm_eps, device, dtype, operations):
        super().__init__()
        self.self = ClapTextSelfAttention(num_attention_heads, hidden_size, device=device, dtype=dtype, operations = operations)
        self.output = ClapTextSelfOutput(hidden_size, layer_norm_eps, device=device, dtype=dtype, operations=operations)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        return self.output(self_outputs, hidden_states)

# same as AlignTextLayer
class ClapTextLayer(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, intermediate_size, layer_norm_eps, device, dtype, operations):
        super().__init__()
        # maybe we could allow chunking dynamically
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.attention = ClapTextAttention(num_attention_heads, hidden_size, layer_norm_eps, device=device, dtype=dtype, operations=operations)
        self.intermediate = ClapTextIntermediate(hidden_size, intermediate_size, device=device, dtype=dtype, operations=operations)
        self.output = ClapTextOutput(intermediate_size, hidden_size, layer_norm_eps, device=device, dtype=dtype, operations=operations)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        attention_output = self_attention_outputs

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# same as AlignTextEncoder
class ClapTextEncoder(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, intermediate_size, layer_norm_eps, num_hidden_layers, device, dtype, operations):
        super().__init__()
        self.layer = nn.ModuleList([ClapTextLayer(num_attention_heads, hidden_size, intermediate_size, layer_norm_eps, device=device, dtype=dtype, operations=operations)
                                    for _ in range(num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):

        for _, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )
        return hidden_states

class ClapTextPooler(nn.Module):
    def __init__(self, hidden_size, device, dtype, operations):
        super().__init__()
        self.dense = operations.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ClapTextModel(nn.Module):

    def __init__(self, num_attention_heads, vocab_size, hidden_size, intermediate_size, pad_token_id, max_position_embeddings, type_vocab_size, layer_norm_eps, num_hidden_layers,
                 device, dtype, operations, add_pooling_layer=True):
        super().__init__()

        self.embeddings = ClapTextEmbeddings(vocab_size, hidden_size, pad_token_id, max_position_embeddings, type_vocab_size, layer_norm_eps,
                                             device=device, dtype=dtype, operations=operations,)
        self.encoder = ClapTextEncoder(num_attention_heads, hidden_size, intermediate_size, layer_norm_eps, num_hidden_layers, device=device, dtype=dtype, operations=operations,)
        self.pooler = ClapTextPooler(hidden_size, device=device, dtype=dtype, operations=operations,) if add_pooling_layer else None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
    ):

        if input_ids is not None:
            input_shape = input_ids.size()
        elif embeds is not None:
            input_shape = embeds.size()[:-1]

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else embeds.device

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
        )
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return sequence_output, pooled_output

class ClapTextModelWithProjection(nn.Module):
    def __init__(
        self,
        config,
        dtype=None,
        device=None,
        operations=None,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        layer_norm_eps: float = 1e-12,
        max_position_embeddings: int = 514,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        projection_dim: int = 512,
        type_vocab_size: int = 1,
        vocab_size: int = 50265,
        pad_token_id: int = 1,
    ):
        super().__init__()
        self.num_layers = num_hidden_layers
        self.text_model = ClapTextModel(num_attention_heads, vocab_size, hidden_size, intermediate_size, pad_token_id, max_position_embeddings,
                                        type_vocab_size, layer_norm_eps, num_hidden_layers, device=device, dtype=dtype, operations=operations)
        self.text_projection = ClapProjectionLayer(hidden_size, projection_dim, device=device, dtype=dtype, operations=operations,)

    def get_input_embeddings(self):
        return self.text_model.embeddings.word_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        embeds = None,
        **kwargs
    ):

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            embeds=embeds
        )

        pooled_output = text_outputs[1]
        text_embeds = self.text_projection(pooled_output)

        return text_outputs[0], torch.tensor([]), text_embeds

class ClapTextEncoderModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        self.dtypes = set([dtype])
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 1}, layer_norm_hidden_state=False, model_class=ClapTextModelWithProjection, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

class ClapLargeTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clap_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=2048, embedding_key='clap_l', tokenizer_class=AutoTokenizer, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=1, tokenizer_data=tokenizer_data)

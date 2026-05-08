# Jina CLIP v2 and Jina Embeddings v3 both use their modified XLM-RoBERTa architecture. Reference implementation:
# Jina CLIP v2 (both text and vision): https://huggingface.co/jinaai/jina-clip-implementation/blob/39e6a55ae971b59bea6e44675d237c99762e7ee2/modeling_clip.py
# Jina XLM-RoBERTa (text only): http://huggingface.co/jinaai/xlm-roberta-flash-implementation/blob/2b6bc3f30750b3a9648fe9b63448c09920efe9be/modeling_xlm_roberta.py

from dataclasses import dataclass

import torch
from torch import nn as nn
from torch.nn import functional as F

import comfy.model_management
import comfy.ops
from comfy import sd1_clip
from .spiece_tokenizer import SPieceTokenizer

class JinaClip2Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = tokenizer_data.get("spiece_model", None)
        # The official NewBie uses max_length=8000, but Jina Embeddings v3 actually supports 8192
        super().__init__(tokenizer, pad_with_end=False, embedding_size=1024, embedding_key='jina_clip_2', tokenizer_class=SPieceTokenizer, has_start_token=True, has_end_token=True, pad_to_max_length=False, max_length=8192, min_length=1, pad_token=1, end_token=2, tokenizer_args={"add_bos": True, "add_eos": True}, tokenizer_data=tokenizer_data)

    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}

class JinaClip2TokenizerWrapper(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, tokenizer=JinaClip2Tokenizer, name="jina_clip_2")

# https://huggingface.co/jinaai/jina-embeddings-v3/blob/343dbf534c76fe845f304fa5c2d1fd87e1e78918/config.json
@dataclass
class XLMRobertaConfig:
    vocab_size: int = 250002
    type_vocab_size: int = 1
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    rotary_emb_base: float = 20000.0
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-05
    bos_token_id: int = 0
    eos_token_id: int = 2
    pad_token_id: int = 1

class XLMRobertaEmbeddings(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        embed_dim = config.hidden_size
        self.word_embeddings = ops.Embedding(config.vocab_size, embed_dim, padding_idx=config.pad_token_id, device=device, dtype=dtype)
        self.token_type_embeddings = ops.Embedding(config.type_vocab_size, embed_dim, device=device, dtype=dtype)

    def forward(self, input_ids=None, embeddings=None):
        if input_ids is not None and embeddings is None:
            embeddings = self.word_embeddings(input_ids)

        if embeddings is not None:
            token_type_ids = torch.zeros(embeddings.shape[1], device=embeddings.device, dtype=torch.int32)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
        return embeddings

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if seqlen > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device or self._cos_cached.dtype != dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(self, q, k):
        batch, seqlen, heads, head_dim = q.shape
        self._update_cos_sin_cache(seqlen, device=q.device, dtype=q.dtype)

        cos = self._cos_cached[:seqlen].view(1, seqlen, 1, head_dim)
        sin = self._sin_cached[:seqlen].view(1, seqlen, 1, head_dim)

        def rotate_half(x):
            size = x.shape[-1] // 2
            x1, x2 = x[..., :size], x[..., size:]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

class MHA(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = embed_dim // config.num_attention_heads

        self.rotary_emb = RotaryEmbedding(self.head_dim, config.rotary_emb_base, device=device)
        self.Wqkv = ops.Linear(embed_dim, 3 * embed_dim, device=device, dtype=dtype)
        self.out_proj = ops.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

    def forward(self, x, mask=None, optimized_attention=None):
        qkv = self.Wqkv(x)
        batch_size, seq_len, _ = qkv.shape
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q, k = self.rotary_emb(q, k)

        # NHD -> HND
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = optimized_attention(q, k, v, heads=self.num_heads, mask=mask, skip_reshape=True)
        return self.out_proj(out)

class MLP(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.fc1 = ops.Linear(config.hidden_size, config.intermediate_size, device=device, dtype=dtype)
        self.activation = F.gelu
        self.fc2 = ops.Linear(config.intermediate_size, config.hidden_size, device=device, dtype=dtype)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.mixer = MHA(config, device=device, dtype=dtype, ops=ops)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.norm1 = ops.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, device=device, dtype=dtype)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.norm2 = ops.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, device=device, dtype=dtype)

    def forward(self, hidden_states, mask=None, optimized_attention=None):
        mixer_out = self.mixer(hidden_states, mask=mask, optimized_attention=optimized_attention)
        hidden_states = self.norm1(self.dropout1(mixer_out) + hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = self.norm2(self.dropout2(mlp_out) + hidden_states)
        return hidden_states

class XLMRobertaEncoder(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.layers = nn.ModuleList([Block(config, device=device, dtype=dtype, ops=ops) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        optimized_attention = comfy.ldm.modules.attention.optimized_attention_for_device(hidden_states.device, mask=attention_mask is not None, small_input=True)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=attention_mask, optimized_attention=optimized_attention)
        return hidden_states

class XLMRobertaModel_(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.embeddings = XLMRobertaEmbeddings(config, device=device, dtype=dtype, ops=ops)
        self.emb_ln = ops.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, device=device, dtype=dtype)
        self.emb_drop = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = XLMRobertaEncoder(config, device=device, dtype=dtype, ops=ops)

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[]):
        x = self.embeddings(input_ids=input_ids, embeddings=embeds)
        x = self.emb_ln(x)
        x = self.emb_drop(x)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, 1, attention_mask.shape[-1]))
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(x.dtype).max)

        sequence_output = self.encoder(x, attention_mask=mask)

        # Mean pool, see https://huggingface.co/jinaai/jina-clip-implementation/blob/39e6a55ae971b59bea6e44675d237c99762e7ee2/hf_model.py
        pooled_output = None
        if attention_mask is None:
            pooled_output = sequence_output.mean(dim=1)
        else:
            attention_mask = attention_mask.to(sequence_output.dtype)
            pooled_output = (sequence_output * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)

        # Intermediate output is not yet implemented, use None for placeholder
        return sequence_output, None, pooled_output

class XLMRobertaModel(nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.config = XLMRobertaConfig(**config_dict)
        self.model = XLMRobertaModel_(self.config, device=device, dtype=dtype, ops=operations)
        self.num_layers = self.config.num_hidden_layers

    def get_input_embeddings(self):
        return self.model.embeddings.word_embeddings

    def set_input_embeddings(self, embeddings):
        self.model.embeddings.word_embeddings = embeddings

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class JinaClip2TextModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, textmodel_json_config={}, model_class=XLMRobertaModel, special_tokens={"start": 0, "end": 2, "pad": 1}, enable_attention_masks=True, return_attention_masks=True, model_options=model_options)

class JinaClip2TextModelWrapper(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, clip_model=JinaClip2TextModel, name="jina_clip_2", model_options=model_options)

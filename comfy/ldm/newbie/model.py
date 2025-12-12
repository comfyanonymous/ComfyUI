# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from typing import List, Optional, Tuple

from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import RMSNorm


def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


#############################################################################
#             Embedding Layers for Timesteps and Class Labels               #
#############################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                hidden_size,
                bias=True,
            ),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


#############################################################################
#                               Core NextDiT Model                              #
#############################################################################


class JointAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        qk_norm: bool,
    ):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(
            dim,
            (n_heads + self.n_kv_heads + self.n_kv_heads) * self.head_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.qkv.weight)

        self.out = nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.out.weight)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    @staticmethod
    def apply_rotary_emb(
        x_in: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensors using the given frequency
        tensor.

        This function applies rotary embeddings to the given query 'xq' and
        key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
        input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors
        contain rotary embeddings and are returned as real tensors.

        Args:
            x_in (torch.Tensor): Query or Key tensor to apply rotary embeddings.
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
                exponentials.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor
                and key tensor with rotary embeddings.
        """
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)

    # copied from huggingface modeling_llama.py
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            return (
                indices,
                cu_seqlens,
                max_seqlen_in_batch,
            )

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.n_local_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            x:
            x_mask:
            freqs_cis:

        Returns:

        """
        bsz, seqlen, _ = x.shape
        dtype = x.dtype

        xq, xk, xv = torch.split(
            self.qkv(x),
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xq = JointAttention.apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = JointAttention.apply_rotary_emb(xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        softmax_scale = math.sqrt(1 / self.head_dim)

        if dtype in [torch.float16, torch.bfloat16]:
            # begin var_len flash attn
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(xq, xk, xv, x_mask, seqlen)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=0.0,
                causal=False,
                softmax_scale=softmax_scale,
            )
            output = pad_input(attn_output_unpad, indices_q, bsz, seqlen)
            # end var_len_flash_attn

        else:
            n_rep = self.n_local_heads // self.n_local_kv_heads
            if n_rep >= 1:
                xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            output = (
                F.scaled_dot_product_attention(
                    xq.permute(0, 2, 1, 3),
                    xk.permute(0, 2, 1, 3),
                    xv.permute(0, 2, 1, 3),
                    attn_mask=x_mask.bool().view(bsz, 1, 1, seqlen).expand(-1, self.n_local_heads, seqlen, -1),
                    scale=softmax_scale,
                )
                .permute(0, 2, 1, 3)
                .to(dtype)
            )

        output = output.flatten(-2)

        return self.out(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension. Defaults to None.

        """
        super().__init__()
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.w1.weight)
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.w2.weight)
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.w3.weight)

    # @torch.compile
    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class JointTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        modulation=True
    ) -> None:
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            dim (int): Embedding dimension of the input features.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int):
            ffn_dim_multiplier (float):
            norm_eps (float):

        """
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    min(dim, 1024),
                    4 * dim,
                    bias=True,
                ),
            )
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor]=None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(
                    modulate(self.attention_norm1(x), scale_msa),
                    x_mask,
                    freqs_cis,
                )
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(
                    modulate(self.ffn_norm1(x), scale_mlp),
                )
            )
        else:
            assert adaln_input is None
            x = x + self.attention_norm2(
                self.attention(
                    self.attention_norm1(x),
                    x_mask,
                    freqs_cis,
                )
            )
            x = x + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )
            )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of NextDiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(hidden_size, 1024),
                hidden_size,
                bias=True,
            ),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self, theta: float = 10000.0, axes_dims: List[int] = (16, 56, 56), axes_lens: List[int] = (1, 512, 512)
    ):
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis = NextDiT.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)

    def __call__(self, ids: torch.Tensor):
        self.freqs_cis = [freqs_cis.to(ids.device) for freqs_cis in self.freqs_cis]
        result = []
        for i in range(len(self.axes_dims)):
            # import torch.distributed as dist
            # if not dist.is_initialized() or dist.get_rank() == 0:
            #     import pdb
            #     pdb.set_trace()
            index = ids[:, :, i:i+1].repeat(1, 1, self.freqs_cis[i].shape[-1]).to(torch.int64)
            result.append(torch.gather(self.freqs_cis[i].unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
        return torch.cat(result, dim=-1)


class NextDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_refiner_layers: int = 2,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (1, 512, 512),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
        )
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0.0)

        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(
                cap_feat_dim,
                dim,
                bias=True,
            ),
        )
        nn.init.trunc_normal_(self.cap_embedder[1].weight, std=0.02)
        # nn.init.zeros_(self.cap_embedder[1].weight)
        nn.init.zeros_(self.cap_embedder[1].bias)

        self.layers = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.norm_final = RMSNorm(dim, eps=norm_eps)
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = RopeEmbedder(axes_dims=axes_dims, axes_lens=axes_lens)
        self.dim = dim
        self.n_heads = n_heads

    def unpatchify(
        self, x: torch.Tensor, img_size: List[Tuple[int, int]], cap_size: List[int], return_tensor=False
    ) -> List[torch.Tensor]:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        pH = pW = self.patch_size
        imgs = []
        for i in range(x.size(0)):
            H, W = img_size[i]
            begin = cap_size[i]
            end = begin + (H // pH) * (W // pW)
            imgs.append(
                x[i][begin:end]
                .view(H // pH, W // pW, pH, pW, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )

        if return_tensor:
            imgs = torch.stack(imgs, dim=0)
        return imgs

    def patchify_and_embed(
        self, x: List[torch.Tensor] | torch.Tensor, cap_feats: torch.Tensor, cap_mask: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], List[int], torch.Tensor]:
        bsz = len(x)
        pH = pW = self.patch_size
        device = x[0].device

        l_effective_cap_len = cap_mask.sum(dim=1).tolist()
        img_sizes = [(img.size(1), img.size(2)) for img in x]
        l_effective_img_len = [(H // pH) * (W // pW) for (H, W) in img_sizes]

        max_seq_len = max(
            (cap_len+img_len for cap_len, img_len in zip(l_effective_cap_len, l_effective_img_len))
        )
        max_cap_len = max(l_effective_cap_len)
        max_img_len = max(l_effective_img_len)

        position_ids = torch.zeros(bsz, max_seq_len, 3, dtype=torch.int32, device=device)

        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            H, W = img_sizes[i]
            H_tokens, W_tokens = H // pH, W // pW
            assert H_tokens * W_tokens == img_len

            position_ids[i, :cap_len, 0] = torch.arange(cap_len, dtype=torch.int32, device=device)
            position_ids[i, cap_len:cap_len+img_len, 0] = cap_len
            row_ids = torch.arange(H_tokens, dtype=torch.int32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
            col_ids = torch.arange(W_tokens, dtype=torch.int32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
            position_ids[i, cap_len:cap_len+img_len, 1] = row_ids
            position_ids[i, cap_len:cap_len+img_len, 2] = col_ids

        freqs_cis = self.rope_embedder(position_ids)

        # build freqs_cis for cap and image individually
        cap_freqs_cis_shape = list(freqs_cis.shape)
        # cap_freqs_cis_shape[1] = max_cap_len
        cap_freqs_cis_shape[1] = cap_feats.shape[1]
        cap_freqs_cis = torch.zeros(*cap_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        img_freqs_cis_shape = list(freqs_cis.shape)
        img_freqs_cis_shape[1] = max_img_len
        img_freqs_cis = torch.zeros(*img_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            cap_freqs_cis[i, :cap_len] = freqs_cis[i, :cap_len]
            img_freqs_cis[i, :img_len] = freqs_cis[i, cap_len:cap_len+img_len]

        # refine context
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis)

        # refine image
        flat_x = []
        for i in range(bsz):
            img = x[i]
            C, H, W = img.size()
            img = img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 2, 4, 0).flatten(2).flatten(0, 1)
            flat_x.append(img)
        x = flat_x
        padded_img_embed = torch.zeros(bsz, max_img_len, x[0].shape[-1], device=device, dtype=x[0].dtype)
        padded_img_mask = torch.zeros(bsz, max_img_len, dtype=torch.bool, device=device)
        for i in range(bsz):
            padded_img_embed[i, :l_effective_img_len[i]] = x[i]
            padded_img_mask[i, :l_effective_img_len[i]] = True

        padded_img_embed = self.x_embedder(padded_img_embed)
        for layer in self.noise_refiner:
            padded_img_embed = layer(padded_img_embed, padded_img_mask, img_freqs_cis, t)

        mask = torch.zeros(bsz, max_seq_len, dtype=torch.bool, device=device)
        padded_full_embed = torch.zeros(bsz, max_seq_len, self.dim, device=device, dtype=x[0].dtype)
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]

            mask[i, :cap_len+img_len] = True
            padded_full_embed[i, :cap_len] = cap_feats[i, :cap_len]
            padded_full_embed[i, cap_len:cap_len+img_len] = padded_img_embed[i, :img_len]

        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis


    def forward(self, x, t, cap_feats, cap_mask):
        """
        Forward pass of NextDiT.
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of text tokens/features
        """

        # import torch.distributed as dist
        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     import pdb
        #     pdb.set_trace()
            # torch.save([x, t, cap_feats, cap_mask], "./fake_input.pt")
        t = self.t_embedder(t)  # (N, D)
        adaln_input = t

        cap_feats = self.cap_embedder(cap_feats)  # (N, L, D)  # todo check if able to batchify w.o. redundant compute

        x_is_tensor = isinstance(x, torch.Tensor)
        x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(x, cap_feats, cap_mask, t)
        freqs_cis = freqs_cis.to(x.device)

        for layer in self.layers:
            x = layer(x, mask, freqs_cis, adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, img_size, cap_size, return_tensor=x_is_tensor)

        return x

    def forward_with_cfg(
        self,
        x,
        t,
        cap_feats,
        cap_mask,
        cfg_scale,
        cfg_trunc=100,
        renorm_cfg=1
    ):
        """
        Forward pass of NextDiT, but also batches the unconditional forward pass
        for classifier-free guidance.
        """
        # # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        if t[0] < cfg_trunc:
            combined = torch.cat([half, half], dim=0) # [2, 16, 128, 128]
            model_out = self.forward(combined, t, cap_feats, cap_mask) # [2, 16, 128, 128]
            # For exact reproducibility reasons, we apply classifier-free guidance on only
            # three channels by default. The standard approach to cfg applies it to all channels.
            # This can be done by uncommenting the following line and commenting-out the line following that.
            eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)  
            if float(renorm_cfg) > 0.0: 
                ori_pos_norm = torch.linalg.vector_norm(cond_eps
                        , dim=tuple(range(1, len(cond_eps.shape))), keepdim=True
                )
                max_new_norm = ori_pos_norm * float(renorm_cfg)
                new_pos_norm = torch.linalg.vector_norm(
                        half_eps, dim=tuple(range(1, len(half_eps.shape))), keepdim=True
                    )
                if new_pos_norm >= max_new_norm:
                    half_eps = half_eps * (max_new_norm / new_pos_norm)
        else:
            combined = half
            model_out = self.forward(combined, t[:len(x) // 2], cap_feats[:len(x) // 2], cap_mask[:len(x) // 2])
            eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
            half_eps = eps

        output = torch.cat([half_eps, half_eps], dim=0)
        return output

    @staticmethod
    def precompute_freqs_cis(
        dim: List[int],
        end: List[int],
        theta: float = 10000.0,
    ):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (list): Dimension of the frequency tensor.
            end (list): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """
        freqs_cis = []
        for i, (d, e) in enumerate(zip(dim, end)):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
            timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64
            freqs_cis.append(freqs_cis_i)

        return freqs_cis

    def parameter_count(self) -> int:
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            for param in module.parameters(recurse=False):
                total_params += param.numel()
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

    def get_checkpointing_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32


class NextDiT_CLIP(NextDiT):
    def __init__(self, *args, **kwargs):

        clip_text_dim = kwargs.pop('clip_text_dim', 1024)
        clip_img_dim = kwargs.pop('clip_img_dim', 1024)

        super().__init__(*args, **kwargs)

        self.enable_clip = True

        self.time_text_embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(self.dim, 1024) + clip_text_dim, min(self.dim, 1024)),
        )
        nn.init.zeros_(self.time_text_embed[1].weight)
        nn.init.zeros_(self.time_text_embed[1].bias)

        self.clip_text_pooled_proj = nn.Sequential(
            RMSNorm(clip_text_dim),
            nn.Linear(clip_text_dim, clip_text_dim, bias=True),
        )
        nn.init.normal_(self.clip_text_pooled_proj[1].weight, std=0.01) 
        nn.init.zeros_(self.clip_text_pooled_proj[1].bias)


    def forward(self, x, t, cap_feats, cap_mask, **kwargs):

        clip_text_pooled = kwargs.get('clip_text_pooled')
        clip_img_pooled = kwargs.get('clip_img_pooled')

        t_emb = self.t_embedder(t)
        adaln_input = t_emb
        cap_feats = self.cap_embedder(cap_feats)
        if clip_text_pooled is not None:
            clip_emb = self.clip_text_pooled_proj(clip_text_pooled)
            combined_features = torch.cat([t_emb, clip_emb], dim=-1)
            adaln_input = self.time_text_embed(combined_features)
        else:
            adaln_input = t_emb

        if clip_img_pooled is not None:
            clip_img_pooled_emb = self.clip_img_pooled_embedder(clip_img_pooled)
            adaln_input = adaln_input + clip_img_pooled_emb

        x_is_tensor = isinstance(x, torch.Tensor)
        x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(x, cap_feats, cap_mask, adaln_input)
        freqs_cis = freqs_cis.to(x.device)

        for layer in self.layers:
            x = layer(x, mask, freqs_cis, adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, img_size, cap_size, return_tensor=x_is_tensor)

        return x

    def forward_with_cfg(
            self,
            x,
            t,
            cap_feats,
            cap_mask,
            cfg_scale,
            cfg_trunc=100,
            renorm_cfg=1,
            **kwargs
    ):

        half = x[: len(x) // 2]

        if t[0] < cfg_trunc:

            model_out = self.forward(x, t, cap_feats, cap_mask, **kwargs)

            eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

            if float(renorm_cfg) > 0.0:
                ori_pos_norm = torch.linalg.vector_norm(
                    cond_eps, dim=tuple(range(1, len(cond_eps.shape))), keepdim=True
                )
                max_new_norm = ori_pos_norm * float(renorm_cfg)
                new_pos_norm = torch.linalg.vector_norm(
                    half_eps, dim=tuple(range(1, len(half_eps.shape))), keepdim=True
                )
                if new_pos_norm.item() >= max_new_norm.item():
                    half_eps = half_eps * (max_new_norm / new_pos_norm)
        else:

            cond_x = half
            cond_t = t[:len(x) // 2]
            cond_cap_feats = cap_feats[:len(x) // 2]
            cond_cap_mask = cap_mask[:len(x) // 2]

            cond_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    cond_kwargs[k] = v[:len(x) // 2]

            model_out = self.forward(
                cond_x, cond_t, cond_cap_feats, cond_cap_mask, **cond_kwargs
            )
            eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
            half_eps = eps

        output = torch.cat([half_eps, half_eps], dim=0)
        return output


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        hidden_states = x
        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + hidden_states


class CNNEncoder(nn.Module):

    def __init__(self, in_channels=16, base_channels=128, out_channels=64):
        super().__init__()

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        self.resnet_blocks = nn.ModuleList([
            ResnetBlock(base_channels, base_channels),
            ResnetBlock(base_channels, base_channels * 2),
            ResnetBlock(base_channels * 2, base_channels * 2),
        ])

        # Final projection to match patch dimension
        self.conv_out = nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)

        # Initialize conv_out to zero
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        # x: [B, 16, H, W] (latent space)
        h = self.conv_in(x)

        for resnet in self.resnet_blocks:
            h = resnet(h)

        # Project to patch dimension
        h = self.conv_out(h)  # [B, 64, H, W]

        return h


class NextDiT_CLIP_CNN(NextDiT_CLIP):
    def __init__(self, *args, **kwargs):

        cnn_base_channels = kwargs.pop('cnn_base_channels', 128)

        super().__init__(*args, **kwargs)

        #patch_dim = self.patch_size * self.patch_size * self.in_channels  # 2*2*16=64
        self.cnn_encoder = CNNEncoder(
            in_channels=self.in_channels,  # 16
            base_channels=cnn_base_channels,
            out_channels=cnn_base_channels*2  # 64
        )

        self.cnn_proj = nn.Linear(self.patch_size * self.patch_size*cnn_base_channels * 2, self.dim, bias=True)
        nn.init.zeros_(self.cnn_proj.weight)
        nn.init.zeros_(self.cnn_proj.bias)

    def patchify_and_embed(self, x, cap_feats, cap_mask, adaln_input):

        bsz = len(x)
        pH = pW = self.patch_size
        device = x[0].device

        original_img_sizes = [(img.size(1), img.size(2)) for img in x]
        l_effective_cap_len = cap_mask.sum(dim=1).tolist()
        l_effective_img_len = [(H // pH) * (W // pW) for (H, W) in original_img_sizes]
        max_img_len = max(l_effective_img_len)

        list_of_fused_embeds = []
        for i in range(bsz):
            img_tensor = x[i]
            C_in, H, W = img_tensor.shape

            # [C, H, W] -> [1, N, P*P*C]
            raw_patches = img_tensor.view(C_in, H // pH, pH, W // pW, pW).permute(1, 3, 0, 2, 4).flatten(2)
            raw_patches = raw_patches.flatten(0, 1).unsqueeze(0)

            # [C, H, W] -> [1, C, H, W] -> [1, C_cnn, H, W]
            cnn_feature_map = self.cnn_encoder(img_tensor.unsqueeze(0))
            C_cnn, H_fm, W_fm = cnn_feature_map.shape[1:]

            cnn_patches = cnn_feature_map.view(1, C_cnn, H_fm // pH, pH, W_fm // pW, pW).permute(0, 2, 4, 1, 3,
                                                                                                 5).flatten(3)
            cnn_patches = cnn_patches.flatten(1, 2)

            raw_embed = self.x_embedder(raw_patches)  # [1, N, D]
            cnn_embed = self.cnn_proj(cnn_patches)  # [1, N, D]

            fused_embed = raw_embed + cnn_embed  # [1, N, D]

            list_of_fused_embeds.append(fused_embed.squeeze(0))

        padded_img_embed = torch.zeros(bsz, max_img_len, self.dim, device=device, dtype=list_of_fused_embeds[0].dtype)
        padded_img_mask = torch.zeros(bsz, max_img_len, dtype=torch.bool, device=device)
        for i in range(bsz):
            img_len = l_effective_img_len[i]
            padded_img_embed[i, :img_len] = list_of_fused_embeds[i]
            padded_img_mask[i, :img_len] = True

        max_seq_len = max((cap_len + img_len for cap_len, img_len in zip(l_effective_cap_len, l_effective_img_len)))
        padded_full_embed = torch.zeros(bsz, max_seq_len, self.dim, device=device, dtype=padded_img_embed.dtype)
        mask = torch.zeros(bsz, max_seq_len, dtype=torch.bool, device=device)

        position_ids = torch.zeros(bsz, max_seq_len, 3, dtype=torch.int32, device=device)
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            H, W = original_img_sizes[i]
            H_tokens, W_tokens = H // pH, W // pW
            position_ids[i, :cap_len, 0] = torch.arange(cap_len, dtype=torch.int32, device=device)
            position_ids[i, cap_len:cap_len + img_len, 0] = cap_len
            row_ids = torch.arange(H_tokens, dtype=torch.int32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
            col_ids = torch.arange(W_tokens, dtype=torch.int32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
            position_ids[i, cap_len:cap_len + img_len, 1] = row_ids
            position_ids[i, cap_len:cap_len + img_len, 2] = col_ids

        freqs_cis = self.rope_embedder(position_ids)
        cap_freqs_cis_shape = list(freqs_cis.shape);
        cap_freqs_cis_shape[1] = cap_feats.shape[1]
        cap_freqs_cis = torch.zeros(*cap_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)
        img_freqs_cis_shape = list(freqs_cis.shape);
        img_freqs_cis_shape[1] = max_img_len
        img_freqs_cis = torch.zeros(*img_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        for i in range(bsz):
            cap_len = l_effective_cap_len[i];
            img_len = l_effective_img_len[i]
            cap_freqs_cis[i, :cap_len] = freqs_cis[i, :cap_len]
            img_freqs_cis[i, :img_len] = freqs_cis[i, cap_len:cap_len + img_len]

        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis)

        for layer in self.noise_refiner:
            padded_img_embed = layer(padded_img_embed, padded_img_mask, img_freqs_cis, adaln_input)

        for i in range(bsz):
            cap_len = l_effective_cap_len[i];
            img_len = l_effective_img_len[i]
            mask[i, :cap_len + img_len] = True
            padded_full_embed[i, :cap_len] = cap_feats[i, :cap_len]
            padded_full_embed[i, cap_len:cap_len + img_len] = padded_img_embed[i, :img_len]

        return padded_full_embed, mask, original_img_sizes, l_effective_cap_len, freqs_cis
#############################################################################
#                                 NextDiT Configs                           #
#############################################################################

def NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP(**kwargs):
    for k in ("image_model", "model_type", "rope_theta", "dtype"):
        kwargs.pop(k, None)
        
    kwargs.setdefault("patch_size", 2)
    kwargs.setdefault("in_channels", 16)
    kwargs.setdefault("dim", 2304)
    kwargs.setdefault("n_layers", 36)
    kwargs.setdefault("n_heads", 24)
    kwargs.setdefault("n_kv_heads", 8)
    kwargs.setdefault("axes_dims", [32, 32, 32])
    kwargs.setdefault("axes_lens", [1024, 512, 512])
    return NextDiT_CLIP(**kwargs)

def NewBieNextDiT(*, device=None, dtype=None, operations=None, **kwargs):
    import torch
    if dtype is None:
        target = torch.bfloat16
        if not torch.cuda.is_available():
            target = torch.float32
        else:
            dev  = torch.cuda.current_device()
            major, minor = torch.cuda.get_device_capability(dev)
            if major < 8:
                target = torch.float16
        dtype = target
    model = NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP(**kwargs)
    model.to(device=device, dtype=dtype)
    _ = operations
    return model
# Original from: https://github.com/ace-step/ACE-Step/blob/main/models/lyrics_utils/lyric_encoder.py
from typing import Optional, Tuple, Union
import math
import torch
from torch import nn

import comfy.model_management

class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True,
                 dtype=None, device=None, operations=None):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()

        self.pointwise_conv1 = operations.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            dtype=dtype, device=device
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = operations.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
            dtype=dtype, device=device
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = operations.LayerNorm(channels, dtype=dtype, device=device)

        self.pointwise_conv2 = operations.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            dtype=dtype, device=device
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:  # cache_t == 0
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)  # equal batch
                assert cache.size(1) == x.size(1)  # equal channel
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            new_cache = x[:, :, -self.lorder:]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache

class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
            self,
            idim: int,
            hidden_units: int,
            dropout_rate: float,
            activation: torch.nn.Module = torch.nn.ReLU(),
            dtype=None, device=None, operations=None
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = operations.Linear(idim, hidden_units, dtype=dtype, device=device)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = operations.Linear(hidden_units, idim, dtype=dtype, device=device)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))

class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True,
                 dtype=None, device=None, operations=None):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = operations.Linear(n_feat, n_feat, dtype=dtype, device=device)
        self.linear_k = operations.Linear(n_feat, n_feat, bias=key_bias, dtype=dtype, device=device)
        self.linear_v = operations.Linear(n_feat, n_feat, dtype=dtype, device=device)
        self.linear_out = operations.Linear(n_feat, n_feat, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)

        if mask is not None and mask.size(2) > 0:  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)

        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h * self.d_k)
             )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                CosyVoice.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q, k, v = self.forward_qkv(query, key, value)
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True,
                 dtype=None, device=None, operations=None):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, key_bias, dtype=dtype, device=device, operations=operations)
        # linear transformation for positional encoding
        self.linear_pos = operations.Linear(n_feat, n_feat, bias=False, dtype=dtype, device=device)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.empty(self.h, self.d_k, dtype=dtype, device=device))
        self.pos_bias_v = nn.Parameter(torch.empty(self.h, self.d_k, dtype=dtype, device=device))
        # torch.nn.init.xavier_uniform_(self.pos_bias_u)
        # torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + comfy.model_management.cast_to(self.pos_bias_u, dtype=q.dtype, device=q.device)).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + comfy.model_management.cast_to(self.pos_bias_v, dtype=q.dtype, device=q.device)).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # NOTE(Xiang Lyu): Keep rel_shift since espnet rel_pos_emb is used
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask), new_cache



def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret

def add_optional_chunk_mask(xs: torch.Tensor,
                            masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, 25] or full context(max_len)
            False: chunk size ~ U[1, 25]

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = torch.randint(1, max_len, (1, )).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks,
                                                    (1, )).item()
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        dtype=None, device=None, operations=None
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = operations.LayerNorm(size, eps=1e-5, dtype=dtype, device=device)  # for the FNN module
        self.norm_mha = operations.LayerNorm(size, eps=1e-5, dtype=dtype, device=device)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = operations.LayerNorm(size, eps=1e-5, dtype=dtype, device=device)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = operations.LayerNorm(size, eps=1e-5, dtype=dtype, device=device)  # for the CNN module
            self.norm_final = operations.LayerNorm(
                size, eps=1e-5, dtype=dtype, device=device)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb,
                                              att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache



class EspnetRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Construct an PositionalEncoding object."""
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.size(1), offset=offset)
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - size + 1: self.pe.size(1) // 2 + size,
        ]
        return pos_emb



class LinearEmbed(torch.nn.Module):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module, dtype=None, device=None, operations=None):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            operations.Linear(idim, odim, dtype=dtype, device=device),
            operations.LayerNorm(odim, eps=1e-5, dtype=dtype, device=device),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class #rel_pos_espnet

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)

    def forward(
        self,
        x: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb


ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}

ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

#https://github.com/FunAudioLLM/CosyVoice/blob/main/examples/magicdata-read/cosyvoice/conf/cosyvoice.yaml
class ConformerEncoder(torch.nn.Module):
    """Conformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 1024,
        attention_heads: int = 16,
        linear_units: int = 4096,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = 'linear',
        pos_enc_layer_type: str = 'rel_pos_espnet',
        normalize_before: bool = True,
        static_chunk_size: int = 1, # 1: causal_mask; 0: full_mask
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool =False,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        dtype=None, device=None, operations=None
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility. #'rel_selfattn'
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            key_bias: whether use bias in attention.linear_k, False for whisper models.
        """
        super().__init__()
        self.output_size = output_size
        self.embed = LinearEmbed(input_size, output_size, dropout_rate,
                                        EspnetRelPositionalEncoding(output_size, positional_dropout_rate), dtype=dtype, device=device, operations=operations)
        self.normalize_before = normalize_before
        self.after_norm = operations.LayerNorm(output_size, eps=1e-5, dtype=dtype, device=device)
        self.use_dynamic_chunk = use_dynamic_chunk

        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        activation = ACTIVATION_CLASSES[activation_type]()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)

        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(
                    *encoder_selfattn_layer_args, dtype=dtype, device=device, operations=operations),
                PositionwiseFeedForward(*positionwise_layer_args, dtype=dtype, device=device, operations=operations),
                PositionwiseFeedForward(
                    *positionwise_layer_args, dtype=dtype, device=device, operations=operations) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args, dtype=dtype, device=device, operations=operations) if use_cnn_module else None,
                dropout_rate,
                normalize_before, dtype=dtype, device=device, operations=operations
            ) for _ in range(num_blocks)
        ])

    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward(
        self,
        xs: torch.Tensor,
        pad_mask: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        masks = None
        if pad_mask is not None:
            masks = pad_mask.to(torch.bool).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb = self.embed(xs)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)

        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks


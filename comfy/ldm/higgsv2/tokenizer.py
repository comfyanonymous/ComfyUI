import math
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import torchaudio
import numpy as np
from torch import vmap
from transformers import AutoModel

def WNConv1d(*args, device = None, dtype = None, operations = None, **kwargs):
    return weight_norm(operations.Conv1d(*args, **kwargs, device = device, dtype = dtype))


def WNConvTranspose1d(*args, device = None, dtype = None, operations = None, **kwargs):
    return weight_norm(operations.ConvTranspose1d(*args, **kwargs, device = device, dtype = dtype))


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels, device = None, dtype = None):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, device = device, dtype = dtype))

    def forward(self, x):
        return snake(x, self.alpha)

class DACResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, device = None, dtype = None, operations = None):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim, device = device, dtype = dtype),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad, device = device, dtype = dtype, operations = operations),
            Snake1d(dim, device = device, dtype = dtype),
            WNConv1d(dim, dim, kernel_size=1, device = device, dtype = dtype, operations = operations),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class DACEncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, device = None, dtype = None, operations = None):
        super().__init__()
        self.block = nn.Sequential(
            DACResidualUnit(dim // 2, dilation=1, device = device, dtype = dtype, operations = operations),
            DACResidualUnit(dim // 2, dilation=3, device = device, dtype = dtype, operations = operations),
            DACResidualUnit(dim // 2, dilation=9, device = device, dtype = dtype, operations = operations),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                device = device, dtype = dtype, operations = operations
            ),
        )

    def forward(self, x):
        return self.block(x)


class DACEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 256,
        device = None, dtype = None, operations = None
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3, device = device, dtype = dtype, operations = operations)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [DACEncoderBlock(d_model, stride=stride, device = device, dtype = dtype, operations = operations)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1, device = device, dtype = dtype, operations = operations),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DACDecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, device = None, dtype = None, operations = None):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim, device = device, dtype = dtype),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,  # out_pad,
                device = device, dtype = dtype, operations = operations
            ),
            DACResidualUnit(output_dim, dilation=1, device = device, dtype = dtype, operations = operations),
            DACResidualUnit(output_dim, dilation=3, device = device, dtype = dtype, operations = operations),
            DACResidualUnit(output_dim, dilation=9, device = device, dtype = dtype, operations = operations),
        )

    def forward(self, x):
        return self.block(x)


class DACDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        device = None, dtype = None, operations = None
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3, device = device, dtype = dtype, operations = operations )]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DACDecoderBlock(input_dim, output_dim, stride, device = device, dtype = dtype, operations = operations)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim, device = device, dtype = dtype),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3, device = device, dtype = dtype, operations = operations),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Conv1d1x1:
    def __new__(cls, in_channels, out_channels, bias=True, device=None, dtype=None, operations=None):
        operations = operations or nn
        return operations.Conv1d(
            in_channels, out_channels, kernel_size=1,
            bias=bias, device=device, dtype=dtype
        )

class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = -1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device = None, dtype = None, operations = None
    ):
        super().__init__()

        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation

        self.dilation = dilation
        self.conv = operations.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device = device, dtype = dtype
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding=-1,
        output_padding=-1,
        groups=1,
        bias=True,
        device = None, dtype = None, operations = None
    ):
        super().__init__()
        if padding < 0:
            padding = (stride + 1) // 2
        if output_padding < 0:
            output_padding = 1 if stride % 2 else 0
        self.deconv = operations.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            device = device, dtype = dtype
        )

    def forward(self, x):
        x = self.deconv(x)
        return x

class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        dilation=1,
        bias=False,
        nonlinear_activation="ELU",
        nonlinear_activation_params={},
        device = None, dtype = None, operations = None
    ):
        super().__init__()
        self.activation = getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
            device = device, dtype = dtype, operations = operations
        )
        self.conv2 = Conv1d1x1(out_channels, out_channels, bias, device = device, dtype = dtype, operations = operations)

    def forward(self, x):
        y = self.conv1(self.activation(x))
        y = self.conv2(self.activation(y))
        return x + y


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int, dilations=(1, 1), unit_kernel_size=3, bias=True, device = None, dtype = None, operations = None
    ):
        super().__init__()
        self.res_units = torch.nn.ModuleList()
        for dilation in dilations:
            self.res_units += [ResidualUnit(in_channels, in_channels, kernel_size=unit_kernel_size, dilation=dilation, device = device, dtype = dtype, operations = operations)]
        self.num_res = len(self.res_units)

        kernel_size=3 if stride == 1 else (2 * stride) # special case: stride=1, do not use kernel=2
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size = kernel_size,
            stride=stride,
            bias=bias,
            device = device, dtype = dtype, operations = operations
        )

    def forward(self, x):
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        encode_channels: int,
        channel_ratios=(1, 1),
        strides=(1, 1),
        kernel_size=3,
        bias=True,
        block_dilations=(1, 1),
        unit_kernel_size=3,
        device = None, dtype = None, operations = None
    ):
        super().__init__()
        assert len(channel_ratios) == len(strides)
        self.conv = Conv1d(
            in_channels=input_channels, out_channels=encode_channels, kernel_size=kernel_size, stride=1, bias=False,
            device = device, dtype = dtype, operations = operations
        )
        self.conv_blocks = torch.nn.ModuleList()
        in_channels = encode_channels
        for idx, stride in enumerate(strides):
            out_channels = int(encode_channels * channel_ratios[idx])  # could be float
            self.conv_blocks += [
                EncoderBlock(
                    in_channels,
                    out_channels,
                    stride,
                    dilations=block_dilations,
                    unit_kernel_size=unit_kernel_size,
                    bias=bias,
                    device = device, dtype = dtype, operations = operations
                )
            ]
            in_channels = out_channels
        self.num_blocks = len(self.conv_blocks)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block (no up-sampling)"""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, dilations=(1, 1), unit_kernel_size=3, bias=True, device = None, dtype = None, operations = None
    ):
        super().__init__()

        if stride == 1:
            self.conv = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,  # fix kernel=3 when stride=1 for unchanged shape
                stride=stride,
                bias=bias,
                device = device, dtype = dtype, operations = operations
            )
        else:
            self.conv = ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(2 * stride),
                stride=stride,
                bias=bias,
                device = device, dtype = dtype, operations = operations
            )

        self.res_units = nn.ModuleList([
            ResidualUnit(out_channels, out_channels, kernel_size=unit_kernel_size, dilation=d, device = device, dtype = dtype, operations = operations)
            for d in dilations
        ])

        self.num_res = len(self.res_units)

    def forward(self, x):
        x = self.conv(x)
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        code_dim: int,
        output_channels: int,
        decode_channels: int,
        channel_ratios=(1, 1),
        strides=(1, 1),
        kernel_size=3,
        bias=True,
        block_dilations=(1, 1),
        unit_kernel_size=3,
        device = None, dtype = None, operations = None
    ):
        super().__init__()
        assert len(channel_ratios) == len(strides)
        self.conv1 = Conv1d(
            in_channels=code_dim,
            out_channels=int(decode_channels * channel_ratios[0]),
            kernel_size=kernel_size,
            stride=1,
            bias=False,
            device = device, dtype = dtype, operations = operations
        )

        self.conv_blocks = torch.nn.ModuleList()
        for idx, stride in enumerate(strides):
            in_channels = int(decode_channels * channel_ratios[idx])
            if idx < (len(channel_ratios) - 1):
                out_channels = int(decode_channels * channel_ratios[idx + 1])
            else:
                out_channels = decode_channels
            self.conv_blocks += [
                DecoderBlock(
                    in_channels,
                    out_channels,
                    stride,
                    dilations=block_dilations,
                    unit_kernel_size=unit_kernel_size,
                    bias=bias,
                    device = device, dtype = dtype, operations = operations
                )
            ]
        self.num_blocks = len(self.conv_blocks)

        self.conv2 = Conv1d(out_channels, output_channels, kernel_size = 3, bias=False, device = device, dtype = dtype, operations = operations)

    def forward(self, z):
        x = self.conv1(z)
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
        x = self.conv2(x)
        return x

class HiggsAudioFeatureExtractor(nn.Module):
    def __init__(self, sampling_rate=16000):
        super().__init__()
        self.sampling_rate = sampling_rate

    def forward(self, audio_signal):
        audio_signal = audio_signal.unsqueeze(0)
        if len(audio_signal.shape) < 3:
            audio_signal = audio_signal.unsqueeze(0)
        return {"input_values": audio_signal}

def uniform_init(*shape: int, device = None, dtype = None):
    t = torch.empty(shape, device = device, dtype = dtype)
    nn.init.kaiming_uniform_(t)
    return t

class EuclideanCodebook(nn.Module):

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        device = None, dtype = None
    ):
        super().__init__()
        self.decay = decay
        init_fn = uniform_init
        embed = init_fn(codebook_size, dim, device = device, dtype = dtype)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # Flag variable to indicate whether the codebook is initialized
        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        # Runing EMA cluster size/count: N_i^t in eq. (6) in vqvae paper
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        # Codebook
        self.register_buffer("embed", embed)
        # EMA codebook: eq. (7) in vqvae paper
        self.register_buffer("embed_avg", embed.clone())

    def preprocess(self, x):
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        embed = self.embed.t()
        if x.dtype != embed.dtype:
            x = x.to(embed.dtype)

        dist = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed + embed.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)  # [B, T, D] -> [B*T, D]
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        orig_shape = x.shape  # [B, T, D]
        flat = x.view(-1, x.shape[-1]) # [B*T, D]

        embed_ind = self.quantize(flat)
        embed_ind = self.postprocess_emb(embed_ind, orig_shape)
        # now embed_ind has shape [B, T]

        quantize = self.dequantize(embed_ind)
        # quantize: [B, T, D]

        return quantize, embed_ind

class VectorQuantization(nn.Module):

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.0,
        device = None, dtype = None, operations = None
    ):
        super().__init__()
        _codebook_dim: int = codebook_dim if codebook_dim is not None else dim

        requires_projection = _codebook_dim != dim
        self.project_in = operations.Linear(dim, _codebook_dim, device = device, dtype = dtype) if requires_projection else nn.Identity()
        self.project_out = operations.Linear(_codebook_dim, dim, device = device, dtype = dtype) if requires_projection else nn.Identity()

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
            device = device, dtype = dtype
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = x.permute(0, 2, 1)
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = quantize.permute(0, 2, 1)
        return quantize

    def forward(self, x):
        device = x.device
        x = x.transpose(1, 2).contiguous()  # [b d n] -> [b n d]
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        quantize = self.project_out(quantize)
        quantize = quantize.transpose(1, 2).contiguous()  # [b n d] -> [b d n]
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    def __init__(self, *, num_quantizers, device = None, dtype = None, operations = None, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(device = device, dtype = dtype, operations = operations, **kwargs) for _ in range(num_quantizers)])

    def forward(self, x, n_q: Optional[int] = None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        """ Vectorized Implementation of dequantization | 2x faster than original impl """

        biases = torch.stack([layer.project_out.bias for layer in self.layers])

        codebook_device = self.layers[0]._codebook.embed.device
        q_indices = q_indices.to(codebook_device)

        def decode_one(codebook_weight, proj_weight, embed_id, proj_biases):
            quantized = F.embedding(embed_id, codebook_weight).transpose(1, 2)  # (B, D, T)
            quantized = F.linear(quantized.transpose(1, 2), proj_weight, proj_biases).transpose(1, 2)
            return quantized

        codebook_weights = torch.stack([q._codebook.embed for q in self.layers])       # (n_codebooks, vocab_size, D)
        proj_weights = torch.stack([q.project_out.weight for q in self.layers])

        quantized = vmap(decode_one)(codebook_weights, proj_weights, q_indices, biases)

        return quantized.sum(0)

class ResidualVectorQuantizer(nn.Module):

    def __init__(
        self,
        dimension: int = 256,
        codebook_dim: int = None,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        device = None,
        dtype = None,
        operations = None
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.codebook_dim = codebook_dim
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_dim=self.codebook_dim,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            device = device, dtype = dtype, operations = operations
        )

    def forward(self, x: torch.Tensor, sample_rate: int, bandwidth: Optional[float] = None):  # -> QuantizedResult:

        bw_per_q = self.get_bandwidth_per_quantizer(sample_rate)
        n_q = self.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        quantized, codes, commit_loss = self.vq(x, n_q=n_q)
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return quantized, codes, bw, torch.mean(commit_loss)

    def get_num_quantizers_for_bandwidth(self, sample_rate: int, bandwidth: Optional[float] = None) -> int:
        """Return n_q based on specified target bandwidth."""
        bw_per_q = self.get_bandwidth_per_quantizer(sample_rate)
        n_q = self.n_q
        if bandwidth and bandwidth > 0.0:
            n_q = int(max(1, math.floor(bandwidth / bw_per_q)))
        return n_q

    def get_bandwidth_per_quantizer(self, sample_rate: int):
        """Return bandwidth per quantizer for a given input sample rate."""
        return math.log2(self.bins) * sample_rate / 1000

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        quantized = self.vq.decode(codes)
        return quantized

class HiggsAudioTokenizer(nn.Module):
    def __init__(
        self,
        D: int = 256,
        target_bandwidths= [0.5, 1, 1.5, 2, 4],
        ratios = [8, 5, 4, 2, 3],  #  downsampling by 320
        sample_rate: int = 24000,
        bins: int = 1024,
        n_q: int = 8,
        codebook_dim: int = 64,
        last_layer_semantic: bool = True,
        downsample_mode: str = "step_down",
        vq_scale: int = 1,
        semantic_sample_rate: int = None,
        device = None,
        dtype = None,
        operations = None,
        **kwargs
    ):
        super().__init__()
        operations = operations or nn
        self.hop_length = np.prod(ratios)

        self.frame_rate = math.ceil(sample_rate / np.prod(ratios))  # 50 Hz

        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.sample_rate = sample_rate
        self.encoder = DACEncoder(64, ratios, D, device = device, dtype = dtype, operations = operations)

        self.decoder_2 = DACDecoder(D, 1024, ratios, device = device, dtype = dtype, operations = operations)
        self.last_layer_semantic = last_layer_semantic
        self.device = device

        self.semantic_model = AutoModel.from_pretrained("bosonai/hubert_base", trust_remote_code=True)
        self.semantic_sample_rate = 16000
        self.semantic_dim = 768
        self.encoder_semantic_dim = 768

        # Overwrite semantic model sr to ensure semantic_downsample_factor is an integer
        if semantic_sample_rate is not None:
            self.semantic_sample_rate = semantic_sample_rate

        self.semantic_model.eval()

        # make the semantic model parameters do not need gradient
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        self.semantic_downsample_factor = int(self.hop_length / (self.sample_rate / self.semantic_sample_rate) / 320)

        self.quantizer_dim = int((D + self.encoder_semantic_dim) // vq_scale)
        self.encoder_semantic = Encoder(input_channels=self.semantic_dim, encode_channels=self.encoder_semantic_dim, device = device, dtype = dtype, operations = operations)
        self.decoder_semantic = Decoder(
            code_dim=self.encoder_semantic_dim, output_channels=self.semantic_dim, decode_channels=self.semantic_dim, device = device, dtype = dtype, operations = operations
        )

        self.quantizer = ResidualVectorQuantizer(
            dimension=self.quantizer_dim, codebook_dim=codebook_dim, n_q=n_q, bins=bins, device = device, dtype = dtype, operations = operations
        )

        self.fc_prior = operations.Linear(D + self.encoder_semantic_dim, self.quantizer_dim, device = device, dtype = dtype)
        self.fc_post1 = operations.Linear(self.quantizer_dim, self.encoder_semantic_dim, device = device, dtype = dtype)
        self.fc_post2 = operations.Linear(self.quantizer_dim, D, device = device, dtype = dtype)

        self.downsample_mode = downsample_mode

        self.audio_tokenizer_feature_extractor = HiggsAudioFeatureExtractor(sampling_rate=self.sample_rate)

    @property
    def sampling_rate(self):
        return self.sample_rate

    @torch.no_grad()
    def get_regress_target(self, x):
        x = torchaudio.functional.resample(x, self.sample_rate, self.semantic_sample_rate)

        x = x[:, 0, :]
        x = F.pad(x, (160, 160))
        target = self.semantic_model(x, output_hidden_states=True).hidden_states
        target = torch.stack(target, dim=1)

        target = target.mean(1)

        if self.downsample_mode == "step_down":
            if self.semantic_downsample_factor > 1:
                target = target[:, :: self.semantic_downsample_factor, :]

        return target

    def forward(self):
        pass

    @property
    def tps(self):
        return self.frame_rate

    def encode(self, wv, sr):

        if sr != self.sampling_rate:
            # best computed values to match librosa's resample
            resampler_torch = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.sampling_rate,
                resampling_method="sinc_interp_kaiser",
                lowpass_filter_width = 121,
                rolloff = 0.9568384289091556,
                beta = 21.01531462440614
            ).to(wv.device)

            wv = resampler_torch(wv)

        if self.audio_tokenizer_feature_extractor is not None:
            inputs = self.audio_tokenizer_feature_extractor(wv)
            input_values = inputs["input_values"].to(self.device)
        else:
            input_values = torch.from_numpy(wv).float().unsqueeze(0)
        with torch.no_grad():
            input_values = input_values.to(wv.device)
            encoder_outputs = self._xcodec_encode(input_values)
            vq_code = encoder_outputs[0]
        return vq_code

    def _xcodec_encode(self, x: torch.Tensor, target_bw: Optional[int] = None) -> torch.Tensor:
        bw = target_bw

        e_semantic_input = self.get_regress_target(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            pad_size = 160 * self.semantic_downsample_factor
            e_acoustic = self.encoder(F.pad(x[:, 0, :], (pad_size, pad_size)).unsqueeze(0))

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            if e_acoustic.shape[2] > e_semantic.shape[2]:
                e_acoustic = e_acoustic[:, :, : e_semantic.shape[2]]
            else:
                e_semantic = e_semantic[:, :, : e_acoustic.shape[2]]

        e = torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2))

        e = e.transpose(1, 2)
        _, codes, _, _ = self.quantizer(e, self.frame_rate, bw)
        codes = codes.permute(1, 0, 2)

        return codes

    def decode(self, vq_code: torch.Tensor) -> torch.Tensor:
        vq_code = vq_code.to(self.device)

        if vq_code.ndim < 3:
            vq_code = vq_code.unsqueeze(0)

        vq_code = vq_code.permute(1, 0, 2)
        quantized = self.quantizer.decode(vq_code)
        quantized = quantized.transpose(1, 2)
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)
        return o.detach()

import copy
import math
import torch
import scipy
import torchaudio
import numpy as np
import torch.nn.functional as F
from typing import Optional, List

# defaulted to the new pytorch api
def _new_rfft(x: torch.Tensor):
    z = torch.fft.rfft(x, dim=-1)
    return torch.view_as_real(z)

def _new_irfft(x: torch.Tensor, length: int):
    x = torch.view_as_complex(x)
    return torch.fft.irfft(x, length, dim=-1)

def _compl_mul_conjugate(a: torch.Tensor, b: torch.Tensor):
    # changed this function to use the pytorch api
    return torch.view_as_real(torch.view_as_complex(a) * torch.view_as_complex(b).conj())

def unfold(input, kernel_size: int, stride: int):

    shape = list(input.shape)
    length = shape.pop(-1)

    n_frames = math.ceil((max(length, kernel_size) - kernel_size) / stride) + 1
    tgt_length = (n_frames - 1) * stride + kernel_size

    padded = F.pad(input, (0, tgt_length - length)).contiguous()
    strides: List[int] = []

    for dim in range(padded.dim()):
        strides.append(padded.stride(dim))

    last_stride = strides.pop(-1)
    assert last_stride == 1, 'data should be contiguous'

    strides = strides + [stride, 1]
    return padded.as_strided(shape + [n_frames, kernel_size], strides)

# convert the signal and filter to frequency domain, multiply them, then inverse FFT to get back to time-domain
# faster than a sliding window over time-domain.
def fft_conv1d(
        input: torch.Tensor, weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None, stride: int = 1, padding: int = 0,
        block_ratio: float = 5):

    input = F.pad(input, (padding, padding))
    batch, _, length = input.shape
    out_channels, _, kernel_size = weight.shape

    _rfft = _new_rfft
    _irfft = _new_irfft

    if length < kernel_size:
        raise RuntimeError(f"Input should be at least as large as the kernel size {kernel_size}, "
                           f"but it is only {length} samples long.")
    if block_ratio < 1:
        raise RuntimeError("Block ratio must be greater than 1.")

    # We are going to process the input blocks by blocks, as for some reason it is faster
    # and less memory intensive (I think the culprit is `torch.einsum`.
    block_size: int = min(int(kernel_size * block_ratio), length)
    fold_stride = block_size - kernel_size + 1

    # replaces to_pad
    weight = F.pad(weight, (0, block_size - weight.shape[-1]), mode = "constant", value = 0)
    weight_z = _rfft(weight)

    # We pad the input and get the different frames, on which
    frames = unfold(input, block_size, fold_stride)

    frames_z = _rfft(frames)
    out_z = _compl_mul_conjugate(frames_z, weight_z)
    out = _irfft(out_z, block_size)
    # The last bit is invalid, because FFT will do a circular convolution.
    out = out[..., :-kernel_size + 1]
    out = out.reshape(batch, out_channels, -1)
    out = out[..., ::stride]
    target_length = (length - kernel_size) // stride + 1
    out = out[..., :target_length]
    if bias is not None:
        out += bias[:, None]
    return out

class IIRfilter(object):

    def __init__(self, G, Q, fc, rate, filter_type, passband_gain=1.0):
        self.G  = G
        self.Q  = Q
        self.fc = fc
        self.rate = rate
        self.filter_type = filter_type
        self.passband_gain = passband_gain

    def generate_coefficients(self):

        A  = 10**(self.G/40.0)
        w0 = 2.0 * np.pi * (self.fc / self.rate)
        alpha = np.sin(w0) / (2.0 * self.Q)

        if self.filter_type == 'high_shelf':
            b0 =      A * ( (A+1) + (A-1) * np.cos(w0) + 2 * np.sqrt(A) * alpha )
            b1 = -2 * A * ( (A-1) + (A+1) * np.cos(w0)                          )
            b2 =      A * ( (A+1) + (A-1) * np.cos(w0) - 2 * np.sqrt(A) * alpha )
            a0 =            (A+1) - (A-1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 =      2 * ( (A-1) - (A+1) * np.cos(w0)                          )
            a2 =            (A+1) - (A-1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        elif self.filter_type == 'high_pass':
            b0 =  (1 + np.cos(w0))/2
            b1 = -(1 + np.cos(w0))
            b2 =  (1 + np.cos(w0))/2
            a0 =   1 + alpha
            a1 =  -2 * np.cos(w0)
            a2 =   1 - alpha

        return np.array([b0, b1, b2])/a0, np.array([a0, a1, a2])/a0

    def apply_filter(self, data):
        return self.passband_gain * scipy.signal.lfilter(self.b, self.a, data)

    @property
    def b_and_a(self):
        return self.generate_coefficients()

class Meter(torch.nn.Module):

    def __init__(
        self,
        rate: int,
        filter_class: str = "K-weighting",
        block_size: float = 0.400,
        zeros: int = 512,
        use_fir: bool = False,
    ):
        super().__init__()

        self.rate = rate
        self.filter_class = filter_class
        self.block_size = block_size
        self.use_fir = use_fir

        G = torch.from_numpy(np.array([1.0, 1.0, 1.0, 1.41, 1.41]))
        self.register_buffer("G", G)

        self._filters = {}
        self._filters['high_shelf'] = IIRfilter(4.0, 1/np.sqrt(2), 1500.0, self.rate, 'high_shelf')
        self._filters['high_pass'] = IIRfilter(0.0, 0.5, 38.0, self.rate, 'high_pass')

        # Compute impulse responses so that filtering is fast via
        # a convolution at runtime, on GPU, unlike lfilter.
        impulse = np.zeros((zeros,))
        impulse[..., 0] = 1.0

        firs = np.zeros((len(self._filters), 1, zeros))
        passband_gain = torch.tensor([filter.passband_gain for filter in self._filters.values()])

        for i, (_, filter_stage) in enumerate(self._filters.items()):
            b, a = filter_stage.b_and_a
            firs[i] = scipy.signal.lfilter(b, a, impulse)

        firs = torch.from_numpy(firs[..., ::-1].copy()).float()

        self.register_buffer("firs", firs)
        self.register_buffer("passband_gain", passband_gain)

    def apply_filter_gpu(self, data: torch.Tensor):

        # Data is of shape (nb, nch, nt)
        # Reshape to (nb*nch, 1, nt)
        nb, nt, nch = data.shape
        data = data.permute(0, 2, 1)
        data = data.reshape(nb * nch, 1, nt)

        # Apply padding
        pad_length = self.firs.shape[-1]

        # Apply filtering in sequence
        for i in range(self.firs.shape[0]):
            data = F.pad(data, (pad_length, pad_length))
            data = fft_conv1d(data, self.firs[i, None, ...])
            data = self.passband_gain[i] * data
            data = data[..., 1 : nt + 1]

        data = data.permute(0, 2, 1)
        data = data[:, :nt, :]
        return data

    def apply_filter_cpu(self, data: torch.Tensor):
        for _, filter_stage in self._filters.items():
            passband_gain = filter_stage.passband_gain
            b, a = filter_stage.b_and_a

            a_coeffs = torch.from_numpy(a).float().to(data.device)
            b_coeffs = torch.from_numpy(b).float().to(data.device)

            _data = data.permute(0, 2, 1)
            filtered = torchaudio.functional.lfilter(
                _data, a_coeffs, b_coeffs, clamp=False
            )
            data = passband_gain * filtered.permute(0, 2, 1)
        return data

    def apply_filter(self, data: torch.Tensor):
        if data.is_cuda or self.use_fir:
            data = self.apply_filter_gpu(data)
        else:
            data = self.apply_filter_cpu(data)
        return data

    def forward(self, data: torch.Tensor):
        return self.integrated_loudness(data)

    def _unfold(self, input_data):
        T_g = self.block_size
        overlap = 0.75  # overlap of 75% of the block duration
        step = 1.0 - overlap  # step size by percentage

        kernel_size = int(T_g * self.rate)
        stride = int(T_g * self.rate * step)
        unfolded = unfold(input_data.permute(0, 2, 1), kernel_size, stride)
        unfolded = unfolded.transpose(-1, -2)

        return unfolded

    def integrated_loudness(self, data: torch.Tensor):

        if not torch.is_tensor(data):
            data = torch.from_numpy(data).float()
        else:
            data = data.float()

        input_data = copy.copy(data)
        # Data always has a batch and channel dimension.
        # Is of shape (nb, nt, nch)
        if input_data.ndim < 2:
            input_data = input_data.unsqueeze(-1)
        if input_data.ndim < 3:
            input_data = input_data.unsqueeze(0)

        nb, _, nch = input_data.shape

        # Apply frequency weighting filters - account
        # for the acoustic respose of the head and auditory system
        input_data = self.apply_filter(input_data)

        G = self.G  # channel gains
        T_g = self.block_size  # 400 ms gating block standard
        Gamma_a = -70.0  # -70 LKFS = absolute loudness threshold

        unfolded = self._unfold(input_data)

        z = (1.0 / (T_g * self.rate)) * unfolded.square().sum(2)
        l = -0.691 + 10.0 * torch.log10((G[None, :nch, None] * z).sum(1, keepdim=True))
        l = l.expand_as(z)

        # find gating block indices above absolute threshold
        z_avg_gated = z
        z_avg_gated[l <= Gamma_a] = 0
        masked = l > Gamma_a
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2)

        # calculate the relative threshold value (see eq. 6)
        Gamma_r = (
            -0.691 + 10.0 * torch.log10((z_avg_gated * G[None, :nch]).sum(-1)) - 10.0
        )
        Gamma_r = Gamma_r[:, None, None]
        Gamma_r = Gamma_r.expand(nb, nch, l.shape[-1])

        # find gating block indices above relative and absolute thresholds  (end of eq. 7)
        z_avg_gated = z
        z_avg_gated[l <= Gamma_a] = 0
        z_avg_gated[l <= Gamma_r] = 0
        masked = (l > Gamma_a) * (l > Gamma_r)
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2)

        # # Cannot use nan_to_num (pytorch 1.8 does not come with GCP-supported cuda version)
        # z_avg_gated = torch.nan_to_num(z_avg_gated)
        z_avg_gated = torch.where(
            z_avg_gated.isnan(), torch.zeros_like(z_avg_gated), z_avg_gated
        )
        z_avg_gated[z_avg_gated == float("inf")] = float(np.finfo(np.float32).max)
        z_avg_gated[z_avg_gated == -float("inf")] = float(np.finfo(np.float32).min)

        LUFS = -0.691 + 10.0 * torch.log10((G[None, :nch] * z_avg_gated).sum(1))
        return LUFS.float()


def loudness(
    audio_data, sample_rate: int, target_loudness: int, filter_class: str = "K-weighting", block_size: float = 0.400, **kwargs
):
    MIN_LOUDNESS = -70
    device = audio_data.device

    original_length = audio_data.shape[-1]
    signal_duration = original_length / sample_rate

    # Pad if too short
    if signal_duration < 0.5:
        pad_len = int((0.5 - signal_duration) * sample_rate)
        audio_data = torch.nn.functional.pad(audio_data, (0, pad_len), mode="constant", value=0)

    # create BS.1770 meter
    meter = Meter(
        sample_rate, filter_class=filter_class, block_size=block_size, **kwargs
    )
    meter = meter.to(audio_data.device)
    # measure loudness
    loudness = meter.integrated_loudness(audio_data.permute(0, 2, 1))
    audio_data = audio_data[..., :original_length]
    min_loudness = (
        torch.ones_like(loudness, device=loudness.device) * MIN_LOUDNESS
    )
    _loudness = torch.maximum(loudness, min_loudness)

    _loudness = _loudness.to(device)

    delta_loudness = target_loudness - _loudness
    gain = torch.pow(torch.tensor(10.0, device=device, dtype=audio_data.dtype), delta_loudness / 20.0)

    output = gain * audio_data

    if torch.max(torch.abs(output)) >= 1.0:
        import warnings
        warnings.warn("Possible clipped samples in output.")

    return output

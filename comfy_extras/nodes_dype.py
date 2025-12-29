# adapted from https://github.com/guyyariv/DyPE

import math
from typing import Callable

import numpy as np
import torch
from typing_extensions import override

from comfy.model_patcher import ModelPatcher
from comfy_api.latest import ComfyExtension, io


def find_correction_factor(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )  # Inverse dim formula to find number of rotations


def find_correction_range(low_ratio, high_ratio, dim, base, ori_max_pe_len):
    """Find the correction range for NTK-by-parts interpolation"""
    low = np.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = np.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def find_newbase_ntk(dim, base, scale):
    """Calculate the new base for NTK-aware scaling"""
    return base * (scale ** (dim / (dim - 2)))


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,
    yarn=False,
    max_pe_len=None,
    ori_max_pe_len=64,
    current_timestep=1.0,
):
    """
    Precompute the frequency tensor for complex exponentials with RoPE.
    Supports YARN interpolation for vision transformers.

    Args:
        dim (`int`):
            Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`):
            Position indices for the frequency tensor. [S] or scalar.
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation.
        use_real (`bool`, *optional*, defaults to False):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for linear interpolation.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for NTK-Aware RoPE.
        repeat_interleave_real (`bool`, *optional*, defaults to True):
            If True and use_real, real and imaginary parts are interleaved with themselves to reach dim.
            Otherwise, they are concatenated.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            Data type of the frequency tensor.
        yarn (`bool`, *optional*, defaults to False):
            If True, use YARN interpolation combining NTK, linear, and base methods.
        max_pe_len (`int`, *optional*):
            Maximum position encoding length (current patches for vision models).
        ori_max_pe_len (`int`, *optional*, defaults to 64):
            Original maximum position encoding length (base patches for vision models).
        current_timestep (`float`, *optional*, defaults to 1.0):
            Current timestep for DyPE, normalized to [0, 1] where 1 is pure noise.

    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
            If use_real=True, returns tuple of (cos, sin) tensors.
    """
    assert dim % 2 == 0

    device = pos.device

    if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
        if not isinstance(max_pe_len, torch.Tensor):
            max_pe_len = torch.tensor(max_pe_len, dtype=freqs_dtype, device=device)

        scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)

        beta_0, beta_1 = 1.25, 0.75
        gamma_0, gamma_1 = 16, 2

        freqs_base = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)
        )

        freqs_linear = 1.0 / torch.einsum(
            "..., f -> ... f",
            scale,
            (
                theta
                ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)
            ),
        )

        new_base = find_newbase_ntk(dim, theta, scale)
        if new_base.dim() > 0:
            new_base = new_base.view(-1, 1)
        freqs_ntk = 1.0 / torch.pow(
            new_base, (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)
        )
        if freqs_ntk.dim() > 1:
            freqs_ntk = freqs_ntk.squeeze()

        beta_0 = beta_0 ** (2.0 * (current_timestep**2.0))
        beta_1 = beta_1 ** (2.0 * (current_timestep**2.0))

        low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
        low = max(0, low)
        high = min(dim // 2, high)

        freqs_mask = 1 - linear_ramp_mask(low, high, dim // 2).to(device).to(
            freqs_dtype
        )
        freqs = freqs_linear * (1 - freqs_mask) + freqs_ntk * freqs_mask

        gamma_0 = gamma_0 ** (2.0 * (current_timestep**2.0))
        gamma_1 = gamma_1 ** (2.0 * (current_timestep**2.0))

        low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
        low = max(0, low)
        high = min(dim // 2, high)

        freqs_mask = 1 - linear_ramp_mask(low, high, dim // 2).to(device).to(
            freqs_dtype
        )
        freqs = freqs * (1 - freqs_mask) + freqs_base * freqs_mask

    else:
        theta_ntk = theta * ntk_factor
        freqs = (
            1.0
            / (
                theta_ntk
                ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)
            )
            / linear_factor
        )

    freqs = pos.unsqueeze(-1) * freqs

    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()

    if use_real:
        if repeat_interleave_real:
            freqs_cos = (
                freqs.cos()
                .repeat_interleave(2, dim=-1, output_size=freqs.shape[-1] * 2)
                .float()
            )
            freqs_sin = (
                freqs.sin()
                .repeat_interleave(2, dim=-1, output_size=freqs.shape[-1] * 2)
                .float()
            )

            if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
                mscale = torch.where(
                    scale <= 1.0, torch.tensor(1.0), 0.1 * torch.log(scale) + 1.0
                ).to(scale)
                freqs_cos = freqs_cos * mscale
                freqs_sin = freqs_sin * mscale

            return freqs_cos, freqs_sin
        else:
            freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
            freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
            return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class FluxPosEmbed(torch.nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], method: str = "yarn"):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.base_resolution = 1024
        self.base_patches = (self.base_resolution // 8) // 2
        self.method = method
        self.current_timestep = 1.0

    def set_timestep(self, timestep: float):
        """Set current timestep for DyPE. Timestep normalized to [0, 1] where 1 is pure noise."""
        self.current_timestep = timestep

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if ids.device.type == "cuda" else torch.float32

        for i in range(n_axes):
            axis_dim = self.axes_dim[i]
            axis_pos = pos[..., i]

            common_kwargs = {
                "dim": axis_dim,
                "pos": axis_pos,
                "theta": self.theta,
                "repeat_interleave_real": True,
                "use_real": True,
                "freqs_dtype": freqs_dtype,
            }

            max_pos = axis_pos.max().item()
            current_patches = max_pos + 1

            if i == 0 or current_patches <= self.base_patches:
                cos, sin = get_1d_rotary_pos_embed(**common_kwargs)

            elif self.method == "yarn":
                max_pe_len = torch.tensor(
                    current_patches, dtype=freqs_dtype, device=pos.device
                )
                cos, sin = get_1d_rotary_pos_embed(
                    **common_kwargs,
                    yarn=True,
                    max_pe_len=max_pe_len,
                    ori_max_pe_len=self.base_patches,
                    current_timestep=self.current_timestep,
                )

            elif self.method == "ntk":
                base_ntk = (current_patches / self.base_patches) ** (
                    self.axes_dim[i] / (self.axes_dim[i] - 2)
                )
                ntk_factor = base_ntk ** (2.0 * (self.current_timestep**2.0))
                ntk_factor = max(1.0, ntk_factor)

                cos, sin = get_1d_rotary_pos_embed(
                    **common_kwargs, ntk_factor=ntk_factor
                )

            cos_out.append(cos)
            sin_out.append(sin)

        emb_parts = []
        for cos, sin in zip(cos_out, sin_out):
            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3)
        return emb.unsqueeze(1).to(ids.device)


def apply_dype_flux(model: ModelPatcher, method: str) -> ModelPatcher:
    _pe_embedder = model.model.diffusion_model.pe_embedder
    _theta, _axes_dim = _pe_embedder.theta, _pe_embedder.axes_dim

    pos_embedder = FluxPosEmbed(_theta, _axes_dim, method)
    model.add_object_patch("diffusion_model.pe_embedder", pos_embedder)

    sigma_max: float = model.model.model_sampling.sigma_max.item()

    def dype_wrapper_function(apply_model: Callable, args: dict):
        timestep: torch.Tensor = args["timestep"]
        sigma: float = timestep.item()

        normalized_timestep = min(max(sigma / sigma_max, 0.0), 1.0)
        pos_embedder.set_timestep(normalized_timestep)

        return apply_model(args["input"], timestep, **args["c"])

    model.set_model_unet_function_wrapper(dype_wrapper_function)

    return model


class DyPEPatchModelFlux(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DyPEPatchModelFlux",
            display_name="DyPE Patch Model (Flux)",
            category="_for_testing",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("method", options=["yarn", "ntk"], default="yarn"),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model: ModelPatcher, method: str) -> io.NodeOutput:
        model = model.clone()
        model = apply_dype_flux(model, method)
        return io.NodeOutput(model)


class DyPEExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DyPEPatchModelFlux,
        ]


async def comfy_entrypoint():
    return DyPEExtension()

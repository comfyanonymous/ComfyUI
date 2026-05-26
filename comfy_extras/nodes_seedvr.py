from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import torch
import math
import logging
from einops import rearrange

import gc
import comfy.model_management
import comfy.sample
import comfy.samplers
from comfy.ldm.seedvr.vae import (
    adain_color_transfer,
    lab_color_transfer,
    wavelet_color_transfer,
)

from torchvision.transforms import functional as TVF
from torchvision.transforms import Lambda
from torchvision.transforms.functional import InterpolationMode


_SEEDVR2_INVALID_MODEL_MSG_PREFIX = (
    "SeedVR2Conditioning: model object does not match expected SeedVR2 structure"
)
LAB_SCALE_MULTIPLIER = 13
WAVELET_SCALE_MULTIPLIER = 10
ADAIN_SCALE_MULTIPLIER = 6
COLOR_CORRECTION_MEMORY_HEADROOM = 0.75

# Private sentinel for getattr default: distinguishes "attribute missing"
# from "attribute present but None" so the failure message is accurate.
_ATTR_MISSING = object()


def _seedvr2_auto_chunk_attempts(t_latent, t_pixel, frames_per_chunk):
    """Return stricter 4n+1 frame chunk sizes for auto OOM retries."""
    attempts = [frames_per_chunk]
    current_chunk_latent = (
        t_latent if t_pixel <= frames_per_chunk
        else (frames_per_chunk - 1) // 4 + 1
    )
    current_chunk_count = max(1, math.ceil(t_latent / current_chunk_latent))
    seen = {frames_per_chunk}

    for target_chunks in range(max(2, current_chunk_count + 1), t_latent + 1):
        chunk_latent = max(1, math.ceil(t_latent / target_chunks))
        candidate = 4 * (chunk_latent - 1) + 1
        if candidate in seen:
            continue
        if candidate >= attempts[-1]:
            continue
        attempts.append(candidate)
        seen.add(candidate)

    return attempts


def _resolve_seedvr2_diffusion_model(model):
    """Resolve the inner SeedVR2 diffusion-model module from a ComfyUI model
    patcher object. Fails loud with a ``RuntimeError`` whose message begins
    with ``_SEEDVR2_INVALID_MODEL_MSG_PREFIX`` when the expected wrapper
    shape (``model.model.diffusion_model``) is absent.

    Distinguishes four failure modes via the ``_ATTR_MISSING`` sentinel:
    ``model.model`` missing, ``model.model is None``,
    ``model.model.diffusion_model`` missing, ``model.model.diffusion_model
    is None``. Each mode produces an accurate error message rather than
    conflating "attribute missing" with "attribute is None".
    """
    inner = getattr(model, "model", _ATTR_MISSING)
    if inner is _ATTR_MISSING:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: input has no 'model' attribute "
            f"(got type {type(model).__name__})."
        )
    if inner is None:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: input.model is None "
            f"(input type {type(model).__name__})."
        )
    diffusion_model = getattr(inner, "diffusion_model", _ATTR_MISSING)
    if diffusion_model is _ATTR_MISSING:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: 'model.model' has no "
            f"'diffusion_model' attribute (got type {type(inner).__name__})."
        )
    if diffusion_model is None:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: 'model.model.diffusion_model' "
            f"is None (model.model type {type(inner).__name__})."
        )
    return diffusion_model


def _apply_rope_freqs_float32_cast(diffusion_model):
    """Cast every nested module's ``rope.freqs`` parameter data to ``float32``
    when it is not already in float32. Idempotency is per-tensor by dtype
    check, NOT a per-instance sentinel attribute — a sentinel would survive
    Comfy's dynamic model unload/reload cycle while ``rope.freqs`` itself
    is restored from the archived dtype, leaving RoPE running in fp16/bf16
    on subsequent calls. The dtype check makes the cast self-correcting
    against weight-restore lifecycle events. Iteration cost is one walk of
    the diffusion-model module tree per ``execute()`` call (microseconds).
    """
    for module in diffusion_model.modules():
        if hasattr(module, 'rope') and hasattr(module.rope, 'freqs'):
            if module.rope.freqs.data.dtype != torch.float32:
                module.rope.freqs.data = module.rope.freqs.data.to(torch.float32)


def clear_vae_memory(vae_model):
    for module in vae_model.modules():
        if hasattr(module, "memory"):
            module.memory = None
    gc.collect()
    comfy.model_management.soft_empty_cache()

def expand_dims(tensor, ndim):
    shape = tensor.shape + (1,) * (ndim - tensor.ndim)
    return tensor.reshape(shape)

def get_conditions(latent, latent_blur):
    t, h, w, c = latent.shape
    cond = torch.ones([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
    cond[:, ..., :-1] = latent_blur[:]
    cond[:, ..., -1:] = 1.0
    return cond

def timestep_transform(timesteps, latents_shapes):
    vt = 4
    vs = 8
    frames = (latents_shapes[:, 0] - 1) * vt + 1
    heights = latents_shapes[:, 1] * vs
    widths = latents_shapes[:, 2] * vs

    # Compute shift factor.
    def get_lin_function(x1, y1, x2, y2):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    img_shift_fn = get_lin_function(x1=256 * 256, y1=1.0, x2=1024 * 1024, y2=3.2)
    vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
    shift = torch.where(
        frames > 1,
        vid_shift_fn(heights * widths * frames),
        img_shift_fn(heights * widths),
    ).to(timesteps.device)

    # Shift timesteps.
    T = 1000.0
    timesteps = timesteps / T
    timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
    timesteps = timesteps * T
    return timesteps

def inter(x_0, x_T, t):
    t = expand_dims(t, x_0.ndim)
    T = 1000.0
    B = lambda t: t / T
    A = lambda t: 1 - (t / T)
    return A(t) * x_0 + B(t) * x_T
def area_resize(image, max_area):

    height, width = image.shape[-2:]
    scale = math.sqrt(max_area / (height * width))

    resized_height, resized_width = round(height * scale), round(width * scale)

    return TVF.resize(
        image,
        size=(resized_height, resized_width),
        interpolation=InterpolationMode.BICUBIC,
    )

def div_pad(image, factor):

    height_factor, width_factor = factor
    height, width = image.shape[-2:]

    pad_height = (height_factor - (height % height_factor)) % height_factor
    pad_width = (width_factor - (width % width_factor)) % width_factor

    if pad_height == 0 and pad_width == 0:
        return image

    if isinstance(image, torch.Tensor):
        padding = (0, pad_width, 0, pad_height)
        image = torch.nn.functional.pad(image, padding, mode='constant', value=0.0)

    return image

def cut_videos(videos):
    t = videos.size(1)
    if t == 1:
        return videos
    if t <= 4 :
        padding = [videos[:, -1].unsqueeze(1)] * (4 - t + 1)
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        return videos
    if (t - 1) % (4) == 0:
        return videos
    else:
        padding = [videos[:, -1].unsqueeze(1)] * (
            4 - ((t - 1) % (4))
        )
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        assert (videos.size(1) - 1) % (4) == 0
        return videos

def side_resize(image, size):
    antialias = not (isinstance(image, torch.Tensor) and image.device.type == 'mps')
    resized = TVF.resize(image, size, InterpolationMode.BICUBIC, antialias=antialias)
    return resized


def _seedvr2_input_shorter_edge(images, node_name):
    if images.dim() == 4:
        return min(images.shape[1], images.shape[2])
    if images.dim() == 5:
        return min(images.shape[2], images.shape[3])
    raise ValueError(
        f"{node_name}: expected 4-D or 5-D IMAGE tensor, "
        f"got shape {tuple(images.shape)}"
    )


def _seedvr2_resize_and_pad(images, upscaled_shorter_edge, node_name):
    if upscaled_shorter_edge < 2:
        raise ValueError(
            f"{node_name}: resolved upscaled_shorter_edge must be at least 2 pixels; "
            f"got {upscaled_shorter_edge}."
        )
    original_image = images
    if images.dim() == 4:
        # Comfy video components arrive as a 4-D IMAGE frame sequence:
        # (frames, H, W, C). SeedVR2 consumes that as one video.
        images = images.unsqueeze(0)
    elif images.dim() != 5:
        raise ValueError(
            f"{node_name}: expected 4-D or 5-D IMAGE tensor, "
            f"got shape {tuple(images.shape)}"
        )
    images = images.permute(0, 1, 4, 2, 3)

    b, t, c, h, w = images.shape
    images = images.reshape(b * t, c, h, w)

    clip = Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
    images = side_resize(images, upscaled_shorter_edge)

    images = clip(images)
    images = div_pad(images, (16, 16))
    _, _, new_h, new_w = images.shape

    images = images.reshape(b, t, c, new_h, new_w)
    images = cut_videos(images)
    images_bthwc = rearrange(images, "b t c h w -> b t h w c")

    return io.NodeOutput(images_bthwc, original_image, upscaled_shorter_edge)


class SeedVR2Resize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2Resize",
            category="image/video",
            inputs=[
                io.Image.Input("images"),
                io.Float.Input("multiplier", default=4.0, min=0.01),
            ],
            outputs=[
                io.Image.Output("input_pixels"),
                io.Image.Output("original_image"),
                io.Int.Output("upscaled_shorter_edge"),
            ]
        )

    @classmethod
    def execute(cls, images, multiplier=4.0):
        if multiplier <= 0:
            raise ValueError(
                f"SeedVR2Resize: multiplier must be > 0; got {multiplier}."
            )
        shorter_edge = _seedvr2_input_shorter_edge(images, "SeedVR2Resize")
        upscaled_shorter_edge = int(round(shorter_edge * multiplier))
        if upscaled_shorter_edge < 2:
            raise ValueError(
                "SeedVR2Resize: multiplier resolved upscaled_shorter_edge "
                f"to {upscaled_shorter_edge}; use a multiplier that resolves "
                "to at least 2 pixels."
            )
        return _seedvr2_resize_and_pad(
            images, upscaled_shorter_edge, "SeedVR2Resize",
        )


class SeedVR2ResizeAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2ResizeAdvanced",
            category="image/video",
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("shorter_edge", default=1280, min=2),
            ],
            outputs=[
                io.Image.Output("input_pixels"),
                io.Image.Output("original_image"),
                io.Int.Output("upscaled_shorter_edge"),
            ]
        )

    @classmethod
    def execute(cls, images, shorter_edge):
        return _seedvr2_resize_and_pad(
            images, shorter_edge, "SeedVR2ResizeAdvanced",
        )


class SeedVR2PostProcessing(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2PostProcessing",
            category="image/video",
            inputs=[
                io.Image.Input("decoded"),
                io.Image.Input("original_image"),
                io.Int.Input("upscaled_shorter_edge", min=2, force_input=True),
                io.Combo.Input("color_correction_method", options=["lab", "wavelet", "adain", "none"], default="lab"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, decoded, original_image, upscaled_shorter_edge, color_correction_method):
        cls._validate_upscaled_shorter_edge(upscaled_shorter_edge)
        decoded_5d, decoded_was_4d = cls._as_bthwc(decoded)
        original_5d, _ = cls._as_bthwc(original_image)
        decoded_5d = cls._restore_reference_batch_time(decoded_5d, original_5d)

        b = min(decoded_5d.shape[0], original_5d.shape[0])
        t = min(decoded_5d.shape[1], original_5d.shape[1])
        reference_h, reference_w = cls._resized_shorter_edge_dims(
            original_5d.shape[2], original_5d.shape[3], upscaled_shorter_edge,
        )

        decoded_5d = decoded_5d[:b, :t, :, :, :]
        target_h = min(decoded_5d.shape[2], reference_h)
        target_w = min(decoded_5d.shape[3], reference_w)
        decoded_5d = decoded_5d[:, :, :target_h, :target_w, :]
        if color_correction_method in ("lab", "wavelet", "adain"):
            reference_5d = cls._resize_original_reference(original_image, upscaled_shorter_edge)
            reference_5d = reference_5d[:b, :t, :, :, :]
            reference_5d = cls._resize_reference(reference_5d, target_h, target_w)
            output_device = decoded_5d.device
            decoded_raw = cls._to_seedvr2_raw(decoded_5d)
            reference_raw = cls._to_seedvr2_raw(reference_5d)
            decoded_flat = rearrange(decoded_raw, "b t h w c -> (b t) c h w")
            reference_flat = rearrange(reference_raw, "b t h w c -> (b t) c h w")
            output = cls._color_transfer_chunked(
                decoded_flat, reference_flat, output_device, color_correction_method,
            )
            output = rearrange(output, "(b t) c h w -> b t h w c", b=b, t=t)
            output = output.add(1.0).div(2.0).clamp(0.0, 1.0)
        elif color_correction_method == "none":
            output = decoded_5d
        else:
            raise ValueError(f"SeedVR2PostProcessing: unknown color_correction_method {color_correction_method!r}")

        h2 = output.shape[-3] - (output.shape[-3] % 2)
        w2 = output.shape[-2] - (output.shape[-2] % 2)
        output = output[:, :, :h2, :w2, :]
        if decoded_was_4d:
            output = output.reshape(-1, output.shape[-3], output.shape[-2], output.shape[-1])
        return io.NodeOutput(output)

    @staticmethod
    def _as_bthwc(images):
        if images.ndim == 4:
            return images.unsqueeze(0), True
        if images.ndim == 5:
            return images, False
        raise ValueError(
            f"SeedVR2PostProcessing: expected 4-D or 5-D IMAGE tensor, got shape {tuple(images.shape)}"
        )

    @staticmethod
    def _restore_reference_batch_time(decoded, reference):
        if decoded.shape[0] != 1:
            return decoded
        ref_b, ref_t = reference.shape[:2]
        if ref_b < 1 or decoded.shape[1] % ref_b != 0:
            return decoded
        decoded_t = decoded.shape[1] // ref_b
        if decoded_t < ref_t:
            return decoded
        return decoded.reshape(ref_b, decoded_t, decoded.shape[2], decoded.shape[3], decoded.shape[4])

    @staticmethod
    def _to_seedvr2_raw(images):
        return images.mul(2.0).sub(1.0)

    @staticmethod
    def _validate_upscaled_shorter_edge(upscaled_shorter_edge):
        if not isinstance(upscaled_shorter_edge, int) or upscaled_shorter_edge < 2:
            raise ValueError(
                "SeedVR2PostProcessing: upscaled_shorter_edge must be an integer "
                f"of at least 2 pixels; got {upscaled_shorter_edge!r}."
            )

    @staticmethod
    def _resized_shorter_edge_dims(height, width, upscaled_shorter_edge):
        if height <= width:
            return upscaled_shorter_edge, int(upscaled_shorter_edge * width / height)
        return int(upscaled_shorter_edge * height / width), upscaled_shorter_edge

    @classmethod
    def _resize_original_reference(cls, original, upscaled_shorter_edge):
        original_5d, _ = cls._as_bthwc(original)
        b, t = original_5d.shape[:2]
        original_flat = rearrange(original_5d, "b t h w c -> (b t) c h w")
        resized_flat = side_resize(original_flat, upscaled_shorter_edge).clamp(0.0, 1.0)
        return rearrange(resized_flat, "(b t) c h w -> b t h w c", b=b, t=t)

    @staticmethod
    def _color_transfer_on_vae_device(decoded_flat, reference_flat, output_device, transfer_fn):
        color_device = comfy.model_management.vae_device()
        decoded_flat = decoded_flat.to(device=color_device)
        reference_flat = reference_flat.to(device=color_device)
        output = transfer_fn(decoded_flat, reference_flat)
        return output.to(device=output_device)

    @staticmethod
    def _lab_color_transfer_on_vae_device(decoded_flat, reference_flat, output_device):
        color_device = comfy.model_management.vae_device()
        result = None
        for start in range(decoded_flat.shape[0]):
            decoded_frame = decoded_flat[start:start + 1].to(device=color_device).clone()
            reference_frame = reference_flat[start:start + 1].to(device=color_device).clone()
            output = lab_color_transfer(decoded_frame, reference_frame).to(device=output_device)
            if result is None:
                result = torch.empty(
                    (decoded_flat.shape[0],) + tuple(output.shape[1:]),
                    device=output_device,
                    dtype=output.dtype,
                )
            result[start:start + 1].copy_(output)
        if result is None:
            raise ValueError("SeedVR2PostProcessing: LAB color correction requires at least one frame.")
        return result

    @classmethod
    def _color_transfer_chunked(cls, decoded_flat, reference_flat, output_device, color_correction_method):
        chunk_size = cls._estimate_color_correction_chunk_size(decoded_flat, color_correction_method)
        while True:
            next_chunk_size = None
            try:
                return cls._run_color_transfer_chunks(
                    decoded_flat, reference_flat, output_device, color_correction_method, chunk_size,
                )
            except Exception as e:
                comfy.model_management.raise_non_oom(e)
                if chunk_size <= 1:
                    raise RuntimeError(
                        "SeedVR2PostProcessing: color correction OOM at one frame; "
                        f"color_correction_method={color_correction_method}, shape={tuple(decoded_flat.shape)}."
                    ) from e
                next_chunk_size = max(1, chunk_size // 2)

            comfy.model_management.soft_empty_cache()
            chunk_size = next_chunk_size

    @classmethod
    def _run_color_transfer_chunks(cls, decoded_flat, reference_flat, output_device, color_correction_method, chunk_size):
        result = None
        for start in range(0, decoded_flat.shape[0], chunk_size):
            end = min(start + chunk_size, decoded_flat.shape[0])
            decoded_chunk = decoded_flat[start:end]
            reference_chunk = reference_flat[start:end]
            if color_correction_method == "lab":
                output = cls._lab_color_transfer_on_vae_device(decoded_chunk, reference_chunk, output_device)
            elif color_correction_method == "wavelet":
                output = cls._color_transfer_on_vae_device(
                    decoded_chunk, reference_chunk, output_device, wavelet_color_transfer,
                )
            else:
                output = cls._color_transfer_on_vae_device(
                    decoded_chunk, reference_chunk, output_device, adain_color_transfer,
                )
            if result is None:
                result = torch.empty(
                    (decoded_flat.shape[0],) + tuple(output.shape[1:]),
                    device=output_device,
                    dtype=output.dtype,
                )
            result[start:end].copy_(output)
        if result is None:
            raise ValueError("SeedVR2PostProcessing: color correction requires at least one frame.")
        return result

    @classmethod
    def _estimate_color_correction_chunk_size(cls, decoded_flat, color_correction_method):
        multiplier = cls._color_correction_memory_multiplier(color_correction_method)
        frames = decoded_flat.shape[0]
        _, channels, height, width = decoded_flat.shape
        dtype_bytes = max(decoded_flat.element_size(), 4)
        bytes_per_frame = height * width * channels * dtype_bytes * multiplier
        if bytes_per_frame <= 0:
            return frames
        color_device = comfy.model_management.vae_device()
        free_memory = comfy.model_management.get_free_memory(color_device)
        chunk_size = int((free_memory * COLOR_CORRECTION_MEMORY_HEADROOM) // bytes_per_frame)
        return max(1, min(frames, chunk_size))

    @staticmethod
    def _color_correction_memory_multiplier(color_correction_method):
        if color_correction_method == "lab":
            return LAB_SCALE_MULTIPLIER
        if color_correction_method == "wavelet":
            return WAVELET_SCALE_MULTIPLIER
        if color_correction_method == "adain":
            return ADAIN_SCALE_MULTIPLIER
        raise ValueError(f"SeedVR2PostProcessing: unknown color_correction_method {color_correction_method!r}")

    @staticmethod
    def _resize_reference(reference, height, width):
        if reference.shape[2] == height and reference.shape[3] == width:
            return reference
        b, t = reference.shape[:2]
        reference_flat = rearrange(reference, "b t h w c -> (b t) c h w")
        resized = TVF.resize(
            reference_flat,
            size=(height, width),
            interpolation=InterpolationMode.BICUBIC,
            antialias=not (isinstance(reference_flat, torch.Tensor) and reference_flat.device.type == "mps"),
        )
        return rearrange(resized, "(b t) c h w -> b t h w c", b=b, t=t)


class SeedVR2Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2Conditioning",
            category="image/video",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("vae_conditioning", display_name="LATENT"),
            ],
            outputs=[
                io.Model.Output(display_name = "model"),
                io.Conditioning.Output(display_name = "positive"),
                io.Conditioning.Output(display_name = "negative"),
                io.Latent.Output(display_name = "latent"),
            ],
        )

    @classmethod
    def execute(cls, model, vae_conditioning) -> io.NodeOutput:

        vae_conditioning = vae_conditioning["samples"]
        if vae_conditioning.ndim != 5:
            raise ValueError(
                "SeedVR2Conditioning expects a 5-D VAE latent in Comfy "
                f"channel-first layout; got shape {tuple(vae_conditioning.shape)}."
            )
        if vae_conditioning.shape[-1] == _SEEDVR2_LATENT_CHANNELS and vae_conditioning.shape[1] != _SEEDVR2_LATENT_CHANNELS:
            raise ValueError(
                "SeedVR2Conditioning expects SeedVR2 VAE latents in Comfy "
                f"channel-first layout (B, {_SEEDVR2_LATENT_CHANNELS}, T, H, W); "
                f"got channel-last shape {tuple(vae_conditioning.shape)}."
            )
        vae_conditioning = vae_conditioning.movedim(1, -1).contiguous()
        model_patcher = model
        model = _resolve_seedvr2_diffusion_model(model_patcher)
        pos_cond = model.positive_conditioning
        neg_cond = model.negative_conditioning

        # Fail-loud guard against silently-wrong output when a numz-format
        # DiT-only ``.safetensors`` (no ``positive_conditioning`` /
        # ``negative_conditioning`` keys) is loaded via ``UNETLoader``.
        # ``NaDiT.__init__`` zero-fills the buffers via ``torch.zeros`` (see
        # ``comfy/ldm/seedvr/model.py``); ``load_state_dict(strict=False)``
        # leaves them at zero when the keys are absent. Detect that state
        # here rather than at ``BaseModel.extra_conds`` (per sampling step,
        # wasteful) or at the resolver helper (mixes structural shape with
        # semantic content). Both buffers must be checked together — partial
        # bake regressions could populate one but not the other.
        if (
            pos_cond.float().abs().sum().item() == 0
            and neg_cond.float().abs().sum().item() == 0
        ):
            raise RuntimeError(
                f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: positive_conditioning "
                f"and negative_conditioning buffers are zero-valued — model "
                f"file appears to be a numz-format DiT-only export missing "
                f"the SeedVR2 conditioning tensors. "
                f"Re-bake the file with ``positive_conditioning`` (58, 5120) "
                f"and ``negative_conditioning`` (64, 5120) keys at top level, "
                f"or load via CheckpointLoaderSimple from a bundled "
                f"checkpoint."
            )

        _apply_rope_freqs_float32_cast(model)

        condition = torch.stack([get_conditions(c, c) for c in vae_conditioning])
        condition = condition.movedim(-1, 1)
        latent = vae_conditioning.movedim(-1, 1)

        latent = rearrange(latent, "b c t h w -> b (c t) h w")
        condition = rearrange(condition, "b c t h w -> b (c t) h w")

        negative = [[neg_cond.unsqueeze(0), {"condition": condition}]]
        positive = [[pos_cond.unsqueeze(0), {"condition": condition}]]

        return io.NodeOutput(model_patcher, positive, negative, {"samples": latent})

# SeedVR2 latent / conditioning channel constants. The SeedVR2 conditioning
# stage collapses ``(B, C, T, H, W) -> (B, C*T, H, W)`` for both the latent
# (C=16) and the per-frame condition tensor (C=17 = 16 latent + 1 mask), as
# required by ``NaDiT.forward`` which un-collapses via
# ``view(B, 16, -1, H, W)`` and ``view(B, 17, -1, H, W)`` respectively.
_SEEDVR2_LATENT_CHANNELS = 16
_SEEDVR2_CONDITION_CHANNELS = 17


def _slice_collapsed_4d_along_t(tensor_4d: torch.Tensor, t_start: int,
                                 t_end: int, channels: int) -> torch.Tensor:
    """Slice a SeedVR2-style collapsed 4D tensor ``(B, channels*T, H, W)``
    along the latent T axis, returning ``(B, channels*(t_end - t_start), H, W)``.

    Reshape -> slice -> ``.contiguous()`` -> re-collapse. ``reshape`` is
    used for the un-collapse so non-contiguous incoming tensors from
    cropping or slicing nodes are accepted. The
    ``.contiguous()`` is mandatory: T-axis slicing of a 5D tensor produces a
    non-contiguous view, and the subsequent re-collapse requires contiguous
    storage.
    """
    B, CT, H, W = tensor_4d.shape
    if CT % channels != 0:
        raise ValueError(
            f"_slice_collapsed_4d_along_t: collapsed channel dim {CT} is not "
            f"divisible by channels={channels}; tensor shape {tuple(tensor_4d.shape)}."
        )
    T = CT // channels
    if not (0 <= t_start < t_end <= T):
        raise ValueError(
            f"_slice_collapsed_4d_along_t: slice [{t_start}:{t_end}] out of "
            f"range for T={T}."
        )
    new_T = t_end - t_start
    sliced = tensor_4d.reshape(B, channels, T, H, W)[:, :, t_start:t_end, :, :].contiguous()
    return sliced.reshape(B, channels * new_T, H, W)


def _slice_seedvr2_cond_along_t(cond_list, t_start: int, t_end: int):
    """Build a new SeedVR2 conditioning list with the per-frame ``condition``
    tensor sliced along the latent T axis.

    SeedVR2 conditioning entries have the shape
    ``[text_cond_tensor, options_dict]`` where ``options_dict["condition"]``
    is a 4D collapsed ``(B, 17*T, H, W)`` tensor; the text tensor itself has
    no temporal axis and is passed through unchanged. Other keys in the
    options dict (controlnets, etc.) are also passed through unchanged. If
    an entry has no ``"condition"`` key, the entry is forwarded verbatim.

    A new list of ``[text_cond, new_options_dict]`` pairs is returned; the
    original ``cond_list`` and its options dicts are not mutated.
    """
    new_list = []
    for entry in cond_list:
        text_cond, options = entry[0], entry[1]
        if "condition" not in options:
            new_list.append(entry)
            continue
        new_options = options.copy()
        new_options["condition"] = _slice_collapsed_4d_along_t(
            new_options["condition"], t_start, t_end,
            _SEEDVR2_CONDITION_CHANNELS,
        )
        new_list.append([text_cond, new_options])
    return new_list


def _slice_seedvr2_noise_mask_along_t(noise_mask: torch.Tensor,
                                      samples_4d: torch.Tensor,
                                      t_start: int,
                                      t_end: int):
    """Slice collapsed SeedVR2 masks and preserve standard masks.

    ``SetLatentNoiseMask`` produces ``(B, 1, H, W)`` masks that KSampler
    expands to the latent shape. Only masks already expanded to the full
    collapsed ``(B, 16*T, H, W)`` shape need temporal slicing here.
    """
    if noise_mask.ndim == samples_4d.ndim and noise_mask.shape[1] == samples_4d.shape[1]:
        return _slice_collapsed_4d_along_t(
            noise_mask, t_start, t_end, _SEEDVR2_LATENT_CHANNELS,
        )
    return noise_mask


def _concat_chunks_along_t(chunks_4d, channels: int) -> torch.Tensor:
    """Concatenate a list of SeedVR2-style collapsed 4D tensors
    ``(B, channels*T_i, H, W)`` along the latent T axis. Each chunk is
    un-collapsed to 5D, concatenated on ``dim=2``, then re-collapsed to 4D.
    """
    if len(chunks_4d) == 0:
        raise ValueError("_concat_chunks_along_t: empty chunk list.")
    fives = []
    for ch in chunks_4d:
        B, CT, H, W = ch.shape
        if CT % channels != 0:
            raise ValueError(
                f"_concat_chunks_along_t: chunk shape {tuple(ch.shape)} "
                f"channel dim {CT} not divisible by channels={channels}."
            )
        T = CT // channels
        fives.append(ch.reshape(B, channels, T, H, W))
    cat = torch.cat(fives, dim=2).contiguous()
    B, C, T_total, H, W = cat.shape
    return cat.reshape(B, C * T_total, H, W)


def _hann_blend_weights_1d(overlap: int, device, dtype) -> torch.Tensor:
    """Build a 1D crossfade weight tensor of length ``overlap`` for the
    *previous* chunk's contribution; the current chunk's weight is
    ``1 - w_prev``.

    Mirrors the numz ``blend_overlapping_frames`` shape
    (AInVFX/numz fork ``src/core/generation_utils.py``,
    ``blend_overlapping_frames``): a Hann window with a ``[1/3, 2/3]``
    dead-band when ``overlap >= 3``, and a plain linear ramp when
    ``overlap < 3`` (the dead-band would collapse the transition for
    very small overlap counts). The numz reference operates on
    pixel-space tensors ``[overlap, H, W, C]``; this 1D form is
    reshaped by the caller to broadcast across the latent's
    ``(B, C, T_overlap, H, W)`` axes.
    """
    if overlap < 1:
        raise ValueError(
            f"_hann_blend_weights_1d: overlap must be >= 1; got {overlap}."
        )
    if overlap >= 3:
        t = torch.linspace(0.0, 1.0, steps=overlap, device=device, dtype=dtype)
        blend_start = 1.0 / 3.0
        blend_end = 2.0 / 3.0
        u = ((t - blend_start) / (blend_end - blend_start)).clamp(0.0, 1.0)
        return 0.5 + 0.5 * torch.cos(torch.pi * u)
    return torch.linspace(1.0, 0.0, steps=overlap, device=device, dtype=dtype)


def _blend_overlap_region(prev_tail_5d: torch.Tensor,
                          cur_head_5d: torch.Tensor) -> torch.Tensor:
    """Blend two 5D ``(B, C, T_overlap, H, W)`` tensors of equal shape
    using a 1D Hann/linear ramp along the T axis. ``prev_tail_5d``
    receives the descending weight; ``cur_head_5d`` receives
    ``1 - w_prev``.

    The caller is responsible for ensuring both inputs have identical
    shape and dtype/device.
    """
    if prev_tail_5d.shape != cur_head_5d.shape:
        raise ValueError(
            f"_blend_overlap_region: shape mismatch "
            f"prev {tuple(prev_tail_5d.shape)} vs "
            f"cur {tuple(cur_head_5d.shape)}."
        )
    overlap = int(prev_tail_5d.shape[2])
    w_prev_1d = _hann_blend_weights_1d(
        overlap, prev_tail_5d.device, prev_tail_5d.dtype,
    )
    # Reshape to (1, 1, overlap, 1, 1) for broadcast across B, C, H, W.
    w_prev = w_prev_1d.view(1, 1, overlap, 1, 1)
    w_cur = 1.0 - w_prev
    return prev_tail_5d * w_prev + cur_head_5d * w_cur


def _concat_chunks_with_overlap_blend(chunk_specs, channels: int,
                                      overlap_latent: int) -> torch.Tensor:
    """Concatenate temporally-overlapping chunks back into a single
    collapsed 4D tensor, blending overlap regions with a Hann/linear
    crossfade.

    ``chunk_specs`` is a list of ``(t_start, t_end, chunk_4d)`` tuples
    in source-latent T coordinates. ``overlap_latent == 0`` is a fast
    path that delegates to plain concatenation (and produces output
    bit-identical to ``_concat_chunks_along_t`` of the same chunks).

    The blend at each pair of adjacent chunks acts on the actual
    overlap region width ``min(prev_end - cur_start, current chunk
    length)``, which may be smaller than ``overlap_latent`` when the
    final chunk is a runt shorter than the configured overlap.
    """
    if len(chunk_specs) == 0:
        raise ValueError("_concat_chunks_with_overlap_blend: empty chunk list.")
    if overlap_latent < 0:
        raise ValueError(
            f"_concat_chunks_with_overlap_blend: overlap_latent must be "
            f">= 0; got {overlap_latent}."
        )

    # Validate channel divisibility once and capture per-chunk T.
    chunk_5d = []
    for t_start, t_end, ch in chunk_specs:
        B, CT, H, W = ch.shape
        if CT % channels != 0:
            raise ValueError(
                f"_concat_chunks_with_overlap_blend: chunk shape "
                f"{tuple(ch.shape)} channel dim {CT} not divisible "
                f"by channels={channels}."
            )
        T = CT // channels
        if t_end - t_start != T:
            raise ValueError(
                f"_concat_chunks_with_overlap_blend: chunk T={T} mismatches "
                f"declared range [{t_start}:{t_end}]."
            )
        chunk_5d.append((t_start, t_end, ch.reshape(B, channels, T, H, W)))

    if overlap_latent == 0:
        # Fast path: pure concat in the caller-provided chunk order.
        return _concat_chunks_along_t(
            [c.reshape(c.shape[0], channels * c.shape[2], c.shape[3], c.shape[4])
             for _, _, c in chunk_5d],
            channels,
        )

    T_total = max(t_end for _, t_end, _ in chunk_5d)
    first_5d = chunk_5d[0][2]
    B = first_5d.shape[0]
    H = first_5d.shape[3]
    W = first_5d.shape[4]
    result = torch.empty(
        (B, channels, T_total, H, W),
        device=first_5d.device, dtype=first_5d.dtype,
    )
    filled_until = 0
    for i, (cs, ce, ct_5d) in enumerate(chunk_5d):
        chunk_T = int(ct_5d.shape[2])
        if i == 0:
            result[:, :, cs:ce, :, :] = ct_5d
            filled_until = ce
            continue
        # Overlap region width is bounded by both the previous fill
        # frontier and the current chunk's actual length (for runt
        # final chunks shorter than the configured overlap).
        overlap_len = min(filled_until - cs, chunk_T)
        if overlap_len > 0:
            prev_tail = result[:, :, cs:cs + overlap_len, :, :].contiguous()
            cur_head = ct_5d[:, :, :overlap_len, :, :].contiguous()
            blended = _blend_overlap_region(prev_tail, cur_head)
            result[:, :, cs:cs + overlap_len, :, :] = blended
            tail_start = cs + overlap_len
            tail_end = ce
            if tail_end > tail_start:
                result[:, :, tail_start:tail_end, :, :] = (
                    ct_5d[:, :, overlap_len:, :, :]
                )
        else:
            # Disjoint chunks (overlap_latent set but this pair did not
            # actually overlap, e.g. step_latent equal to chunk_latent
            # in a degenerate config). Treat as concat.
            result[:, :, cs:ce, :, :] = ct_5d
        filled_until = ce

    return result.contiguous().reshape(B, channels * T_total, H, W)


def _run_standard_sample(model, seed: int, steps: int, cfg: float,
                         sampler_name: str, scheduler: str,
                         positive, negative, latent_image: dict,
                         denoise: float) -> dict:
    """Single-shot delegation that mirrors the standard ``common_ksampler``
    flow (``nodes.py:common_ksampler``): generate noise from seed, run
    ``comfy.sample.sample``, return a latent dict. Used by the
    ProgressiveSampler short-circuit when the full sequence fits in one
    chunk so chunking introduces no overhead for small videos.
    """
    samples_in = latent_image["samples"]
    samples_in = comfy.sample.fix_empty_latent_channels(
        model, samples_in, latent_image.get("downscale_ratio_spacial", None),
    )
    batch_inds = latent_image.get("batch_index", None)
    noise = comfy.sample.prepare_noise(samples_in, seed, batch_inds)
    noise_mask = latent_image.get("noise_mask", None)
    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, samples_in,
        denoise=denoise, noise_mask=noise_mask, seed=seed,
    )
    out = latent_image.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return out


class SeedVR2ProgressiveSampler(io.ComfyNode):
    """Sequential temporal chunking sampler for SeedVR2 native.

    Drop-in replacement for ``KSampler`` in SeedVR2 native workflows that
    OOM on long sequences. The latent enters the sampler in SeedVR2's
    collapsed form ``(B, 16*T, H, W)`` (collapsed by ``SeedVR2Conditioning``
    at ``rearrange(b c t h w -> b (c t) h w)``); this node slices that
    tensor along the temporal axis, runs the configured inner sampler
    sequentially per chunk against the standard ``comfy.sample.sample``
    entry point, and concatenates per-chunk outputs back into a single
    ``(B, 16*T_total, H, W)`` latent.

    ``frames_per_chunk`` is expressed in pixel-frame units to match the
    SeedVR2 4n+1 constraint enforced upstream by ``cut_videos`` and the
    VAE's ``temporal_downsample_factor=4``. A pixel chunk size ``F``
    maps to ``(F - 1) // 4 + 1`` latent-frame chunks.

    Determinism contract: a single noise tensor is generated once from
    the user seed and sliced per chunk (rather than re-seeding each
    chunk), so a workflow that fits in a single chunk produces output
    identical to a workflow that fits in N chunks at the same seed,
    modulo the inherent T-axis chunk-boundary independence of the model.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2ProgressiveSampler",
            category="sampling",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("seed", default=0, min=0,
                             max=0xffffffffffffffff,
                             control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0,
                               step=0.1, round=0.01),
                io.Combo.Input("sampler_name",
                               options=comfy.samplers.SAMPLER_NAMES),
                io.Combo.Input("scheduler",
                               options=comfy.samplers.SCHEDULER_NAMES),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0,
                               step=0.01),
                io.Int.Input("frames_per_chunk", default=21, min=1,
                             max=16384, step=4),
                io.Int.Input("temporal_overlap", default=0, min=0,
                             max=16384,
                             tooltip="Latent-frame overlap between "
                                     "adjacent chunks; blended with a "
                                     "Hann window (linear for overlap "
                                     "< 3). 0 = no blend, pure concat. "
                                     "Values >= the chunk's latent-frame "
                                     "length use the maximum valid "
                                     "overlap; 1 latent frame corresponds "
                                     "to ~4 pixel frames."),
                io.Combo.Input("chunking_mode",
                               options=["manual", "auto"],
                               default="manual",
                               tooltip="manual = use frames_per_chunk "
                                       "exactly; auto = retry only real OOM "
                                       "failures with progressively smaller "
                                       "temporal chunks."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image, denoise,
                frames_per_chunk, temporal_overlap,
                chunking_mode="manual") -> io.NodeOutput:
        # 4n+1 validation in pixel-frame domain. The SeedVR2 native pipeline
        # requires pixel-frame counts of the form 4n+1 (1, 5, 9, 13, ...),
        # imposed at ``cut_videos`` upstream and propagated through the VAE's
        # temporal_downsample_factor=4. Reject violations explicitly before
        # any model invocation; a silent rounding would mis-align chunk
        # boundaries with the 4n+1 lattice.
        if frames_per_chunk < 1 or (frames_per_chunk - 1) % 4 != 0:
            raise ValueError(
                f"SeedVR2ProgressiveSampler: frames_per_chunk must be a "
                f"4n+1 pixel-frame count (1, 5, 9, 13, 17, 21, ...); "
                f"got {frames_per_chunk}."
            )

        samples_4d = latent_image["samples"]
        samples_4d = comfy.sample.fix_empty_latent_channels(
            model, samples_4d,
            latent_image.get("downscale_ratio_spacial", None),
        )
        if samples_4d.ndim != 4:
            raise ValueError(
                f"SeedVR2ProgressiveSampler: expected 4D collapsed latent "
                f"(B, 16*T, H, W); got shape {tuple(samples_4d.shape)}."
            )
        B, CT, H, W = samples_4d.shape
        if CT % _SEEDVR2_LATENT_CHANNELS != 0:
            raise ValueError(
                f"SeedVR2ProgressiveSampler: collapsed channel dim {CT} is "
                f"not divisible by SeedVR2 latent channels "
                f"{_SEEDVR2_LATENT_CHANNELS}; latent does not appear to be "
                f"SeedVR2-shaped."
            )
        T_latent = CT // _SEEDVR2_LATENT_CHANNELS
        T_pixel = 4 * (T_latent - 1) + 1

        if chunking_mode not in ("manual", "auto"):
            raise ValueError(
                f"SeedVR2ProgressiveSampler: chunking_mode must be "
                f"'manual' or 'auto'; got {chunking_mode!r}."
            )

        if chunking_mode == "auto":
            attempts = _seedvr2_auto_chunk_attempts(
                T_latent, T_pixel, frames_per_chunk,
            )
            for i, attempt_frames_per_chunk in enumerate(attempts):
                retry = False
                try:
                    return cls.execute(
                        model=model, seed=seed, steps=steps, cfg=cfg,
                        sampler_name=sampler_name, scheduler=scheduler,
                        positive=positive, negative=negative,
                        latent_image=latent_image, denoise=denoise,
                        frames_per_chunk=attempt_frames_per_chunk,
                        temporal_overlap=temporal_overlap,
                        chunking_mode="manual",
                    )
                except Exception as e:
                    comfy.model_management.raise_non_oom(e)
                    if i == len(attempts) - 1:
                        raise RuntimeError(
                            "SeedVR2ProgressiveSampler: exhausted auto "
                            "chunking attempts after OOM. Tried "
                            f"frames_per_chunk values {attempts}."
                        ) from e
                    retry = True

                if retry:
                    logging.warning(
                        "SeedVR2ProgressiveSampler auto chunking OOM at "
                        "frames_per_chunk=%s; retrying with "
                        "frames_per_chunk=%s.",
                        attempt_frames_per_chunk, attempts[i + 1],
                    )
                    comfy.model_management.soft_empty_cache()

        # Short-circuit: total fits in one chunk -> standard path with no
        # chunking overhead. Output of this branch is byte-identical to the
        # built-in KSampler given the same (model, seed, steps, cfg,
        # sampler_name, scheduler, positive, negative, latent_image,
        # denoise) tuple.
        if T_pixel <= frames_per_chunk:
            return io.NodeOutput(_run_standard_sample(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image, denoise,
            ))

        # Map pixel chunk -> latent chunk. Each chunk's latent length is
        # at most ``chunk_latent``; the final chunk may be a runt that
        # is automatically 4n+1-aligned in the pixel domain by the
        # T_pixel = 4*(T_latent-1) + 1 mapping (every positive integer
        # T_latent corresponds to a valid 4n+1 pixel count).
        chunk_latent = (frames_per_chunk - 1) // 4 + 1

        # ``temporal_overlap`` is exposed in latent-frame units, but users
        # do not know the derived latent chunk length. Treat oversized
        # values as "maximum valid overlap" while preserving a strictly
        # positive chunk-loop stride.
        if temporal_overlap < 0:
            raise ValueError(
                f"SeedVR2ProgressiveSampler: temporal_overlap must be >= 0; "
                f"got {temporal_overlap}."
            )
        temporal_overlap = min(temporal_overlap, chunk_latent - 1)
        step_latent = chunk_latent - temporal_overlap

        # Generate full noise once from the user seed, then slice along T
        # per chunk. Using one global noise tensor (rather than re-seeding
        # per chunk) preserves seed-determinism across chunk-count
        # variations: the same (seed, total T_latent) always produces the
        # same noise samples regardless of how the work is partitioned.
        batch_inds = latent_image.get("batch_index", None)
        noise_full = comfy.sample.prepare_noise(samples_4d, seed, batch_inds)

        noise_mask = latent_image.get("noise_mask", None)

        # Build the flat list of chunk ranges first so the chunking
        # geometry is fully known before any sample call.
        chunk_ranges = []
        for chunk_start in range(0, T_latent, step_latent):
            chunk_end = min(chunk_start + chunk_latent, T_latent)
            if chunk_start >= chunk_end:
                # The final iteration of a stride that lands exactly on
                # T_latent produces a zero-length chunk; skip it.
                break
            chunk_ranges.append((chunk_start, chunk_end))
            if chunk_end >= T_latent:
                break

        def _sample_one_chunk(chunk_start, chunk_end):
            samples_chunk = _slice_collapsed_4d_along_t(
                samples_4d, chunk_start, chunk_end,
                _SEEDVR2_LATENT_CHANNELS,
            )
            noise_chunk = _slice_collapsed_4d_along_t(
                noise_full, chunk_start, chunk_end,
                _SEEDVR2_LATENT_CHANNELS,
            )
            positive_chunk = _slice_seedvr2_cond_along_t(
                positive, chunk_start, chunk_end,
            )
            negative_chunk = _slice_seedvr2_cond_along_t(
                negative, chunk_start, chunk_end,
            )

            # Per-chunk noise_mask handling: standard masks are passed
            # through for KSampler expansion; pre-expanded collapsed
            # masks are sliced.
            chunk_noise_mask = None
            if noise_mask is not None:
                chunk_noise_mask = _slice_seedvr2_noise_mask_along_t(
                    noise_mask, samples_4d, chunk_start, chunk_end,
                )

            return comfy.sample.sample(
                model, noise_chunk, steps, cfg, sampler_name, scheduler,
                positive_chunk, negative_chunk, samples_chunk,
                denoise=denoise, noise_mask=chunk_noise_mask, seed=seed,
            )

        chunk_specs = []
        for chunk_start, chunk_end in chunk_ranges:
            chunk_samples = _sample_one_chunk(chunk_start, chunk_end)
            chunk_specs.append((chunk_start, chunk_end, chunk_samples))

        final = _concat_chunks_with_overlap_blend(
            chunk_specs, _SEEDVR2_LATENT_CHANNELS, temporal_overlap,
        )

        out = latent_image.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = final
        return io.NodeOutput(out)


class SeedVRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SeedVR2Conditioning,
            SeedVR2Resize,
            SeedVR2ResizeAdvanced,
            SeedVR2PostProcessing,
            SeedVR2ProgressiveSampler,
        ]

async def comfy_entrypoint() -> SeedVRExtension:
    return SeedVRExtension()

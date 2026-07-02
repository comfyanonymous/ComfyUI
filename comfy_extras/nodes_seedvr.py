from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import torch

import comfy.model_management
from comfy.ldm.seedvr.color_fix import (
    adain_color_transfer,
    lab_color_transfer,
    wavelet_color_transfer,
)
from comfy.ldm.seedvr.constants import (
    SEEDVR2_ADAIN_SCALE_MULTIPLIER,
    SEEDVR2_COLOR_MEM_HEADROOM,
    SEEDVR2_DTYPE_BYTES_FLOOR,
    SEEDVR2_LAB_SCALE_MULTIPLIER,
    SEEDVR2_LATENT_CHANNELS,
    SEEDVR2_OOM_BACKOFF_DIVISOR,
    SEEDVR2_WAVELET_SCALE_MULTIPLIER,
)

from torchvision.transforms import functional as TVF
from torchvision.transforms import Lambda
from torchvision.transforms.functional import InterpolationMode


_SEEDVR2_INVALID_MODEL_MSG_PREFIX = (
    "SeedVR2Conditioning: model object does not match expected SeedVR2 structure"
)

# Private sentinel for getattr default: distinguishes "attribute missing"
# from "attribute present but None" so the failure message is accurate.
_ATTR_MISSING = object()


def _resolve_seedvr2_diffusion_model(model):
    """Resolve ``model.model.diffusion_model``, failing loud via the ``_ATTR_MISSING`` sentinel so each of the four modes (model/diffusion_model missing vs None) gives an accurate message."""
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


def get_conditions(latent, latent_blur):
    t, h, w, c = latent.shape
    cond = torch.ones([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
    cond[:, ..., :-1] = latent_blur[:]
    cond[:, ..., -1:] = 1.0
    return cond

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

def _seedvr2_input_shorter_edge(images, node_name):
    if images.dim() == 4:
        return min(images.shape[1], images.shape[2])
    if images.dim() == 5:
        return min(images.shape[2], images.shape[3])
    raise ValueError(
        f"{node_name}: expected 4-D or 5-D IMAGE tensor, "
        f"got shape {tuple(images.shape)}"
    )


def _seedvr2_pad(images, upscaled_shorter_edge, node_name):
    if upscaled_shorter_edge < 2:
        raise ValueError(
            f"{node_name}: input shorter edge must be at least 2 pixels; "
            f"got {upscaled_shorter_edge}."
        )
    if images.shape[-1] > 3:
        images = images[..., :3]
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
    images = clip(images)
    images = div_pad(images, (16, 16))
    _, _, new_h, new_w = images.shape

    images = images.reshape(b, t, c, new_h, new_w)
    images = cut_videos(images)
    images_bthwc = images.permute(0, 1, 3, 4, 2).contiguous()

    return io.NodeOutput(images_bthwc)


class SeedVR2Preprocess(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2Preprocess",
            display_name="Pre-Process SeedVR2 Input",
            category="image/upscaling",
            description="Pad a resized image for SeedVR2 model. Alpha channel is dropped. The node Post-Process SeedVR2 Output re-applies it from the original resized image.",
            search_aliases=["seedvr2", "upscale", "video upscale", "pad", "preprocess"],
            inputs=[
                io.Image.Input("resized_images", tooltip="The resized image to process."),
            ],
            outputs=[
                io.Image.Output("images", tooltip="The padded image for VAE encoding."),
            ]
        )

    @classmethod
    def execute(cls, resized_images):
        upscaled_shorter_edge = _seedvr2_input_shorter_edge(resized_images, "SeedVR2Preprocess")
        return _seedvr2_pad(
            resized_images, upscaled_shorter_edge, "SeedVR2Preprocess",
        )


class SeedVR2PostProcessing(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2PostProcessing",
            display_name="Post-Process SeedVR2 Output",
            category="image/upscaling",
            description="Align the generated image with the original resized image and apply color correction.",
            search_aliases=["seedvr2", "upscale", "color correction", "color match", "postprocess"],
            inputs=[
                io.Image.Input("images", tooltip="The generated image to process."),
                io.Image.Input("original_resized_images", tooltip="The original resized image before pre-processing, used as reference."),
                io.Combo.Input("color_correction_method", options=["lab", "wavelet", "adain", "none"], default="lab", tooltip="Method to match the generated image colors to the original image. lab: transfer color in CIELAB space, preserving detail (most faithful). wavelet: transfer low-frequency color, keeping upscaled high-frequency detail. adain: match per-channel mean/std (fastest, global tint). none: skip color transfer (geometry alignment only)."),
            ],
            outputs=[io.Image.Output(display_name="images", tooltip="The aligned, color-corrected image.")],
        )

    @classmethod
    def execute(cls, images, original_resized_images, color_correction_method):
        alpha_input = None
        if original_resized_images.shape[-1] == 4:
            alpha_input = original_resized_images[..., 3:4]
            original_resized_images = original_resized_images[..., :3]
        decoded_5d, decoded_was_4d = cls._as_bthwc(images)
        reference_full, _ = cls._as_bthwc(original_resized_images)
        decoded_5d = cls._restore_reference_batch_time(decoded_5d, reference_full)

        b = min(decoded_5d.shape[0], reference_full.shape[0])
        t = min(decoded_5d.shape[1], reference_full.shape[1])
        reference_h = reference_full.shape[2]
        reference_w = reference_full.shape[3]

        decoded_5d = decoded_5d[:b, :t, :, :, :]
        target_h = min(decoded_5d.shape[2], reference_h)
        target_w = min(decoded_5d.shape[3], reference_w)
        decoded_5d = decoded_5d[:, :, :target_h, :target_w, :]
        if color_correction_method in ("lab", "wavelet", "adain"):
            reference_5d = reference_full[:b, :t, :, :, :]
            reference_5d = cls._resize_reference(reference_5d, target_h, target_w)
            output_device = decoded_5d.device
            decoded_raw = cls._to_seedvr2_raw(decoded_5d)
            reference_raw = cls._to_seedvr2_raw(reference_5d)
            decoded_flat = decoded_raw.permute(0, 1, 4, 2, 3).reshape(b * t, decoded_raw.shape[4], target_h, target_w)
            reference_flat = reference_raw.permute(0, 1, 4, 2, 3).reshape(b * t, reference_raw.shape[4], target_h, target_w)
            output = cls._color_transfer_chunked(
                decoded_flat, reference_flat, output_device, color_correction_method,
            )
            output = output.reshape(b, t, output.shape[1], output.shape[2], output.shape[3]).permute(0, 1, 3, 4, 2)
            output = output.add(1.0).div(2.0).clamp(0.0, 1.0)
        elif color_correction_method == "none":
            output = decoded_5d
        else:
            raise ValueError(f"SeedVR2PostProcessing: unknown color_correction_method {color_correction_method!r}")

        if alpha_input is not None:
            alpha_5d, _ = cls._as_bthwc(alpha_input)
            alpha_5d = alpha_5d[:output.shape[0], :output.shape[1], :output.shape[2], :output.shape[3], :]
            output = torch.cat([output, alpha_5d.to(dtype=output.dtype, device=output.device)], dim=-1)
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
                next_chunk_size = max(1, chunk_size // SEEDVR2_OOM_BACKOFF_DIVISOR)

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
        dtype_bytes = max(decoded_flat.element_size(), SEEDVR2_DTYPE_BYTES_FLOOR)
        bytes_per_frame = height * width * channels * dtype_bytes * multiplier
        if bytes_per_frame <= 0:
            return frames
        color_device = comfy.model_management.vae_device()
        free_memory = comfy.model_management.get_free_memory(color_device)
        chunk_size = int((free_memory * SEEDVR2_COLOR_MEM_HEADROOM) // bytes_per_frame)
        return max(1, min(frames, chunk_size))

    @staticmethod
    def _color_correction_memory_multiplier(color_correction_method):
        if color_correction_method == "lab":
            return SEEDVR2_LAB_SCALE_MULTIPLIER
        if color_correction_method == "wavelet":
            return SEEDVR2_WAVELET_SCALE_MULTIPLIER
        if color_correction_method == "adain":
            return SEEDVR2_ADAIN_SCALE_MULTIPLIER
        raise ValueError(f"SeedVR2PostProcessing: unknown color_correction_method {color_correction_method!r}")

    @staticmethod
    def _resize_reference(reference, height, width):
        if reference.shape[2] == height and reference.shape[3] == width:
            return reference
        b, t = reference.shape[:2]
        reference_flat = reference.permute(0, 1, 4, 2, 3).reshape(b * t, reference.shape[4], reference.shape[2], reference.shape[3])
        resized = TVF.resize(
            reference_flat,
            size=(height, width),
            interpolation=InterpolationMode.BICUBIC,
            antialias=not (isinstance(reference_flat, torch.Tensor) and reference_flat.device.type == "mps"),
        )
        return resized.reshape(b, t, resized.shape[1], height, width).permute(0, 1, 3, 4, 2)


class SeedVR2Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2Conditioning",
            display_name="Apply SeedVR2 Conditioning",
            category="conditioning",
            description="Build SeedVR2 positive/negative conditioning from a VAE latent.",
            search_aliases=["seedvr2", "upscale", "conditioning"],
            inputs=[
                io.Model.Input("model", tooltip="The SeedVR2 model."),
                io.Latent.Input("vae_conditioning", display_name="latent"),
            ],
            outputs=[
                io.Model.Output(display_name="model", tooltip="The SeedVR2 model, passed through."),
                io.Conditioning.Output(display_name="positive", tooltip="The positive conditioning for sampling."),
                io.Conditioning.Output(display_name="negative", tooltip="The negative conditioning for sampling."),
                io.Latent.Output(display_name="latent", tooltip="The latent to denoise."),
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
        if vae_conditioning.shape[-1] == SEEDVR2_LATENT_CHANNELS and vae_conditioning.shape[1] != SEEDVR2_LATENT_CHANNELS:
            raise ValueError(
                "SeedVR2Conditioning expects SeedVR2 VAE latents in Comfy "
                f"channel-first layout (B, {SEEDVR2_LATENT_CHANNELS}, T, H, W); "
                f"got channel-last shape {tuple(vae_conditioning.shape)}."
            )
        vae_conditioning = vae_conditioning.movedim(1, -1).contiguous()
        model_patcher = model
        model = _resolve_seedvr2_diffusion_model(model_patcher)
        pos_cond = model.positive_conditioning
        neg_cond = model.negative_conditioning

        condition = torch.stack([get_conditions(c, c) for c in vae_conditioning])
        condition = condition.movedim(-1, 1)
        latent = vae_conditioning.movedim(-1, 1)

        latent = latent.reshape(latent.shape[0], latent.shape[1] * latent.shape[2], latent.shape[3], latent.shape[4])
        condition = condition.reshape(condition.shape[0], condition.shape[1] * condition.shape[2], condition.shape[3], condition.shape[4])

        negative = [[neg_cond.unsqueeze(0), {"condition": condition}]]
        positive = [[pos_cond.unsqueeze(0), {"condition": condition}]]

        return io.NodeOutput(model_patcher, positive, negative, {"samples": latent})

class SeedVRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SeedVR2Conditioning,
            SeedVR2Preprocess,
            SeedVR2PostProcessing,
        ]

async def comfy_entrypoint() -> SeedVRExtension:
    return SeedVRExtension()

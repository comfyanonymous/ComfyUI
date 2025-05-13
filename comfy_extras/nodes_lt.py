import io
import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.model_sampling
import comfy.utils
import math
import numpy as np
import av
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords

class EmptyLTXVLatentVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 768, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 32}),
                              "height": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 32}),
                              "length": ("INT", {"default": 97, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/video/ltxv"

    def generate(self, width, height, length, batch_size=1):
        latent = torch.zeros([batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32], device=comfy.model_management.intermediate_device())
        return ({"samples": latent}, )


class LTXVImgToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE",),
                             "image": ("IMAGE",),
                             "width": ("INT", {"default": 768, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 32}),
                             "height": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 32}),
                             "length": ("INT", {"default": 97, "min": 9, "max": nodes.MAX_RESOLUTION, "step": 8}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                             }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    CATEGORY = "conditioning/video_models"
    FUNCTION = "generate"

    def generate(self, positive, negative, image, vae, width, height, length, batch_size, strength):
        pixels = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)

        latent = torch.zeros([batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32], device=comfy.model_management.intermediate_device())
        latent[:, :, :t.shape[2]] = t

        conditioning_latent_frames_mask = torch.ones(
            (batch_size, 1, latent.shape[2], 1, 1),
            dtype=torch.float32,
            device=latent.device,
        )
        conditioning_latent_frames_mask[:, :, :t.shape[2]] = 1.0 - strength

        return (positive, negative, {"samples": latent, "noise_mask": conditioning_latent_frames_mask}, )


def conditioning_get_any_value(conditioning, key, default=None):
    for t in conditioning:
        if key in t[1]:
            return t[1][key]
    return default


def get_noise_mask(latent):
    noise_mask = latent.get("noise_mask", None)
    latent_image = latent["samples"]
    if noise_mask is None:
        batch_size, _, latent_length, _, _ = latent_image.shape
        noise_mask = torch.ones(
            (batch_size, 1, latent_length, 1, 1),
            dtype=torch.float32,
            device=latent_image.device,
        )
    else:
        noise_mask = noise_mask.clone()
    return noise_mask

def get_keyframe_idxs(cond):
    keyframe_idxs = conditioning_get_any_value(cond, "keyframe_idxs", None)
    if keyframe_idxs is None:
        return None, 0
    num_keyframes = torch.unique(keyframe_idxs[:, 0]).shape[0]
    return keyframe_idxs, num_keyframes

class LTXVAddGuide:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE",),
                             "latent": ("LATENT",),
                             "image": ("IMAGE", {"tooltip": "Image or video to condition the latent video on. Must be 8*n + 1 frames."
                                                 "If the video is not 8*n + 1 frames, it will be cropped to the nearest 8*n + 1 frames."}),
                             "frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999,
                                                   "tooltip": "Frame index to start the conditioning at. For single-frame images or "
                                                   "videos with 1-8 frames, any frame_idx value is acceptable. For videos with 9+ "
                                                   "frames, frame_idx must be divisible by 8, otherwise it will be rounded down to "
                                                   "the nearest multiple of 8. Negative values are counted from the end of the video."}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             }
            }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    CATEGORY = "conditioning/video_models"
    FUNCTION = "generate"

    def __init__(self):
        self._num_prefix_frames = 2
        self._patchifier = SymmetricPatchifier(1)

    def encode(self, vae, latent_width, latent_height, images, scale_factors):
        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        images = images[:(images.shape[0] - 1) // time_scale_factor * time_scale_factor + 1]
        pixels = comfy.utils.common_upscale(images.movedim(-1, 1), latent_width * width_scale_factor, latent_height * height_scale_factor, "bilinear", crop="disabled").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)
        return encode_pixels, t

    def get_latent_index(self, cond, latent_length, guide_length, frame_idx, scale_factors):
        time_scale_factor, _, _ = scale_factors
        _, num_keyframes = get_keyframe_idxs(cond)
        latent_count = latent_length - num_keyframes
        frame_idx = frame_idx if frame_idx >= 0 else max((latent_count - 1) * time_scale_factor + 1 + frame_idx, 0)
        if guide_length > 1:
            frame_idx = frame_idx // time_scale_factor * time_scale_factor # frame index must be divisible by 8

        latent_idx = (frame_idx + time_scale_factor - 1) // time_scale_factor

        return frame_idx, latent_idx

    def add_keyframe_index(self, cond, frame_idx, guiding_latent, scale_factors):
        keyframe_idxs, _ = get_keyframe_idxs(cond)
        _, latent_coords = self._patchifier.patchify(guiding_latent)
        pixel_coords = latent_to_pixel_coords(latent_coords, scale_factors, True)
        pixel_coords[:, 0] += frame_idx
        if keyframe_idxs is None:
            keyframe_idxs = pixel_coords
        else:
            keyframe_idxs = torch.cat([keyframe_idxs, pixel_coords], dim=2)
        return node_helpers.conditioning_set_values(cond, {"keyframe_idxs": keyframe_idxs})

    def append_keyframe(self, positive, negative, frame_idx, latent_image, noise_mask, guiding_latent, strength, scale_factors):
        _, latent_idx = self.get_latent_index(
            cond=positive,
            latent_length=latent_image.shape[2],
            guide_length=guiding_latent.shape[2],
            frame_idx=frame_idx,
            scale_factors=scale_factors,
        )
        noise_mask[:, :, latent_idx:latent_idx + guiding_latent.shape[2]] = 1.0

        positive = self.add_keyframe_index(positive, frame_idx, guiding_latent, scale_factors)
        negative = self.add_keyframe_index(negative, frame_idx, guiding_latent, scale_factors)

        mask = torch.full(
            (noise_mask.shape[0], 1, guiding_latent.shape[2], 1, 1),
            1.0 - strength,
            dtype=noise_mask.dtype,
            device=noise_mask.device,
        )

        latent_image = torch.cat([latent_image, guiding_latent], dim=2)
        noise_mask = torch.cat([noise_mask, mask], dim=2)
        return positive, negative, latent_image, noise_mask

    def replace_latent_frames(self, latent_image, noise_mask, guiding_latent, latent_idx, strength):
        cond_length = guiding_latent.shape[2]
        assert latent_image.shape[2] >= latent_idx + cond_length, "Conditioning frames exceed the length of the latent sequence."

        mask = torch.full(
            (noise_mask.shape[0], 1, cond_length, 1, 1),
            1.0 - strength,
            dtype=noise_mask.dtype,
            device=noise_mask.device,
        )

        latent_image = latent_image.clone()
        noise_mask = noise_mask.clone()

        latent_image[:, :, latent_idx : latent_idx + cond_length] = guiding_latent
        noise_mask[:, :, latent_idx : latent_idx + cond_length] = mask

        return latent_image, noise_mask

    def generate(self, positive, negative, vae, latent, image, frame_idx, strength):
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        image, t = self.encode(vae, latent_width, latent_height, image, scale_factors)

        frame_idx, latent_idx = self.get_latent_index(positive, latent_length, len(image), frame_idx, scale_factors)
        assert latent_idx + t.shape[2] <= latent_length, "Conditioning frames exceed the length of the latent sequence."

        num_prefix_frames = min(self._num_prefix_frames, t.shape[2])

        positive, negative, latent_image, noise_mask = self.append_keyframe(
            positive,
            negative,
            frame_idx,
            latent_image,
            noise_mask,
            t[:, :, :num_prefix_frames],
            strength,
            scale_factors,
        )

        latent_idx += num_prefix_frames

        t = t[:, :, num_prefix_frames:]
        if t.shape[2] == 0:
            return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask},)

        latent_image, noise_mask = self.replace_latent_frames(
            latent_image,
            noise_mask,
            t,
            latent_idx,
            strength,
        )

        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask},)


class LTXVCropGuides:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "latent": ("LATENT",),
                             }
            }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    CATEGORY = "conditioning/video_models"
    FUNCTION = "crop"

    def __init__(self):
        self._patchifier = SymmetricPatchifier(1)

    def crop(self, positive, negative, latent):
        latent_image = latent["samples"].clone()
        noise_mask = get_noise_mask(latent)

        _, num_keyframes = get_keyframe_idxs(positive)
        if num_keyframes == 0:
            return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask},)

        latent_image = latent_image[:, :, :-num_keyframes]
        noise_mask = noise_mask[:, :, :-num_keyframes]

        positive = node_helpers.conditioning_set_values(positive, {"keyframe_idxs": None})
        negative = node_helpers.conditioning_set_values(negative, {"keyframe_idxs": None})

        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask},)


class LTXVConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "append"

    CATEGORY = "conditioning/video_models"

    def append(self, positive, negative, frame_rate):
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})
        return (positive, negative)


class ModelSamplingLTXV:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step":0.01}),
                              "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step":0.01}),
                              },
                "optional": {"latent": ("LATENT",), }
                }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, max_shift, base_shift, latent=None):
        m = model.clone()

        if latent is None:
            tokens = 4096
        else:
            tokens = math.prod(latent["samples"].shape[2:])

        x1 = 1024
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (tokens) * mm + b

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)

        return (m, )


class LTXVScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step":0.01}),
                     "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step":0.01}),
                     "stretch": ("BOOLEAN", {
                        "default": True,
                        "tooltip": "Stretch the sigmas to be in the range [terminal, 1]."
                    }),
                     "terminal": (
                        "FLOAT",
                        {
                            "default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01,
                            "tooltip": "The terminal value of the sigmas after stretching."
                        },
                    ),
                    },
                "optional": {"latent": ("LATENT",), }
               }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, max_shift, base_shift, stretch, terminal, latent=None):
        if latent is None:
            tokens = 4096
        else:
            tokens = math.prod(latent["samples"].shape[2:])

        sigmas = torch.linspace(1.0, 0.0, steps + 1)

        x1 = 1024
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = (tokens) * mm + b

        power = 1
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

        # Stretch sigmas so that its final value matches the given terminal value.
        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        return (sigmas,)

def encode_single_frame(output_file, image_array: np.ndarray, crf):
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def decode_single_frame(video_file):
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def preprocess(image: torch.Tensor, crf=29):
    if crf == 0:
        return image

    image_array = (image[:(image.shape[0] // 2) * 2, :(image.shape[1] // 2) * 2] * 255.0).byte().cpu().numpy()
    with io.BytesIO() as output_file:
        encode_single_frame(output_file, image_array, crf)
        video_bytes = output_file.getvalue()
    with io.BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    tensor = torch.tensor(image_array, dtype=image.dtype, device=image.device) / 255.0
    return tensor


class LTXVPreprocess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "img_compression": (
                    "INT",
                    {
                        "default": 35,
                        "min": 0,
                        "max": 100,
                        "tooltip": "Amount of compression to apply on image.",
                    },
                ),
            }
        }

    FUNCTION = "preprocess"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    CATEGORY = "image"

    def preprocess(self, image, img_compression):
        output_images = []
        for i in range(image.shape[0]):
            output_images.append(preprocess(image[i], img_compression))
        return (torch.stack(output_images),)


NODE_CLASS_MAPPINGS = {
    "EmptyLTXVLatentVideo": EmptyLTXVLatentVideo,
    "LTXVImgToVideo": LTXVImgToVideo,
    "ModelSamplingLTXV": ModelSamplingLTXV,
    "LTXVConditioning": LTXVConditioning,
    "LTXVScheduler": LTXVScheduler,
    "LTXVAddGuide": LTXVAddGuide,
    "LTXVPreprocess": LTXVPreprocess,
    "LTXVCropGuides": LTXVCropGuides,
}

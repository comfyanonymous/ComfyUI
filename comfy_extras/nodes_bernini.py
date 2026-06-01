import torch

import comfy.model_management
import comfy.utils
import node_helpers


def _resize_long_edge(image, max_size, stride=16):
    """Resize (preserve aspect) so the long edge <= max_size, snapped to `stride`."""
    h, w = image.shape[1], image.shape[2]
    scale = min(max_size / max(h, w), 1.0)
    nh = max(stride, round(h * scale / stride) * stride)
    nw = max(stride, round(w * scale / stride) * stride)
    return comfy.utils.common_upscale(image[:, :, :, :3].movedim(-1, 1), nw, nh, "bilinear", "disabled").movedim(1, -1)


class BerniniConditioning:
    """Bernini-R in-context conditioning for a Wan2.2-A14B model.

    Attaches the VAE-encoded source video / reference images to BOTH positive and
    negative conditioning as ``context_latents`` -- an ordered list of clean
    latent streams (source video first, then each reference image), which the Wan
    model appends as extra tokens with per-stream source_id rope. With stock CFG
    the conditions stay fixed and only the text varies; at cfg=1.0 (distill LoRA)
    it's a single forward over the full conditioning.

    The task is inferred from which inputs are connected -- no model input and no
    task selector needed; the model loads as a normal Wan2.2 checkpoint via the
    stock UNETLoader:
      (nothing)                 -> t2v
      source_video              -> v2v
      source_video + ref images -> rv2v
      ref images only           -> r2v   (each kept at native aspect)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "source_video": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "ref_max_size": ("INT", {"default": 848, "min": 16, "max": 8192, "step": 16}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "build"
    CATEGORY = "conditioning/video_models"

    def build(self, positive, negative, vae, width, height, length, batch_size,
              source_video=None, reference_images=None, ref_max_size=848):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
                             device=comfy.model_management.intermediate_device())

        # Ordered list of condition streams: source video (source_id 1) first,
        # then each reference image (source_id 2, 3, ...). The model assigns the
        # source_id from list order. The task (t2v/v2v/rv2v/r2v) is implied by
        # which inputs are present.
        context = []
        if source_video is not None:
            vid = comfy.utils.common_upscale(source_video[:length, :, :, :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            context.append(vae.encode(vid[:, :, :, :3]))

        if reference_images is not None:
            for i in range(reference_images.shape[0]):
                img = _resize_long_edge(reference_images[i:i + 1], ref_max_size)  # native aspect per ref
                context.append(vae.encode(img[:, :, :, :3]))

        if context:
            positive = node_helpers.conditioning_set_values(positive, {"context_latents": context})
            negative = node_helpers.conditioning_set_values(negative, {"context_latents": context})

        return (positive, negative, {"samples": latent})


NODE_CLASS_MAPPINGS = {
    "BerniniConditioning": BerniniConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BerniniConditioning": "Bernini Conditioning",
}

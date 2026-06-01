import torch

import comfy.ldm.wan.model
import comfy.model_base
import comfy.model_management
import comfy.utils
import node_helpers


def _patch_bernini(model):
    """Flip a loaded Wan2.2-A14B model into Bernini-R mode.

    The Bernini checkpoint is architecturally identical to Wan2.2-A14B (no new
    params), so we just swap the forward (BerniniWanModel) and the conditioning
    plumbing (WAN22_Bernini) onto the already-loaded model. Idempotent.
    """
    model.model.diffusion_model.__class__ = comfy.ldm.wan.model.BerniniWanModel
    model.model.__class__ = comfy.model_base.WAN22_Bernini
    model.model.memory_usage_factor_conds = ("bernini_video_latent", "bernini_image_latents")
    return model


def _encode_frames(vae, image, width, height):
    image = comfy.utils.common_upscale(image[:, :, :, :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    return vae.encode(image[:, :, :, :3])


class BerniniConditioning:
    """Routes Bernini-R inputs and activates Bernini mode on the model(s).

    Attaches the VAE-encoded source video / reference images to BOTH the
    positive and negative conditioning so stock CFG keeps the conditions fixed
    and only varies the text -- giving Bernini's v2v / rv2v guidance form. For
    cfg=1.0 (distill LoRA) the same setup is a single forward with the full
    conditioning. t2v attaches nothing.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "task_type": (["t2v", "v2v", "rv2v"],),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "model_low": ("MODEL",),
                "source_video": ("IMAGE",),
                "reference_images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MODEL", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "model_low", "positive", "negative", "latent")
    FUNCTION = "build"
    CATEGORY = "conditioning/video_models"

    def build(self, model, positive, negative, vae, task_type, width, height, length, batch_size,
              model_low=None, source_video=None, reference_images=None):
        model = _patch_bernini(model)
        if model_low is not None:
            model_low = _patch_bernini(model_low)

        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
                             device=comfy.model_management.intermediate_device())

        values = {}
        if task_type in ("v2v", "rv2v") and source_video is not None:
            values["bernini_video_latent"] = _encode_frames(vae, source_video[:length], width, height)

        if task_type == "rv2v" and reference_images is not None:
            # each reference image is an independent single-frame stream (its own source_id)
            refs = [_encode_frames(vae, reference_images[i:i + 1], width, height) for i in range(reference_images.shape[0])]
            values["bernini_image_latents"] = torch.cat(refs, dim=2)

        if values:
            positive = node_helpers.conditioning_set_values(positive, values)
            negative = node_helpers.conditioning_set_values(negative, values)

        return (model, model_low, positive, negative, {"samples": latent})


NODE_CLASS_MAPPINGS = {
    "BerniniConditioning": BerniniConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BerniniConditioning": "Bernini Conditioning",
}

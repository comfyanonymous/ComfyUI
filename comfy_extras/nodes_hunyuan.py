from numpy import arccos
import nodes
import node_helpers
import torch
import re
import comfy.model_management


class CLIPTextEncodeHunyuanDiT:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP", ),
            "bert": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "mt5xl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, bert, mt5xl):
        tokens = clip.tokenize(bert)
        tokens["mt5xl"] = clip.tokenize(mt5xl)["mt5xl"]

        return (clip.encode_from_tokens_scheduled(tokens), )

class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

def normalized_guidance_apg(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
    use_original_formulation: bool = False,
):
    diff = pred_cond - pred_uncond
    dim = [-i for i in range(1, len(diff.shape))]

    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dim, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    v0, v1 = diff.double(), pred_cond.double()
    v1 = torch.nn.functional.normalize(v1, dim=dim)
    v0_parallel = (v0 * v1).sum(dim=dim, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    diff_parallel, diff_orthogonal = v0_parallel.type_as(diff), v0_orthogonal.type_as(diff)

    normalized_update = diff_orthogonal + eta * diff_parallel
    pred = pred_cond if use_original_formulation else pred_uncond
    pred = pred + guidance_scale * normalized_update

    return pred

class AdaptiveProjectedGuidance:
    def __init__(
        self,
        guidance_scale: float = 7.5,
        adaptive_projected_guidance_momentum=None,
        adaptive_projected_guidance_rescale: float = 15.0,
        # eta: float = 1.0,
        eta: float = 0.0,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        super().__init__()

        self.guidance_scale = guidance_scale
        self.adaptive_projected_guidance_momentum = adaptive_projected_guidance_momentum
        self.adaptive_projected_guidance_rescale = adaptive_projected_guidance_rescale
        self.eta = eta
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation
        self.momentum_buffer = None

    def __call__(self, pred_cond: torch.Tensor, pred_uncond=None, step=None) -> torch.Tensor:

        if step == 0 and self.adaptive_projected_guidance_momentum is not None:
            self.momentum_buffer = MomentumBuffer(self.adaptive_projected_guidance_momentum)

        pred = normalized_guidance_apg(
            pred_cond,
            pred_uncond,
            self.guidance_scale,
            self.momentum_buffer,
            self.eta,
            self.adaptive_projected_guidance_rescale,
            self.use_original_formulation,
        )

        return pred

class HunyuanMixModeAPG:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "has_quoted_text": ("HAS_QUOTED_TEXT", ),

                "guidance_scale": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 30.0, "step": 0.1}),

                "general_eta": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "general_norm_threshold": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "general_momentum": ("FLOAT", {"default": -0.5, "min": -5.0, "max": 1.0, "step": 0.01}),
                "general_start_step": ("INT", {"default": 10, "min": -1, "max": 1000}),

                "ocr_eta": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "ocr_norm_threshold": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "ocr_momentum": ("FLOAT", {"default": -0.5, "min": -5.0, "max": 1.0, "step": 0.01}),
                "ocr_start_step": ("INT", {"default": 75, "min": -1, "max": 1000}),

            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_mix_mode_apg"
    CATEGORY = "sampling/custom_sampling/hunyuan"


    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True

    def apply_mix_mode_apg(self, model,  has_quoted_text, guidance_scale, general_eta, general_norm_threshold, general_momentum, general_start_step,
                          ocr_eta, ocr_norm_threshold, ocr_momentum, ocr_start_step):

        general_apg = AdaptiveProjectedGuidance(
            guidance_scale=guidance_scale,
            eta=general_eta,
            adaptive_projected_guidance_rescale=general_norm_threshold,
            adaptive_projected_guidance_momentum=general_momentum
        )

        ocr_apg = AdaptiveProjectedGuidance(
            eta=ocr_eta,
            adaptive_projected_guidance_rescale=ocr_norm_threshold,
            adaptive_projected_guidance_momentum=ocr_momentum
        )

        current_step = {"step": 0}

        def cfg_function(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]

            step = current_step["step"]
            current_step["step"] += 1

            if not has_quoted_text:
                if step > general_start_step:
                    modified_cond = general_apg(cond, uncond, step).to(torch.bfloat16)
                    return modified_cond
                else:
                    if cond_scale > 1:
                        _ = general_apg(cond, uncond, step) # track momentum
                        return uncond + (cond - uncond) * cond_scale
            else:
                if step > ocr_start_step:
                    modified_cond = ocr_apg(cond, uncond, step)
                    return modified_cond
                else:
                    if cond_scale > 1:
                        _ = ocr_apg(cond, uncond, step)
                        return uncond + (cond - uncond) * cond_scale

            return cond


        m = model.clone()
        m.set_model_sampler_cfg_function(cfg_function, disable_cfg1_optimization=True)
        return (m,)

class CLIPTextEncodeHunyuanDiTWithTextDetection:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP", ),
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }}
    RETURN_TYPES = ("CONDITIONING", "HAS_QUOTED_TEXT", "STRING")
    RETURN_NAMES = ("conditioning", "has_quoted_text", "text")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning/hunyuan"

    def detect_quoted_text(self, text):
        """Detect quoted text in the prompt"""
        text_prompt_texts = []

        # Patterns to match different quote styles
        pattern_quote_double = r'\"(.*?)\"'
        pattern_quote_chinese_single = r'‘(.*?)’'
        pattern_quote_chinese_double = r'“(.*?)”'

        matches_quote_double = re.findall(pattern_quote_double, text)
        matches_quote_chinese_single = re.findall(pattern_quote_chinese_single, text)
        matches_quote_chinese_double = re.findall(pattern_quote_chinese_double, text)

        text_prompt_texts.extend(matches_quote_double)
        text_prompt_texts.extend(matches_quote_chinese_single)
        text_prompt_texts.extend(matches_quote_chinese_double)

        return len(text_prompt_texts) > 0

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        has_quoted_text = self.detect_quoted_text(text)

        conditioning = clip.encode_from_tokens_scheduled(tokens)

        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['has_quoted_text'] = has_quoted_text
            c.append(n)

        return (c, has_quoted_text, text)

class CLIPTextEncodeHunyuanImageRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP", ),
            "text": ("STRING", ),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning/hunyuan"


    def encode(self, clip, text):
        tokens = clip.tokenize(text)

        conditioning = clip.encode_from_tokens_scheduled(tokens)

        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c.append(n)

        return (c, )



class EmptyHunyuanLatentVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 848, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "length": ("INT", {"default": 25, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/video"

    def generate(self, width, height, length, batch_size=1):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return ({"samples":latent}, )

PROMPT_TEMPLATE_ENCODE_VIDEO_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

class TextEncodeHunyuanVideo_ImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "image_interleave": ("INT", {"default": 2, "min": 1, "max": 512, "tooltip": "How much the image influences things vs the text prompt. Higher number means more influence from the text prompt."}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, clip_vision_output, prompt, image_interleave):
        tokens = clip.tokenize(prompt, llama_template=PROMPT_TEMPLATE_ENCODE_VIDEO_I2V, image_embeds=clip_vision_output.mm_projected, image_interleave=image_interleave)
        return (clip.encode_from_tokens_scheduled(tokens), )

class HunyuanImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 848, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 53, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "guidance_type": (["v1 (concat)", "v2 (replace)", "custom"], )
                },
                "optional": {"start_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, vae, width, height, length, batch_size, guidance_type, start_image=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        out_latent = {}

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length, :, :, :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

            concat_latent_image = vae.encode(start_image)
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            if guidance_type == "v1 (concat)":
                cond = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            elif guidance_type == "v2 (replace)":
                cond = {'guiding_frame_index': 0}
                latent[:, :, :concat_latent_image.shape[2]] = concat_latent_image
                out_latent["noise_mask"] = mask
            elif guidance_type == "custom":
                cond = {"ref_latent": concat_latent_image}

            positive = node_helpers.conditioning_set_values(positive, cond)

        out_latent["samples"] = latent
        return (positive, out_latent)

class EmptyHunyuanImageLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 2048, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 32}),
                              "height": ("INT", {"default": 2048, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 32}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 64, height // 32, width // 32], device=comfy.model_management.intermediate_device())
        return ({"samples":latent}, )

class HunyuanRefinerLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "latent": ("LATENT", ),
                             "noise_augmentation": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                             }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    FUNCTION = "execute"

    def execute(self, positive, negative, latent, noise_augmentation):
        latent = latent["samples"]
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": latent, "noise_augmentation": noise_augmentation})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": latent, "noise_augmentation": noise_augmentation})
        out_latent = {}
        out_latent["samples"] = torch.zeros([latent.shape[0], 32, latent.shape[-3], latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())
        return (positive, negative, out_latent)



NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanMixModeAPG": "Hunyuan Mix Mode APG",
}

NODE_CLASS_MAPPINGS = {
    "HunyuanMixModeAPG": HunyuanMixModeAPG,
    "CLIPTextEncodeHunyuanDiT": CLIPTextEncodeHunyuanDiT,
    "CLIPTextEncodeHunyuanDiTWithTextDetection": CLIPTextEncodeHunyuanDiTWithTextDetection,
    "CLIPTextEncodeHunyuanImageRefiner": CLIPTextEncodeHunyuanImageRefiner,
    "TextEncodeHunyuanVideo_ImageToVideo": TextEncodeHunyuanVideo_ImageToVideo,
    "EmptyHunyuanLatentVideo": EmptyHunyuanLatentVideo,
    "HunyuanImageToVideo": HunyuanImageToVideo,
    "EmptyHunyuanImageLatent": EmptyHunyuanImageLatent,
    "HunyuanRefinerLatent": HunyuanRefinerLatent,
}

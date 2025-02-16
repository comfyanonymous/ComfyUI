from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import torch


class RenormCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "cfg_trunc": ("FLOAT", {"default": 100, "min": 0.0, "max": 100.0, "step": 0.01}),
                              "renorm_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, cfg_trunc, renorm_cfg):
        def renorm_cfg_func(args):
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            timestep = args["timestep"]
            x_orig = args["input"]
            in_channels = model.model.diffusion_model.in_channels

            if timestep[0] < cfg_trunc:
                cond_eps, uncond_eps = cond_denoised[:, :in_channels], uncond_denoised[:, :in_channels]
                cond_rest, _ = cond_denoised[:, in_channels:], uncond_denoised[:, in_channels:]
                half_eps = uncond_eps + cond_scale * (cond_eps - uncond_eps)
                half_rest = cond_rest

                if float(renorm_cfg) > 0.0:
                    ori_pos_norm = torch.linalg.vector_norm(cond_eps
                            , dim=tuple(range(1, len(cond_eps.shape))), keepdim=True
                    )
                    max_new_norm = ori_pos_norm * float(renorm_cfg)
                    new_pos_norm = torch.linalg.vector_norm(
                            half_eps, dim=tuple(range(1, len(half_eps.shape))), keepdim=True
                        )
                    if new_pos_norm >= max_new_norm:
                        half_eps = half_eps * (max_new_norm / new_pos_norm)
            else:
                cond_eps, uncond_eps = cond_denoised[:, :in_channels], uncond_denoised[:, :in_channels]
                cond_rest, _ = cond_denoised[:, in_channels:], uncond_denoised[:, in_channels:]
                half_eps = cond_eps
                half_rest = cond_rest

            cfg_result = torch.cat([half_eps, half_rest], dim=1)

            # cfg_result = uncond_denoised + (cond_denoised - uncond_denoised) * cond_scale

            return x_orig - cfg_result

        m = model.clone()
        m.set_model_sampler_cfg_function(renorm_cfg_func)
        return (m, )


class CLIPTextEncodeLumina2(ComfyNodeABC):
    SYSTEM_PROMPT = {
        "superior": "You are an assistant designed to generate superior images with the superior "\
            "degree of image-text alignment based on textual prompts or user prompts.",
        "alignment": "You are an assistant designed to generate high-quality images with the "\
            "highest degree of image-text alignment based on textual prompts."
    }
    SYSTEM_PROMPT_TIP = "Lumina2 provide two types of system prompts:" \
        "Superior: You are an assistant designed to generate superior images with the superior "\
        "degree of image-text alignment based on textual prompts or user prompts. "\
        "Alignment: You are an assistant designed to generate high-quality images with the highest "\
        "degree of image-text alignment based on textual prompts."
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "system_prompt": (list(CLIPTextEncodeLumina2.SYSTEM_PROMPT.keys()), {"tooltip": CLIPTextEncodeLumina2.SYSTEM_PROMPT_TIP}),
                "user_prompt": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes a system prompt and a user prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, clip, user_prompt, system_prompt):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        system_prompt = CLIPTextEncodeLumina2.SYSTEM_PROMPT[system_prompt]
        prompt = f'{system_prompt} <Prompt Start> {user_prompt}'
        tokens = clip.tokenize(prompt)
        return (clip.encode_from_tokens_scheduled(tokens), )


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeLumina2": CLIPTextEncodeLumina2,
    "RenormCFG": RenormCFG
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeLumina2": "CLIP Text Encode for Lumina2",
}

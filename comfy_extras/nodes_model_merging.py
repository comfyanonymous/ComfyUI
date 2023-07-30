import comfy.sd
import comfy.utils
import comfy.model_base

import folder_paths
import json
import os

from comfy.cli_args import args

class ModelMergeSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model1": ("MODEL",),
                              "model2": ("MODEL",),
                              "ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    CATEGORY = "advanced/model_merging"

    def merge(self, model1, model2, ratio):
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for k in kp:
            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
        return (m, )

class CLIPMergeSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip1": ("CLIP",),
                              "clip2": ("CLIP",),
                              "ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "merge"

    CATEGORY = "advanced/model_merging"

    def merge(self, clip1, clip2, ratio):
        m = clip1.clone()
        kp = clip2.get_key_patches()
        for k in kp:
            if k.endswith(".position_ids") or k.endswith(".logit_scale"):
                continue
            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
        return (m, )

class ModelMergeBlocks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model1": ("MODEL",),
                              "model2": ("MODEL",),
                              "input": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "middle": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    CATEGORY = "advanced/model_merging"

    def merge(self, model1, model2, **kwargs):
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        default_ratio = next(iter(kwargs.values()))

        for k in kp:
            ratio = default_ratio
            k_unet = k[len("diffusion_model."):]

            last_arg_size = 0
            for arg in kwargs:
                if k_unet.startswith(arg) and last_arg_size < len(arg):
                    ratio = kwargs[arg]
                    last_arg_size = len(arg)

            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
        return (m, )

class CheckpointSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP",),
                              "vae": ("VAE",),
                              "filename_prefix": ("STRING", {"default": "checkpoints/ComfyUI"}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "advanced/model_merging"

    def save(self, model, clip, vae, filename_prefix, prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = {}

        enable_modelspec = True
        if isinstance(model.model, comfy.model_base.SDXL):
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base"
        elif isinstance(model.model, comfy.model_base.SDXLRefiner):
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-refiner"
        else:
            enable_modelspec = False

        if enable_modelspec:
            metadata["modelspec.sai_model_spec"] = "1.0.0"
            metadata["modelspec.implementation"] = "sgm"
            metadata["modelspec.title"] = "{} {}".format(filename, counter)

        #TODO:
        # "stable-diffusion-v1", "stable-diffusion-v1-inpainting", "stable-diffusion-v2-512",
        # "stable-diffusion-v2-768-v", "stable-diffusion-v2-unclip-l", "stable-diffusion-v2-unclip-h",
        # "v2-inpainting"

        if model.model.model_type == comfy.model_base.ModelType.EPS:
            metadata["modelspec.predict_key"] = "epsilon"
        elif model.model.model_type == comfy.model_base.ModelType.V_PREDICTION:
            metadata["modelspec.predict_key"] = "v"

        if not args.disable_metadata:
            metadata["prompt"] = prompt_info
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        comfy.sd.save_checkpoint(output_checkpoint, model, clip, vae, metadata=metadata)
        return {}


NODE_CLASS_MAPPINGS = {
    "ModelMergeSimple": ModelMergeSimple,
    "ModelMergeBlocks": ModelMergeBlocks,
    "CheckpointSave": CheckpointSave,
    "CLIPMergeSimple": CLIPMergeSimple,
}

import comfy.sd
import comfy.utils
import folder_paths
import json
import os

class ModelMergeSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model1": ("MODEL",),
                              "model2": ("MODEL",),
                              "ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    CATEGORY = "_for_testing/model_merging"

    def merge(self, model1, model2, ratio):
        m = model1.clone()
        sd = model2.model_state_dict("diffusion_model.")
        for k in sd:
            m.add_patches({k: (sd[k], )}, 1.0 - ratio, ratio)
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

    CATEGORY = "_for_testing/model_merging"

    def merge(self, model1, model2, **kwargs):
        m = model1.clone()
        sd = model2.model_state_dict("diffusion_model.")
        default_ratio = next(iter(kwargs.values()))

        for k in sd:
            ratio = default_ratio
            k_unet = k[len("diffusion_model."):]

            for arg in kwargs:
                if k_unet.startswith(arg):
                    ratio = kwargs[arg]

            m.add_patches({k: (sd[k], )}, 1.0 - ratio, ratio)
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

    CATEGORY = "_for_testing/model_merging"

    def save(self, model, clip, vae, filename_prefix, prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = {"prompt": prompt_info}
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
}

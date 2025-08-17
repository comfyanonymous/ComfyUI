import os
import folder_paths
from ..libs.utils import AlwaysEqualProxy

class showLoaderSettingsNames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("ckpt_name", "vae_name", "lora_name")

    FUNCTION = "notify"
    OUTPUT_NODE = True

    CATEGORY = "EasyUse/Util"

    def notify(self, pipe, names=None, unique_id=None, extra_pnginfo=None):
        if unique_id and extra_pnginfo and "workflow" in extra_pnginfo:
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                ckpt_name = pipe['loader_settings']['ckpt_name'] if 'ckpt_name' in pipe['loader_settings'] else ''
                vae_name = pipe['loader_settings']['vae_name'] if 'vae_name' in pipe['loader_settings'] else ''
                lora_name = pipe['loader_settings']['lora_name'] if 'lora_name' in pipe['loader_settings'] else ''

                if ckpt_name:
                    ckpt_name = os.path.basename(os.path.splitext(ckpt_name)[0])
                if vae_name:
                    vae_name = os.path.basename(os.path.splitext(vae_name)[0])
                if lora_name:
                    lora_name = os.path.basename(os.path.splitext(lora_name)[0])

                names = "ckpt_name: " + ckpt_name + '\n' + "vae_name: " + vae_name + '\n' + "lora_name: " + lora_name
                node["widgets_values"] = names

        return {"ui": {"text": [names]}, "result": (ckpt_name, vae_name, lora_name)}

class sliderControl:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (['ipadapter layer weights'],),
                "model_type": (['sdxl', 'sd1'],),
            },
            "hidden": {
                "prompt": "PROMPT",
                "my_unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layer_weights",)

    FUNCTION = "control"

    CATEGORY = "EasyUse/Util"

    def control(self, mode, model_type, prompt=None, my_unique_id=None, extra_pnginfo=None):
        values = ''
        if my_unique_id in prompt:
            if 'values' in prompt[my_unique_id]["inputs"]:
                values = prompt[my_unique_id]["inputs"]['values']

        return (values,)

class setCkptName:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy('*'),)
    RETURN_NAMES = ("ckpt_name",)
    FUNCTION = "set_name"
    CATEGORY = "EasyUse/Util"

    def set_name(self, ckpt_name):
        return (ckpt_name,)

class setControlName:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "controlnet_name": (folder_paths.get_filename_list("controlnet"),),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy('*'),)
    RETURN_NAMES = ("controlnet_name",)
    FUNCTION = "set_name"
    CATEGORY = "EasyUse/Util"

    def set_name(self, controlnet_name):
        return (controlnet_name,)
    
class setLoraName:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy('*'),)
    RETURN_NAMES = ("lora_name",)
    FUNCTION = "set_name"
    CATEGORY = "EasyUse/Util"

    def set_name(self, lora_name):
        return (lora_name,)


NODE_CLASS_MAPPINGS = {
    "easy showLoaderSettingsNames": showLoaderSettingsNames,
    "easy sliderControl": sliderControl,
    "easy ckptNames": setCkptName,
    "easy controlnetNames": setControlName,
    "easy loraNames": setLoraName,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy showLoaderSettingsNames": "Show Loader Settings Names",
    "easy sliderControl": "Easy Slider Control",
    "easy ckptNames": "Ckpt Names",
    "easy controlnetNames": "ControlNet Names",
    "easy loraNames": "Lora Names",
}

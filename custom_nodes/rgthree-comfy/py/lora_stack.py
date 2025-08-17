from .constants import get_category, get_name
from nodes import LoraLoader
import folder_paths


class RgthreeLoraLoaderStack:

    NAME = get_name('Lora Loader Stack')
    CATEGORY = get_category()

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP", ),

                "lora_01": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_01":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "lora_02": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_02":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "lora_03": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_03":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "lora_04": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_04":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    def load_lora(self, model, clip, lora_01, strength_01, lora_02, strength_02, lora_03, strength_03, lora_04, strength_04):
        if lora_01 != "None" and strength_01 != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_01, strength_01, strength_01)
        if lora_02 != "None" and strength_02 != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_02, strength_02, strength_02)
        if lora_03 != "None" and strength_03 != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_03, strength_03, strength_03)
        if lora_04 != "None" and strength_04 != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_04, strength_04, strength_04)

        return (model, clip)


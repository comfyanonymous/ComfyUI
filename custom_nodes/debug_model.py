import os

from torch import Tensor


class DebugModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_input": ("MODEL",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "debug_node"
    OUTPUT_NODE = True

    CATEGORY = "inflamously"

    def debug_node(self, model_input):
        print("Model:", model_input)
        return {}


NODE_CLASS_MAPPINGS = {
    "DebugModel": DebugModel
}
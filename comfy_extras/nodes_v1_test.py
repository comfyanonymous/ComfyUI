import torch
from comfy.comfy_types.node_typing import ComfyNodeABC, IO


class TestNode(ComfyNodeABC):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "xyz": ("XYZ",),
                "some_int": (IO.INT, {"display_name": "new_name",
                                      "min": 0, "max": 127, "default": 42,
                                      "tooltip": "My tooltip 😎", "display": "slider"}),
                "combo": (IO.COMBO, {"options": ["a", "b", "c"], "tooltip": "This is a combo input"}),
            },
            "optional": {
                "mask": (IO.MASK,),
            }
        }
    
    RETURN_TYPES = (IO.INT, IO.IMAGE)
    RETURN_NAMES = ("INT", "img🖼️")
    OUTPUT_TOOLTIPS = (None, "This is an image")
    FUNCTION = "do_thing"

    OUTPUT_NODE = True

    CATEGORY = "v3 nodes"

    def do_thing(self, image: torch.Tensor, xyz, some_int: int, combo: str, mask: torch.Tensor=None):
        return (some_int, image)


NODE_CLASS_MAPPINGS = {
    "V1TestNode1": TestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "V1TestNode1": "V1 Test Node",
}

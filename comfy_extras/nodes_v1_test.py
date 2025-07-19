import torch
from comfy.comfy_types.node_typing import ComfyNodeABC, IO
import asyncio
from comfy.utils import ProgressBar
import time


class TestNode(ComfyNodeABC):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "some_int": (IO.INT, {"display_name": "new_name",
                                      "min": 0, "max": 127, "default": 42,
                                      "tooltip": "My tooltip 😎", "display": "slider"}),
                "combo": (IO.COMBO, {"options": ["a", "b", "c"], "tooltip": "This is a combo input"}),
                "combo2": (IO.COMBO, {"options": ["a", "b", "c"], "multi_select": True, "tooltip": "This is a combo input"}),
            },
            "optional": {
                "xyz": ("XYZ",),
                "mask": (IO.MASK,),
            }
        }

    RETURN_TYPES = (IO.INT, IO.IMAGE)
    RETURN_NAMES = ("INT", "img🖼️")
    OUTPUT_TOOLTIPS = (None, "This is an image")
    FUNCTION = "do_thing"

    OUTPUT_NODE = True

    CATEGORY = "v3 nodes"

    def do_thing(self, image: torch.Tensor, some_int: int, combo: str, combo2: list[str], xyz=None, mask: torch.Tensor=None):
        return (some_int, image)


class TestSleep(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (IO.ANY, {}),
                "seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 9999.0, "step": 0.01, "tooltip": "The amount of seconds to sleep."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "sleep"

    CATEGORY = "_for_testing"

    async def sleep(self, value, seconds, unique_id):
        pbar = ProgressBar(seconds, node_id=unique_id)
        start = time.time()
        expiration = start + seconds
        now = start
        while now < expiration:
            now = time.time()
            pbar.update_absolute(now - start)
            await asyncio.sleep(0.02)
        return (value,)


NODE_CLASS_MAPPINGS = {
    "V1TestNode1": TestNode,
    "V1TestSleep": TestSleep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "V1TestNode1": "V1 Test Node",
    "V1TestSleep": "V1 Test Sleep",
}

import datetime
import math
import os
import random

import PIL
import einops
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision.transforms as T

class DebugCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "cond_input": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE",)
    FUNCTION = "debug_node"
    OUTPUT_NODE = True

    CATEGORY = "inflamously"

    @classmethod
    def IS_CHANGED(s, clip, cond_input):
        # TODO: Why does this not cache immidiately
        return random.randint(0, 10000)

    def debug_node(self, clip, cond_input):
        # print("Cond Shape:", cond_input[0][0].shape)
        # signal = cond_input[0][0].reshape(-1)
        # stripped_signal = signal[::2048]
        plt.plot(cond_input[0][0][0])
        img = PIL.Image.frombytes('RGB', plt.gcf().canvas.get_width_height(), plt.gcf().canvas.tostring_rgb())
        img_tensor = T.PILToTensor()(img) / 255.0
        img_tensor = einops.reduce(img_tensor, "a b c -> 1 b c a", "max")
        return cond_input, img_tensor

NODE_CLASS_MAPPINGS = {
    "DebugCond": DebugCond
}

# TODO: Impl into execution.py
SCRIPT_TEMPLATE_PATH = os.path.join(os.path.join(__file__, os.pardir), "debug_cond.js")

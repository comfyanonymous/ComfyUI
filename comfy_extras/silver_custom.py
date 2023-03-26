import datetime

import torch

import os
import sys
import json
import hashlib
import copy
import traceback

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy_extras.clip_vision
import model_management
import importlib
import folder_paths


class Note:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ()
    FUNCTION = "Note"

    OUTPUT_NODE = False

    CATEGORY = "silver_custom"


NODE_CLASS_MAPPINGS = {
    "Note": Note,
}

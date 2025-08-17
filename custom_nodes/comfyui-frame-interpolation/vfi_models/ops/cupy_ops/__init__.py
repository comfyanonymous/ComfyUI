from .costvol import *
from .sepconv import *
from .softsplat import *
from .adacof import *
from .correlation import *
from comfy.model_management import is_nvidia, get_torch_device_name, get_torch_device

def init():
    if not is_nvidia():
        raise NotImplementedError(f"CuPy ops backend only support CUDA device but found {get_torch_device_name(get_torch_device())} instead. Try Taichi ops backend by editing config.yaml")
    return
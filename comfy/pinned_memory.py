import torch
import comfy.model_management
import comfy.memory_management

from comfy.cli_args import args

def get_pin(module):
    return getattr(module, "_pin", None)

def pin_memory(module):
    if module.pin_failed or args.disable_pinned_memory or get_pin(module) is not None:
        return
    #FIXME: This is a RAM cache trigger event
    size = comfy.memory_management.vram_aligned_size([ module.weight, module.bias ])
    pin = torch.empty((size,), dtype=torch.uint8)
    if comfy.model_management.pin_memory(pin):
        module._pin = pin
    else:
        module.pin_failed = True
        return False
    return True

def unpin_memory(module):
    if get_pin(module) is None:
        return 0
    size = module._pin.numel() * module._pin.element_size()
    comfy.model_management.unpin_memory(module._pin)
    del module._pin
    return size

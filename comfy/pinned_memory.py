import comfy.model_management
import comfy.memory_management
import comfy_aimdo.host_buffer
import comfy_aimdo.torch

from comfy.cli_args import args

def get_pin(module):
    return getattr(module, "_pin", None)

def pin_memory(module):
    if module.pin_failed or args.disable_pinned_memory or get_pin(module) is not None:
        return
    #FIXME: This is a RAM cache trigger event
    size = comfy.memory_management.vram_aligned_size([ module.weight, module.bias ])

    if comfy.model_management.MAX_PINNED_MEMORY <= 0 or (comfy.model_management.TOTAL_PINNED_MEMORY + size) > comfy.model_management.MAX_PINNED_MEMORY:
        module.pin_failed = True
        return False

    try:
        hostbuf = comfy_aimdo.host_buffer.HostBuffer(size)
    except RuntimeError:
        module.pin_failed = True
        return False

    module._pin = comfy_aimdo.torch.hostbuf_to_tensor(hostbuf)
    module._pin_hostbuf = hostbuf
    comfy.model_management.TOTAL_PINNED_MEMORY += size
    return True

def unpin_memory(module):
    if get_pin(module) is None:
        return 0
    size = module._pin.numel() * module._pin.element_size()

    comfy.model_management.TOTAL_PINNED_MEMORY -= size
    if comfy.model_management.TOTAL_PINNED_MEMORY < 0:
        comfy.model_management.TOTAL_PINNED_MEMORY = 0

    del module._pin
    del module._pin_hostbuf
    return size

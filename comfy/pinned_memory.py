import comfy.model_management
import comfy.memory_management
import comfy_aimdo.host_buffer
import comfy_aimdo.torch

from comfy.cli_args import args

def get_pin(module):
    return getattr(module, "_pin", None)

def pin_memory(module):
    pin_state = module._pin_state
    if pin_state["failed"] or args.disable_pinned_memory or get_pin(module) is not None:
        return

    hostbuf = pin_state["hostbuf"]
    size = comfy.memory_management.vram_aligned_size([ module.weight, module.bias ])
    offset = hostbuf.size
    if comfy.model_management.MAX_PINNED_MEMORY <= 0 or (comfy.model_management.TOTAL_PINNED_MEMORY + size) > comfy.model_management.MAX_PINNED_MEMORY:
        pin_state["failed"] = True
        return False

    try:
        hostbuf.extend(size=size)
    except RuntimeError:
        pin_state["failed"] = True
        return False

    module._pin = comfy_aimdo.torch.hostbuf_to_tensor(hostbuf)[offset:offset + size]
    module._pin.untyped_storage()._comfy_hostbuf = hostbuf
    pin_state["stack"].append((module, offset))
    comfy.model_management.TOTAL_PINNED_MEMORY += size
    return True

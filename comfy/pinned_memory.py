import comfy.model_management
import comfy.memory_management
import comfy_aimdo.host_buffer
import comfy_aimdo.torch

from comfy.cli_args import args

def get_pin(module, subset="weights"):
    return getattr(module, "_pin", None)

def pin_memory(module, subset="weights", size=None):
    pin_state = module._pin_state
    if pin_state["failed"] or args.disable_pinned_memory or get_pin(module, subset) is not None:
        return

    hostbuf, stack = pin_state[subset]
    if size is None:
        size = comfy.memory_management.vram_aligned_size([ module.weight, module.bias ])
    offset = hostbuf.size
    comfy.model_management.ensure_pin_budget(size)

    try:
        hostbuf.extend(size=size)
    except RuntimeError:
        pin_state["failed"] = True
        return False

    module._pin = comfy_aimdo.torch.hostbuf_to_tensor(hostbuf)[offset:offset + size]
    module._pin.untyped_storage()._comfy_hostbuf = hostbuf
    stack.append((module, offset))
    comfy.model_management.TOTAL_PINNED_MEMORY += size
    return True

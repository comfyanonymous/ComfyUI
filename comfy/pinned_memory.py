import comfy.model_management
import comfy.memory_management
import comfy_aimdo.host_buffer
import comfy_aimdo.torch
import torch

from comfy.cli_args import args

def get_pin(module, subset="weights"):
    return getattr(module, "_pin", None)

def pin_memory(module, subset="weights", size=None):
    pin_state = module._pin_state
    if pin_state["failed"] or args.disable_pinned_memory:
        return

    hostbuf, stack, stack_split = pin_state[subset]
    pin = get_pin(module, subset)
    if pin is not None:
        if module._pin_registered:
            return

        size = module._pin.nbytes
        comfy.model_management.ensure_pin_registerable(size)

        if torch.cuda.cudart().cudaHostRegister(module._pin.data_ptr(), size, 1) != 0:
            comfy.model_management.discard_cuda_async_error()
            return False
        module._pin_registered = True
        stack_split[0] = max(stack_split[0], module._pin_stack_index)
        comfy.model_management.TOTAL_PINNED_MEMORY += size
        return True

    if size is None:
        size = comfy.memory_management.vram_aligned_size([ module.weight, module.bias ])
    offset = hostbuf.size

    comfy.memory_management.extra_ram_release(comfy.memory_management.RAM_CACHE_HEADROOM)
    comfy.model_management.ensure_pin_budget(size)
    comfy.model_management.ensure_pin_registerable(size)

    try:
        hostbuf.extend(size=size)
    except RuntimeError:
        pin_state["failed"] = True
        return False

    module._pin = comfy_aimdo.torch.hostbuf_to_tensor(hostbuf)[offset:offset + size]
    module._pin.untyped_storage()._comfy_hostbuf = hostbuf
    stack.append((module, offset))
    module._pin_registered = True
    module._pin_stack_index = len(stack) - 1
    stack_split[0] = max(stack_split[0], module._pin_stack_index)
    comfy.model_management.TOTAL_MODEL_MEMORY += size
    comfy.model_management.TOTAL_PINNED_MEMORY += size
    return True

import bisect

import comfy.model_management
import comfy.memory_management
import comfy.pin_order
import comfy.utils
import comfy_aimdo.host_buffer
import comfy_aimdo.torch
import torch

from comfy.cli_args import args


PIN_SCHEDULER_STATS = {
    "register_failures": 0,
    "pageable_prefetches": 0,
    "evicted": 0,
}


def _host_register(pin, size):
    return torch.cuda.cudart().cudaHostRegister(pin.data_ptr(), size, 1)


def _host_unregister(pin):
    return torch.cuda.cudart().cudaHostUnregister(pin.data_ptr())


def _pin_device(module):
    return module._pin_state.get("device")


def copy_prefetch_order(source, target):
    order = getattr(source, "_pin_prefetch_order", None)
    if order is not None:
        target._pin_prefetch_order = order


def _in_flight(module_pin):
    event = module_pin.get("in_flight")
    if event is None:
        return False
    if not event.query():
        return True
    del module_pin["in_flight"]
    return False


def pin_eviction_state(module, subset):
    module_pin = module._pins[subset]
    state = comfy.pin_order.prefetch_pin_state(module)
    protected = _in_flight(module_pin) or (state is not None and state[0])
    return protected, state


def mark_modules_in_flight(comfy_modules, stream):
    if stream is None:
        return
    event = torch.cuda.Event()
    event.record(stream)
    for module in comfy_modules:
        sources = [module]
        for param_key in ("weight", "bias"):
            lowvram_source = getattr(module, param_key + "_lowvram_function", None)
            if lowvram_source is not None:
                sources.append(lowvram_source)
        for source in sources:
            for module_pin in source.__dict__.get("_pins", {}).values():
                if module_pin.get("registered"):
                    module_pin["in_flight"] = event


def _registered(module, subset, pin, size):
    module_pin = module._pins[subset]
    _, _, stack_split, pinned_size, *_ = module._pin_state[subset]
    module_pin["registered"] = True
    stack_split[0] = max(stack_split[0], module_pin["stack_index"])
    comfy.model_management.TOTAL_PINNED_MEMORY += size
    pinned_size[0] += size


def unregister_pin(module, subset, ignore_prefetch=False):
    module_pin = module._pins[subset]
    if not module_pin["registered"]:
        return 0
    protected, _ = pin_eviction_state(module, subset)
    if protected and not ignore_prefetch:
        return 0
    pin = module_pin["pin"]
    size = pin.nbytes
    if _host_unregister(pin) != 0:
        comfy.model_management.discard_cuda_async_error()
        return 0

    module_pin["registered"] = False
    _, stack, stack_split, pinned_size, *_ = module._pin_state[subset]
    while stack_split[0] >= 0 and not stack[stack_split[0]][0]._pins[subset]["registered"]:
        stack_split[0] -= 1
    comfy.model_management.TOTAL_PINNED_MEMORY = max(0, comfy.model_management.TOTAL_PINNED_MEMORY - size)
    pinned_size[0] = max(0, pinned_size[0] - size)
    PIN_SCHEDULER_STATS["evicted"] += size
    return size

def _add_to_bucket(module, module_pin, buckets, size, priority):
    bucket = buckets.setdefault(size, [])
    entry = [-priority, 0, module]
    entry[1] = id(entry)
    bisect.insort(bucket, entry)
    module_pin["balancer_entry"] = entry

def _steal_pin(module, stack, buckets, size, priority, subset):
    bucket = buckets.get(size)
    if bucket is None:
        return False

    while bucket and bucket[-1][-1] is None:
        bucket.pop()
    if not bucket:
        del buckets[size]
        return False

    incoming_state = comfy.pin_order.prefetch_pin_state(module)
    victim_index = None
    victim_distance = -1
    for index in range(len(bucket) - 1, -1, -1):
        *_, candidate = bucket[index]
        if candidate is None:
            continue
        protected, candidate_state = pin_eviction_state(candidate, subset)
        if protected:
            continue
        ordered_reuse = incoming_state is not None and incoming_state[0] and candidate_state is not None
        if not ordered_reuse and priority <= -bucket[index][0]:
            continue
        distance = -1 if candidate_state is None else candidate_state[1]
        if victim_index is None or distance > victim_distance:
            victim_index = index
            victim_distance = distance
    if victim_index is None:
        return False

    *_, victim = bucket.pop(victim_index)
    module_pin = module._pins[subset]
    victim_pin = victim._pins[subset]
    module_pin["pin"] = victim_pin["pin"]
    module_pin["registered"] = victim_pin["registered"]
    module_pin["stack_index"] = victim_pin["stack_index"]
    stack_index = module_pin["stack_index"]
    stack[stack_index] = (module, stack[stack_index][1])

    victim_pin["registered"] = False
    del victim_pin["pin"]
    del victim_pin["stack_index"]
    del victim_pin["balancer_entry"]

    _add_to_bucket(module, module_pin, buckets, size, priority)
    return True

def get_pin(module, subset="weights"):
    pins = module.__dict__.get("_pins")
    module_pin = None if pins is None else pins.get(subset)
    pin = None if module_pin is None else module_pin.get("pin")
    if pin is None or args.disable_pinned_memory:
        return pin

    size = pin.nbytes
    device = _pin_device(module)
    protected = {(module, subset)}
    if module_pin["registered"]:
        if not comfy.model_management.has_live_pin_budget(device):
            return pin
        if comfy.pin_order.prefetch_budget_checked(module):
            return pin
        if comfy.model_management.ensure_pin_registerable(0, device=device, protected=protected):
            return pin
        if not _in_flight(module_pin):
            if unregister_pin(module, subset, ignore_prefetch=True):
                PIN_SCHEDULER_STATS["pageable_prefetches"] += 1
        return pin

    live_budget = comfy.model_management.has_live_pin_budget(device)
    registerable = comfy.model_management.ensure_pin_registerable(size, device=device, protected=protected)
    if live_budget and not registerable:
        PIN_SCHEDULER_STATS["pageable_prefetches"] += 1
        return pin

    if _host_register(pin, size) != 0:
        PIN_SCHEDULER_STATS["register_failures"] += 1
        comfy.model_management.discard_cuda_async_error()
        if not live_budget:
            return pin
        comfy.model_management.free_registrations(size, protected=protected, prefetch_first=True)
        if _host_register(pin, size) != 0:
            PIN_SCHEDULER_STATS["register_failures"] += 1
            comfy.model_management.discard_cuda_async_error()
            PIN_SCHEDULER_STATS["pageable_prefetches"] += 1
            return pin

    _registered(module, subset, pin, size)
    return pin

def pin_memory(module, subset="weights", size=None):
    pin_state = module._pin_state
    if args.disable_pinned_memory:
        return

    pin = get_pin(module, subset)
    if pin is not None:
        return

    pins = module.__dict__.setdefault("_pins", {})
    module_pin = pins.setdefault(subset, {})
    hostbuf, stack, _, _, counter, buckets = pin_state[subset]
    if size is None:
        size = comfy.memory_management.vram_aligned_size([ module.weight, module.bias ])
    registerable_size = size
    loaded = subset.endswith("-loaded")
    priority = module_pin.get("balancer_priority")

    if priority is None:
        priority = comfy.utils.bit_reverse_range(counter[0], 16)
        counter[0] += 1
        module_pin["balancer_priority"] = priority

    comfy.memory_management.extra_ram_release(comfy.memory_management.RAM_CACHE_HEADROOM)
    protected = {(module, subset)}
    if (not comfy.model_management.ensure_pin_budget(size, loaded=loaded, device=_pin_device(module), protected=protected) or
        not comfy.model_management.ensure_pin_registerable(registerable_size, device=_pin_device(module), protected=protected)):
        return _steal_pin(module, stack, buckets, size, priority, subset)

    offset = hostbuf.size
    extended = False
    try:
        hostbuf.extend(size=size, register=False)
        extended = True
        pin = comfy_aimdo.torch.hostbuf_to_tensor(hostbuf)[offset:offset + size]
        pin.untyped_storage()._comfy_hostbuf = hostbuf
        if _host_register(pin, size) != 0:
            PIN_SCHEDULER_STATS["register_failures"] += 1
            comfy.model_management.discard_cuda_async_error()
            comfy.model_management.free_registrations(size, protected={(module, subset)}, prefetch_first=True)
            if _host_register(pin, size) != 0:
                PIN_SCHEDULER_STATS["register_failures"] += 1
                comfy.model_management.discard_cuda_async_error()
                del pin
                hostbuf.truncate(offset, do_unregister=False)
                return _steal_pin(module, stack, buckets, size, priority, subset)
    except RuntimeError:
        if extended:
            hostbuf.truncate(offset, do_unregister=False)
        return _steal_pin(module, stack, buckets, size, priority, subset)

    module_pin["pin"] = pin
    stack.append((module, offset))
    module_pin["stack_index"] = len(stack) - 1
    _registered(module, subset, pin, size)
    _add_to_bucket(module, module_pin, buckets, size, priority)
    return True

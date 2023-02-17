
CPU = 0
NO_VRAM = 1
LOW_VRAM = 2
NORMAL_VRAM = 3

accelerate_enabled = False
vram_state = NORMAL_VRAM

total_vram = 0
total_vram_available_mb = -1

import sys

set_vram_to = NORMAL_VRAM

try:
    import torch
    total_vram = torch.cuda.mem_get_info(torch.cuda.current_device())[1] / (1024 * 1024)
    if total_vram <= 4096 and not "--normalvram" in sys.argv:
        print("Trying to enable lowvram mode because your GPU seems to have 4GB or less. If you don't want this use: --normalvram")
        set_vram_to = LOW_VRAM
except:
    pass

if "--lowvram" in sys.argv:
    set_vram_to = LOW_VRAM
if "--novram" in sys.argv:
    set_vram_to = NO_VRAM



if set_vram_to != NORMAL_VRAM:
    try:
        import accelerate
        accelerate_enabled = True
        vram_state = set_vram_to
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print("ERROR: COULD NOT ENABLE LOW VRAM MODE.")

    total_vram_available_mb = (total_vram - 1024) // 2
    total_vram_available_mb = int(max(256, total_vram_available_mb))


print("Set vram state to:", ["CPU", "NO VRAM", "LOW VRAM", "NORMAL VRAM"][vram_state])


current_loaded_model = None
current_gpu_controlnets = []

model_accelerated = False


def unload_model():
    global current_loaded_model
    global model_accelerated
    global current_gpu_controlnets
    if current_loaded_model is not None:
        if model_accelerated:
            accelerate.hooks.remove_hook_from_submodules(current_loaded_model.model)
            model_accelerated = False

        current_loaded_model.model.cpu()
        current_loaded_model.unpatch_model()
        current_loaded_model = None
    if len(current_gpu_controlnets) > 0:
        for n in current_gpu_controlnets:
            n.cpu()
        current_gpu_controlnets = []


def load_model_gpu(model):
    global current_loaded_model
    global vram_state
    global model_accelerated

    if model is current_loaded_model:
        return
    unload_model()
    try:
        real_model = model.patch_model()
    except Exception as e:
        model.unpatch_model()
        raise e
    current_loaded_model = model
    if vram_state == CPU:
        pass
    elif vram_state == NORMAL_VRAM:
        model_accelerated = False
        real_model.cuda()
    else:
        if vram_state == NO_VRAM:
            device_map = accelerate.infer_auto_device_map(real_model, max_memory={0: "256MiB", "cpu": "16GiB"})
        elif vram_state == LOW_VRAM:
            device_map = accelerate.infer_auto_device_map(real_model, max_memory={0: "{}MiB".format(total_vram_available_mb), "cpu": "16GiB"})

        accelerate.dispatch_model(real_model, device_map=device_map, main_device="cuda")
        model_accelerated = True
    return current_loaded_model

def load_controlnet_gpu(models):
    global current_gpu_controlnets
    global vram_state

    if vram_state == LOW_VRAM or vram_state == NO_VRAM:
        #don't load controlnets like this if low vram because they will be loaded right before running and unloaded right after
        return

    for m in current_gpu_controlnets:
        if m not in models:
            m.cpu()

    current_gpu_controlnets = []
    for m in models:
        current_gpu_controlnets.append(m.cuda())


def load_if_low_vram(model):
    global vram_state
    if vram_state == LOW_VRAM or vram_state == NO_VRAM:
        return model.cuda()
    return model

def unload_if_low_vram(model):
    global vram_state
    if vram_state == LOW_VRAM or vram_state == NO_VRAM:
        return model.cpu()
    return model


def get_free_memory():
    dev = torch.cuda.current_device()
    stats = torch.cuda.memory_stats(dev)
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
    mem_free_torch = mem_reserved - mem_active
    return mem_free_cuda + mem_free_torch

def maximum_batch_area():
    global vram_state
    if vram_state == NO_VRAM:
        return 0

    memory_free = get_free_memory() / (1024 * 1024)
    area = ((memory_free - 1024) * 0.9) / (0.6)
    return int(max(area, 0))


CPU = 0
NO_VRAM = 1
LOW_VRAM = 2
NORMAL_VRAM = 3

accelerate_enabled = False
vram_state = NORMAL_VRAM

total_vram_available_mb = -1

import sys

set_vram_to = NORMAL_VRAM
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
    try:
        import torch
        total_vram_available_mb = torch.cuda.mem_get_info(torch.cuda.current_device())[1] / (1024 * 1024)
    except:
        pass
    total_vram_available_mb = (total_vram_available_mb - 1024) // 2
    total_vram_available_mb = int(max(256, total_vram_available_mb))


print("Set vram state to:", ["CPU", "NO VRAM", "LOW VRAM", "NORMAL VRAM"][vram_state])


current_loaded_model = None


model_accelerated = False


def unload_model():
    global current_loaded_model
    global model_accelerated
    if current_loaded_model is not None:
        if model_accelerated:
            accelerate.hooks.remove_hook_from_submodules(current_loaded_model.model)
            model_accelerated = False

        current_loaded_model.model.cpu()
        current_loaded_model.unpatch_model()
        current_loaded_model = None


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

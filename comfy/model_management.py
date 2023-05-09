import psutil
from enum import Enum
from comfy.cli_args import args

class VRAMState(Enum):
    CPU = 0
    NO_VRAM = 1
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    MPS = 5

# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM

total_vram = 0
total_vram_available_mb = -1

accelerate_enabled = False
xpu_available = False

directml_enabled = False
if args.directml is not None:
    import torch_directml
    directml_enabled = True
    device_index = args.directml
    if device_index < 0:
        directml_device = torch_directml.device()
    else:
        directml_device = torch_directml.device(device_index)
    print("Using directml with device:", torch_directml.device_name(device_index))
    # torch_directml.disable_tiled_resources(True)

try:
    import torch
    if directml_enabled:
        total_vram = 4097 #TODO
    else:
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                xpu_available = True
                total_vram = torch.xpu.get_device_properties(torch.xpu.current_device()).total_memory / (1024 * 1024)
        except:
            total_vram = torch.cuda.mem_get_info(torch.cuda.current_device())[1] / (1024 * 1024)
    total_ram = psutil.virtual_memory().total / (1024 * 1024)
    if not args.normalvram and not args.cpu:
        if total_vram <= 4096:
            print("Trying to enable lowvram mode because your GPU seems to have 4GB or less. If you don't want this use: --normalvram")
            set_vram_to = VRAMState.LOW_VRAM
        elif total_vram > total_ram * 1.1 and total_vram > 14336:
            print("Enabling highvram mode because your GPU has more vram than your computer has ram. If you don't want this use: --normalvram")
            vram_state = VRAMState.HIGH_VRAM
except:
    pass

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if args.disable_xformers:
    XFORMERS_IS_AVAILABLE = False
else:
    try:
        import xformers
        import xformers.ops
        XFORMERS_IS_AVAILABLE = True
        try:
            XFORMERS_VERSION = xformers.version.__version__
            print("xformers version:", XFORMERS_VERSION)
            if XFORMERS_VERSION.startswith("0.0.18"):
                print()
                print("WARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
                print("Please downgrade or upgrade xformers to a different version.")
                print()
                XFORMERS_ENABLED_VAE = False
        except:
            pass
    except:
        XFORMERS_IS_AVAILABLE = False

ENABLE_PYTORCH_ATTENTION = args.use_pytorch_cross_attention
if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    XFORMERS_IS_AVAILABLE = False

if args.lowvram:
    set_vram_to = VRAMState.LOW_VRAM
elif args.novram:
    set_vram_to = VRAMState.NO_VRAM
elif args.highvram:
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
if args.force_fp32:
    print("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True


if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
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

try:
    if torch.backends.mps.is_available():
        vram_state = VRAMState.MPS
except:
    pass

if args.cpu:
    vram_state = VRAMState.CPU

print(f"Set vram state to: {vram_state.name}")


current_loaded_model = None
current_gpu_controlnets = []

model_accelerated = False


def unload_model():
    global current_loaded_model
    global model_accelerated
    global current_gpu_controlnets
    global vram_state

    if current_loaded_model is not None:
        if model_accelerated:
            accelerate.hooks.remove_hook_from_submodules(current_loaded_model.model)
            model_accelerated = False

        #never unload models from GPU on high vram
        if vram_state != VRAMState.HIGH_VRAM:
            current_loaded_model.model.cpu()
            current_loaded_model.model_patches_to("cpu")
        current_loaded_model.unpatch_model()
        current_loaded_model = None

    if vram_state != VRAMState.HIGH_VRAM:
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

    model.model_patches_to(get_torch_device())
    current_loaded_model = model
    if vram_state == VRAMState.CPU:
        pass
    elif vram_state == VRAMState.MPS:
        mps_device = torch.device("mps")
        real_model.to(mps_device)
        pass
    elif vram_state == VRAMState.NORMAL_VRAM or vram_state == VRAMState.HIGH_VRAM:
        model_accelerated = False
        real_model.to(get_torch_device())
    else:
        if vram_state == VRAMState.NO_VRAM:
            device_map = accelerate.infer_auto_device_map(real_model, max_memory={0: "256MiB", "cpu": "16GiB"})
        elif vram_state == VRAMState.LOW_VRAM:
            device_map = accelerate.infer_auto_device_map(real_model, max_memory={0: "{}MiB".format(total_vram_available_mb), "cpu": "16GiB"})

        accelerate.dispatch_model(real_model, device_map=device_map, main_device=get_torch_device())
        model_accelerated = True
    return current_loaded_model

def load_controlnet_gpu(control_models):
    global current_gpu_controlnets
    global vram_state
    if vram_state == VRAMState.CPU:
        return

    if vram_state == VRAMState.LOW_VRAM or vram_state == VRAMState.NO_VRAM:
        for m in control_models:
            if hasattr(m, 'set_lowvram'):
                m.set_lowvram(True)
        #don't load controlnets like this if low vram because they will be loaded right before running and unloaded right after
        return

    models = []
    for m in control_models:
        models += m.get_models()

    for m in current_gpu_controlnets:
        if m not in models:
            m.cpu()

    device = get_torch_device()
    current_gpu_controlnets = []
    for m in models:
        current_gpu_controlnets.append(m.to(device))


def load_if_low_vram(model):
    global vram_state
    if vram_state == VRAMState.LOW_VRAM or vram_state == VRAMState.NO_VRAM:
        return model.to(get_torch_device())
    return model

def unload_if_low_vram(model):
    global vram_state
    if vram_state == VRAMState.LOW_VRAM or vram_state == VRAMState.NO_VRAM:
        return model.cpu()
    return model

def get_torch_device():
    global xpu_available
    global directml_enabled
    if directml_enabled:
        global directml_device
        return directml_device
    if vram_state == VRAMState.MPS:
        return torch.device("mps")
    if vram_state == VRAMState.CPU:
        return torch.device("cpu")
    else:
        if xpu_available:
            return torch.device("xpu")
        else:
            return torch.cuda.current_device()

def get_autocast_device(dev):
    if hasattr(dev, 'type'):
        return dev.type
    return "cuda"


def xformers_enabled():
    global xpu_available
    global directml_enabled
    if vram_state == VRAMState.CPU:
        return False
    if xpu_available:
        return False
    if directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae():
    enabled = xformers_enabled()
    if not enabled:
        return False

    return XFORMERS_ENABLED_VAE

def pytorch_attention_enabled():
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION

def pytorch_attention_flash_attention():
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        #TODO: more reliable way of checking for flash attention?
        if torch.version.cuda: #pytorch flash attention only works on Nvidia
            return True
    return False

def get_free_memory(dev=None, torch_free_too=False):
    global xpu_available
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024 #TODO
            mem_free_torch = mem_free_total
        elif xpu_available:
            mem_free_total = torch.xpu.get_device_properties(dev).total_memory - torch.xpu.memory_allocated(dev)
            mem_free_torch = mem_free_total
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total

def maximum_batch_area():
    global vram_state
    if vram_state == VRAMState.NO_VRAM:
        return 0

    memory_free = get_free_memory() / (1024 * 1024)
    if xformers_enabled() or pytorch_attention_flash_attention():
        #TODO: this needs to be tweaked
        area = 20 * memory_free
    else:
        #TODO: this formula is because AMD sucks and has memory management issues which might be fixed in the future
        area = ((memory_free - 1024) * 0.9) / (0.6)
    return int(max(area, 0))

def cpu_mode():
    global vram_state
    return vram_state == VRAMState.CPU

def mps_mode():
    global vram_state
    return vram_state == VRAMState.MPS

def should_use_fp16():
    global xpu_available
    global directml_enabled

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if cpu_mode() or mps_mode() or xpu_available:
        return False #TODO ?

    if torch.cuda.is_bf16_supported():
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major < 7:
        return False

    #FP32 is faster on those cards?
    nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600"]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True

def soft_empty_cache():
    global xpu_available
    if xpu_available:
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if torch.version.cuda: #This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

#TODO: might be cleaner to put this somewhere else
import threading

class InterruptProcessingException(Exception):
    pass

interrupt_processing_mutex = threading.RLock()

interrupt_processing = False
def interrupt_current_processing(value=True):
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        interrupt_processing = value

def processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        return interrupt_processing

def throw_exception_if_processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()

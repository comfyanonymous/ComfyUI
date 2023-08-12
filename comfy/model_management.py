import psutil
from enum import Enum
from comfy.cli_args import args
import torch

class VRAMState(Enum):
    DISABLED = 0    #No vram present: no need to move models to vram
    NO_VRAM = 1     #Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5      #No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0

lowvram_available = True
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
    lowvram_available = False #TODO: need to find a way to get free memory in directml before this can be enabled by default.

try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        xpu_available = True
except:
    pass

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass

if args.cpu:
    cpu_state = CPUState.CPU

def get_torch_device():
    global xpu_available
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if xpu_available:
            return torch.device("xpu")
        else:
            return torch.device(torch.cuda.current_device())

def get_total_memory(dev=None, torch_total_too=False):
    global xpu_available
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        if directml_enabled:
            mem_total = 1024 * 1024 * 1024 #TODO
            mem_total_torch = mem_total
        elif xpu_available:
            mem_total = torch.xpu.get_device_properties(dev).total_memory
            mem_total_torch = mem_total
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total

total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
print("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))
if not args.normalvram and not args.cpu:
    if lowvram_available and total_vram <= 4096:
        print("Trying to enable lowvram mode because your GPU seems to have 4GB or less. If you don't want this use: --normalvram")
        set_vram_to = VRAMState.LOW_VRAM
    elif total_vram > total_ram * 1.1 and total_vram > 14336:
        print("Enabling highvram mode because your GPU has more vram than your computer has ram. If you don't want this use: --normalvram")
        vram_state = VRAMState.HIGH_VRAM

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

def is_nvidia():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True

ENABLE_PYTORCH_ATTENTION = args.use_pytorch_cross_attention

if ENABLE_PYTORCH_ATTENTION == False and XFORMERS_IS_AVAILABLE == False and args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
    try:
        if is_nvidia():
            torch_version = torch.version.__version__
            if int(torch_version[0]) >= 2:
                ENABLE_PYTORCH_ATTENTION = True
    except:
        pass

if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    XFORMERS_IS_AVAILABLE = False

if args.lowvram:
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
elif args.novram:
    set_vram_to = VRAMState.NO_VRAM
elif args.highvram or args.gpu_only:
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
FORCE_FP16 = False
if args.force_fp32:
    print("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if args.force_fp16:
    print("Forcing FP16.")
    FORCE_FP16 = True

if lowvram_available:
    try:
        import accelerate
        if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
            vram_state = set_vram_to
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print("ERROR: LOW VRAM MODE NEEDS accelerate.")
        lowvram_available = False


if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

print(f"Set vram state to: {vram_state.name}")


def get_torch_device_name(device):
    if hasattr(device, 'type'):
        if device.type == "cuda":
            try:
                allocator_backend = torch.cuda.get_allocator_backend()
            except:
                allocator_backend = ""
            return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
        else:
            return "{}".format(device.type)
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))

try:
    print("Device:", get_torch_device_name(get_torch_device()))
except:
    print("Could not pick default device.")


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

        current_loaded_model.unpatch_model()
        current_loaded_model.model.to(current_loaded_model.offload_device)
        current_loaded_model.model_patches_to(current_loaded_model.offload_device)
        current_loaded_model = None
        if vram_state != VRAMState.HIGH_VRAM:
            soft_empty_cache()

    if vram_state != VRAMState.HIGH_VRAM:
        if len(current_gpu_controlnets) > 0:
            for n in current_gpu_controlnets:
                n.cpu()
            current_gpu_controlnets = []

def minimum_inference_memory():
    return (768 * 1024 * 1024)

def load_model_gpu(model):
    global current_loaded_model
    global vram_state
    global model_accelerated

    if model is current_loaded_model:
        return
    unload_model()

    torch_dev = model.load_device
    model.model_patches_to(torch_dev)
    model.model_patches_to(model.model_dtype())
    current_loaded_model = model

    if is_device_cpu(torch_dev):
        vram_set_state = VRAMState.DISABLED
    else:
        vram_set_state = vram_state

    if lowvram_available and (vram_set_state == VRAMState.LOW_VRAM or vram_set_state == VRAMState.NORMAL_VRAM):
        model_size = model.model_size()
        current_free_mem = get_free_memory(torch_dev)
        lowvram_model_memory = int(max(256 * (1024 * 1024), (current_free_mem - 1024 * (1024 * 1024)) / 1.3 ))
        if model_size > (current_free_mem - minimum_inference_memory()): #only switch to lowvram if really necessary
            vram_set_state = VRAMState.LOW_VRAM

    real_model = model.model
    patch_model_to = None
    if vram_set_state == VRAMState.DISABLED:
        pass
    elif vram_set_state == VRAMState.NORMAL_VRAM or vram_set_state == VRAMState.HIGH_VRAM or vram_set_state == VRAMState.SHARED:
        model_accelerated = False
        patch_model_to = torch_dev

    try:
        real_model = model.patch_model(device_to=patch_model_to)
    except Exception as e:
        model.unpatch_model()
        unload_model()
        raise e

    if patch_model_to is not None:
        real_model.to(torch_dev)

    if vram_set_state == VRAMState.NO_VRAM:
        device_map = accelerate.infer_auto_device_map(real_model, max_memory={0: "256MiB", "cpu": "16GiB"})
        accelerate.dispatch_model(real_model, device_map=device_map, main_device=torch_dev)
        model_accelerated = True
    elif vram_set_state == VRAMState.LOW_VRAM:
        device_map = accelerate.infer_auto_device_map(real_model, max_memory={0: "{}MiB".format(lowvram_model_memory // (1024 * 1024)), "cpu": "16GiB"})
        accelerate.dispatch_model(real_model, device_map=device_map, main_device=torch_dev)
        model_accelerated = True

    return current_loaded_model

def load_controlnet_gpu(control_models):
    global current_gpu_controlnets
    global vram_state
    if vram_state == VRAMState.DISABLED:
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

def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")

def text_encoder_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def text_encoder_device():
    if args.gpu_only:
        return get_torch_device()
    elif vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.NORMAL_VRAM:
        #NOTE: on a Ryzen 5 7600X with 4080 it's faster to shift to GPU
        if torch.get_num_threads() < 8: #leaving the text encoder on the CPU is faster than shifting it if the CPU is fast enough.
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")

def vae_device():
    return get_torch_device()

def vae_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def vae_dtype():
    if args.fp16_vae:
        return torch.float16
    elif args.bf16_vae:
        return torch.bfloat16
    else:
        return torch.float32

def get_autocast_device(dev):
    if hasattr(dev, 'type'):
        return dev.type
    return "cuda"


def xformers_enabled():
    global xpu_available
    global directml_enabled
    global cpu_state
    if cpu_state != CPUState.GPU:
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
        if is_nvidia(): #pytorch flash attention only works on Nvidia
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
    global cpu_state
    return cpu_state == CPUState.CPU

def mps_mode():
    global cpu_state
    return cpu_state == CPUState.MPS

def is_device_cpu(device):
    if hasattr(device, 'type'):
        if (device.type == 'cpu'):
            return True
    return False

def is_device_mps(device):
    if hasattr(device, 'type'):
        if (device.type == 'mps'):
            return True
    return False

def should_use_fp16(device=None, model_params=0):
    global xpu_available
    global directml_enabled

    if FORCE_FP16:
        return True

    if device is not None: #TODO
        if is_device_cpu(device) or is_device_mps(device):
            return False

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if cpu_mode() or mps_mode() or xpu_available:
        return False #TODO ?

    if torch.cuda.is_bf16_supported():
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major < 6:
        return False

    fp16_works = False
    #FP16 is confirmed working on a 1080 (GP104) but it's a bit slower than FP32 so it should only be enabled
    #when the model doesn't actually fit on the card
    #TODO: actually test if GP106 and others have the same type of behavior
    nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050"]
    for x in nvidia_10_series:
        if x in props.name.lower():
            fp16_works = True

    if fp16_works:
        free_model_memory = (get_free_memory() * 0.9 - minimum_inference_memory())
        if model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    #FP16 is just broken on these cards
    nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX"]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True

def soft_empty_cache():
    global xpu_available
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif xpu_available:
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if is_nvidia(): #This seems to make things worse on ROCm so I only do it for cuda
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

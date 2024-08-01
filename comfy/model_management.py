import psutil
import logging
from enum import Enum
from comfy.cli_args import args
import torch
import sys
import platform

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

if args.deterministic:
    logging.info("Using deterministic algorithms for pytorch")
    torch.use_deterministic_algorithms(True, warn_only=True)

directml_enabled = False
if args.directml is not None:
    import torch_directml
    directml_enabled = True
    device_index = args.directml
    if device_index < 0:
        directml_device = torch_directml.device()
    else:
        directml_device = torch_directml.device(device_index)
    logging.info("Using directml with device: {}".format(torch_directml.device_name(device_index)))
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

def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False

def get_torch_device():
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
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        else:
            return torch.device(torch.cuda.current_device())

def get_total_memory(dev=None, torch_total_too=False):
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
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            mem_total_torch = mem_reserved
            mem_total = torch.xpu.get_device_properties(dev).total_memory
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
logging.info("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

try:
    logging.info("pytorch version: {}".format(torch.version.__version__))
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
            XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
        except:
            pass
        try:
            XFORMERS_VERSION = xformers.version.__version__
            logging.info("xformers version: {}".format(XFORMERS_VERSION))
            if XFORMERS_VERSION.startswith("0.0.18"):
                logging.warning("\nWARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
                logging.warning("Please downgrade or upgrade xformers to a different version.\n")
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
    return False

ENABLE_PYTORCH_ATTENTION = False
if args.use_pytorch_cross_attention:
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False

VAE_DTYPES = [torch.float32]

try:
    if is_nvidia():
        torch_version = torch.version.__version__
        if int(torch_version[0]) >= 2:
            if ENABLE_PYTORCH_ATTENTION == False and args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
                ENABLE_PYTORCH_ATTENTION = True
            if torch.cuda.is_bf16_supported() and torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8:
                VAE_DTYPES = [torch.bfloat16] + VAE_DTYPES
    if is_intel_xpu():
        if args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
            ENABLE_PYTORCH_ATTENTION = True
except:
    pass

if is_intel_xpu():
    VAE_DTYPES = [torch.bfloat16] + VAE_DTYPES

if args.cpu_vae:
    VAE_DTYPES = [torch.float32]


if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

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
    logging.info("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if args.force_fp16:
    logging.info("Forcing FP16.")
    FORCE_FP16 = True

if lowvram_available:
    if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        vram_state = set_vram_to


if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

logging.info(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = args.disable_smart_memory

if DISABLE_SMART_MEMORY:
    logging.info("Disabling smart memory management")

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
    elif is_intel_xpu():
        return "{} {}".format(device, torch.xpu.get_device_name(device))
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))

try:
    logging.info("Device: {}".format(get_torch_device_name(get_torch_device())))
except:
    logging.warning("Could not pick default device.")


current_loaded_models = []

def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem

class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device
        self.weights_loaded = False
        self.real_model = None
        self.currently_used = True

    def model_memory(self):
        return self.model.model_size()

    def model_memory_required(self, device):
        if device == self.model.current_device:
            return 0
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        load_weights = not self.weights_loaded

        try:
            if lowvram_model_memory > 0 and load_weights:
                self.real_model = self.model.patch_model_lowvram(device_to=patch_model_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights)
            else:
                self.real_model = self.model.patch_model(device_to=patch_model_to, patch_weights=load_weights)
        except Exception as e:
            self.model.unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e

        if is_intel_xpu() and not args.disable_ipex_optimize:
            self.real_model = ipex.optimize(self.real_model.eval(), graph_mode=True, concat_linear=True)

        self.weights_loaded = True
        return self.real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter > 0:
            return True
        return False

    def model_unload(self, unpatch_weights=True):
        self.model.unpatch_model(self.model.offload_device, unpatch_weights=unpatch_weights)
        self.model.model_patches_to(self.model.offload_device)
        self.weights_loaded = self.weights_loaded and not unpatch_weights
        self.real_model = None

    def __eq__(self, other):
        return self.model is other.model

def minimum_inference_memory():
    return (1024 * 1024 * 1024) * 1.2

def unload_model_clones(model, unload_weights_only=True, force_unload=True):
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload

    if len(to_unload) == 0:
        return True

    same_weights = 0
    for i in to_unload:
        if model.clone_has_same_weights(current_loaded_models[i].model):
            same_weights += 1

    if same_weights == len(to_unload):
        unload_weight = False
    else:
        unload_weight = True

    if not force_unload:
        if unload_weights_only and unload_weight == False:
            return None

    for i in to_unload:
        logging.debug("unload clone {} {}".format(i, unload_weight))
        current_loaded_models.pop(i).model_unload(unpatch_weights=unload_weight)

    return unload_weight

def free_memory(memory_required, device, keep_loaded=[]):
    unloaded_model = []
    can_unload = []

    for i in range(len(current_loaded_models) -1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded:
                can_unload.append((sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
                shift_model.currently_used = False

    for x in sorted(can_unload):
        i = x[-1]
        if not DISABLE_SMART_MEMORY:
            if get_free_memory(device) > memory_required:
                break
        current_loaded_models[i].model_unload()
        unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        current_loaded_models.pop(i)

    if len(unloaded_model) > 0:
        soft_empty_cache()
    else:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()

def load_models_gpu(models, memory_required=0, force_patch_weights=False):
    global vram_state

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required)

    models = set(models)

    models_to_load = []
    models_already_loaded = []
    for x in models:
        loaded_model = LoadedModel(x)
        loaded = None

        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models[loaded_model_index]
            if loaded.should_reload_model(force_patch_weights=force_patch_weights): #TODO: cleanup this model reload logic
                current_loaded_models.pop(loaded_model_index).model_unload(unpatch_weights=True)
                loaded = None
            else:
                loaded.currently_used = True
                models_already_loaded.append(loaded)

        if loaded is None:
            if hasattr(x, "model"):
                logging.info(f"Requested to load {x.model.__class__.__name__}")
            models_to_load.append(loaded_model)

    if len(models_to_load) == 0:
        devs = set(map(lambda a: a.device, models_already_loaded))
        for d in devs:
            if d != torch.device("cpu"):
                free_memory(extra_mem, d, models_already_loaded)
        return

    logging.info(f"Loading {len(models_to_load)} new model{'s' if len(models_to_load) > 1 else ''}")

    total_memory_required = {}
    for loaded_model in models_to_load:
        if unload_model_clones(loaded_model.model, unload_weights_only=True, force_unload=False) == True:#unload clones where the weights are different
            total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(total_memory_required[device] * 1.3 + extra_mem, device, models_already_loaded)

    for loaded_model in models_to_load:
        weights_unloaded = unload_model_clones(loaded_model.model, unload_weights_only=False, force_unload=False) #unload the rest of the clones where the weights can stay loaded
        if weights_unloaded is not None:
            loaded_model.weights_loaded = not weights_unloaded

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state
        lowvram_model_memory = 0
        if lowvram_available and (vram_set_state == VRAMState.LOW_VRAM or vram_set_state == VRAMState.NORMAL_VRAM):
            model_size = loaded_model.model_memory_required(torch_dev)
            current_free_mem = get_free_memory(torch_dev)
            lowvram_model_memory = int(max(64 * (1024 * 1024), (current_free_mem - extra_mem)))
            if model_size <= (current_free_mem - inference_memory): #only switch to lowvram if really necessary
                lowvram_model_memory = 0

        if vram_set_state == VRAMState.NO_VRAM:
            lowvram_model_memory = 64 * 1024 * 1024

        cur_loaded_model = loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
        current_loaded_models.insert(0, loaded_model)
    return


def load_model_gpu(model):
    return load_models_gpu([model])

def loaded_models(only_currently_used=False):
    output = []
    for m in current_loaded_models:
        if only_currently_used:
            if not m.currently_used:
                continue

        output.append(m.model)
    return output

def cleanup_models(keep_clone_weights_loaded=False):
    to_delete = []
    for i in range(len(current_loaded_models)):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            if not keep_clone_weights_loaded:
                to_delete = [i] + to_delete
            #TODO: find a less fragile way to do this.
            elif sys.getrefcount(current_loaded_models[i].real_model) <= 3: #references from .real_model + the .model
                to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x

def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except: #Old pytorch doesn't have .itemsize
            pass
    return dtype_size

def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")

def unet_inital_load_device(parameters, dtype):
    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM:
        return torch_dev

    cpu_dev = torch.device("cpu")
    if DISABLE_SMART_MEMORY:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev

def unet_dtype(device=None, model_params=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    if args.bf16_unet:
        return torch.bfloat16
    if args.fp16_unet:
        return torch.float16
    if args.fp8_e4m3fn_unet:
        return torch.float8_e4m3fn
    if args.fp8_e5m2_unet:
        return torch.float8_e5m2
    if should_use_fp16(device=device, model_params=model_params, manual_cast=True):
        if torch.float16 in supported_dtypes:
            return torch.float16
    if should_use_bf16(device, model_params=model_params, manual_cast=True):
        if torch.bfloat16 in supported_dtypes:
            return torch.bfloat16
    return torch.float32

# None means no manual cast
def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    if weight_dtype == torch.float32:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
    if fp16_supported and weight_dtype == torch.float16:
        return None

    bf16_supported = should_use_bf16(inference_device)
    if bf16_supported and weight_dtype == torch.bfloat16:
        return None

    if fp16_supported and torch.float16 in supported_dtypes:
        return torch.float16

    elif bf16_supported and torch.bfloat16 in supported_dtypes:
        return torch.bfloat16
    else:
        return torch.float32

def text_encoder_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def text_encoder_device():
    if args.gpu_only:
        return get_torch_device()
    elif vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.NORMAL_VRAM:
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")

def text_encoder_dtype(device=None):
    if args.fp8_e4m3fn_text_enc:
        return torch.float8_e4m3fn
    elif args.fp8_e5m2_text_enc:
        return torch.float8_e5m2
    elif args.fp16_text_enc:
        return torch.float16
    elif args.fp32_text_enc:
        return torch.float32

    if is_device_cpu(device):
        return torch.float16

    return torch.float16


def intermediate_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def vae_device():
    if args.cpu_vae:
        return torch.device("cpu")
    return get_torch_device()

def vae_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def vae_dtype(device=None, allowed_dtypes=[]):
    global VAE_DTYPES
    if args.fp16_vae:
        return torch.float16
    elif args.bf16_vae:
        return torch.bfloat16
    elif args.fp32_vae:
        return torch.float32

    for d in allowed_dtypes:
        if d == torch.float16 and should_use_fp16(device, prioritize_performance=False):
            return d
        if d in VAE_DTYPES:
            return d

    return VAE_DTYPES[0]

def get_autocast_device(dev):
    if hasattr(dev, 'type'):
        return dev.type
    return "cuda"

def supports_dtype(device, dtype): #TODO
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False

def supports_cast(device, dtype): #TODO
    if dtype == torch.float32:
        return True
    if dtype == torch.float16:
        return True
    if directml_enabled: #TODO: test this
        return False
    if dtype == torch.bfloat16:
        return True
    if is_device_mps(device):
        return False
    if dtype == torch.float8_e4m3fn:
        return True
    if dtype == torch.float8_e5m2:
        return True
    return False

def pick_weight_dtype(dtype, fallback_dtype, device=None):
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype

    if not supports_cast(device, dtype):
        dtype = fallback_dtype

    return dtype

def device_supports_non_blocking(device):
    if is_device_mps(device):
        return False #pytorch bug? mps doesn't support non blocking
    if is_intel_xpu():
        return False
    if args.deterministic: #TODO: figure out why deterministic breaks non blocking from gpu to cpu (previews)
        return False
    if directml_enabled:
        return False
    return True

def device_should_use_non_blocking(device):
    if not device_supports_non_blocking(device):
        return False
    return False
    # return True #TODO: figure out why this causes memory issues on Nvidia and possibly others

def force_channels_last():
    if args.force_channels_last:
        return True

    #TODO
    return False

def cast_to_device(tensor, device, dtype, copy=False):
    device_supports_cast = False
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if hasattr(device, 'type') and device.type.startswith("cuda"):
            device_supports_cast = True
        elif is_intel_xpu():
            device_supports_cast = True

    non_blocking = device_should_use_non_blocking(device)

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
            return tensor.to(device, copy=copy, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
        else:
            return tensor.to(device, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
    else:
        return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)

def xformers_enabled():
    global directml_enabled
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
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
        if is_intel_xpu():
            return True
    return False

def force_upcast_attention_dtype():
    upcast = args.force_upcast_attention
    try:
        if platform.mac_ver()[0] in ['14.5']: #black image bug on OSX Sonoma 14.5
            upcast = True
    except:
        pass
    if upcast:
        return torch.float32
    else:
        return None

def get_free_memory(dev=None, torch_free_too=False):
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
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_torch = mem_reserved - mem_active
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_total = mem_free_xpu + mem_free_torch
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

def cpu_mode():
    global cpu_state
    return cpu_state == CPUState.CPU

def mps_mode():
    global cpu_state
    return cpu_state == CPUState.MPS

def is_device_type(device, type):
    if hasattr(device, 'type'):
        if (device.type == type):
            return True
    return False

def is_device_cpu(device):
    return is_device_type(device, 'cpu')

def is_device_mps(device):
    return is_device_type(device, 'mps')

def is_device_cuda(device):
    return is_device_type(device, 'cuda')

def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    global directml_enabled

    if device is not None:
        if is_device_cpu(device):
            return False

    if FORCE_FP16:
        return True

    if device is not None:
        if is_device_mps(device):
            return True

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return True

    if torch.version.hip:
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major >= 8:
        return True

    if props.major < 6:
        return False

    fp16_works = False
    #FP16 is confirmed working on a 1080 (GP104) but it's a bit slower than FP32 so it should only be enabled
    #when the model doesn't actually fit on the card
    #TODO: actually test if GP106 and others have the same type of behavior
    nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050", "p40", "p100", "p6", "p4"]
    for x in nvidia_10_series:
        if x in props.name.lower():
            fp16_works = True

    if fp16_works or manual_cast:
        free_model_memory = (get_free_memory() * 0.9 - minimum_inference_memory())
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    #FP16 is just broken on these cards
    nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True

def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device is not None:
        if is_device_cpu(device): #TODO ? bf16 works on CPU but is extremely slow
            return False

    if device is not None:
        if is_device_mps(device):
            return True

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if cpu_mode() or mps_mode():
        return False

    if is_intel_xpu():
        return True

    if device is None:
        device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8:
        return True

    bf16_works = torch.cuda.is_bf16_supported()

    if bf16_works or manual_cast:
        free_model_memory = (get_free_memory() * 0.9 - minimum_inference_memory())
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    return False

def soft_empty_cache(force=False):
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if force or is_nvidia(): #This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def unload_all_models():
    free_memory(1e30, get_torch_device())


def resolve_lowvram_weight(weight, model, key): #TODO: remove
    print("WARNING: The comfy.model_management.resolve_lowvram_weight function will be removed soon, please stop using it.")
    return weight

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

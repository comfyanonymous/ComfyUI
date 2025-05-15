"""
This file is part of ComfyUI.
Copyright (C) 2024 Comfy
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import psutil
import logging
import torch
import sys
import platform
import contextlib
import weakref
import time
import gc
import re
import threading
import traceback
from enum import Enum
from comfy.cli_args import args, PerformanceFeature
from comfy.ldm.models.autoencoder import AutoencoderKL

try:
    import torch_directml
    _torch_directml_available = True
except ImportError:
    _torch_directml_available = False

def log_vram_state(device=None):
    if not DEBUG_ENABLED:
        return
    if device is None:
        device = get_torch_device()
    free_vram, free_torch = get_free_memory(device, torch_free_too=True)
    active_models = [(m.model.__class__.__name__, m.model_memory_required(device) / 1024**3)
                     for m in current_loaded_models if m.device == device]
    logging.debug(
        f"VRAM state: free_vram={free_vram / 1024**3:.2f} GB, free_torch={free_torch / 1024**3:.2f} GB, models={active_models}")

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

cpu_state = CPUState.GPU  # Default to GPU

# Global flags
PROFILING_ENABLED = args.profile
DEBUG_ENABLED = args.debug
VERBOSE_ENABLED = False

# Configure logging
logging.basicConfig(level=logging.DEBUG if args.debug or args.profile else logging.INFO)

# Cache for device and dtype checks
_device_cache = {}

# VRAM optimizers for extensibility
_vram_optimizers = []

class VRAMState(Enum):
    DISABLED = 0    # No VRAM: models stay on CPU
    NO_VRAM = 1     # Very low VRAM: maximum memory saving
    LOW_VRAM = 2    # Low VRAM: partial model loading
    NORMAL_VRAM = 3 # Default: balanced memory management
    HIGH_VRAM = 4   # High VRAM: keep models in VRAM
    SHARED = 5      # Shared CPU/GPU memory (e.g., MPS)

# Global state
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU
total_vram = 0
total_ram = psutil.virtual_memory().total / (1024 * 1024)

# Cache for DirectML VRAM
_directml_vram_cache = {}

# Cache for active models memory in DirectML
_directml_active_memory_cache = {}

def cpu_mode():
    """Check if system is in CPU mode."""
    global cpu_state
    return cpu_state == CPUState.CPU

def mps_mode():
    """Check if system is in MPS (Apple Metal) mode."""
    global cpu_state
    return cpu_state == CPUState.MPS

def is_device_cpu(device):
    return is_device_type(device, 'cpu')

def is_device_mps(device):
    return is_device_type(device, 'mps')

def is_device_cuda(device):
    return is_device_type(device, 'cuda')

def is_directml_enabled():
    global directml_enabled
    if directml_enabled:
        return True

    return False

def get_supported_float8_types():
    """Get supported float8 data types."""
    float8_types = []
    for dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz, torch.float8_e8m0fnu]:
        try:
            float8_types.append(dtype)
        except:
            pass
    return float8_types

def get_directml_vram(dev):
    """
    Estimate VRAM for DirectML device, trying CUDA first, then heuristic, then fallback.

    Args:
        dev: Torch device (DirectML).

    Returns:
        int: Estimated VRAM in bytes.
    """
    if dev in _directml_vram_cache:
        return _directml_vram_cache[dev]

    # Use args.reserve_vram if provided
    if args.reserve_vram is not None:
        vram = int(args.reserve_vram * 1024 * 1024 * 1024)
        _directml_vram_cache[dev] = vram
        return vram

    # Try CUDA if available
    if torch.cuda.is_available():
        try:
            free_vram, total_vram = torch.cuda.mem_get_info()
            _directml_vram_cache[dev] = total_vram
            if DEBUG_ENABLED:
                logging.debug(f"DirectML VRAM from CUDA: {total_vram / (1024**3):.0f} GB")
            return total_vram
        except Exception as e:
            logging.warning(f"Failed to get CUDA VRAM: {e}")

    # Try torch_directml heuristic
    if _torch_directml_available:
        try:
            device_index = dev.index if hasattr(dev, 'index') else 0
            device_name = torch_directml.device_name(device_index).lower()
            vram_map = {
                'gtx 1660': 6 * 1024 * 1024 * 1024,
                'gtx 1650': 4 * 1024 * 1024 * 1024,
                'rtx 2060': 6 * 1024 * 1024 * 1024,
                'rtx 3060': 12 * 1024 * 1024 * 1024,
                'rtx 4060': 8 * 1024 * 1024 * 1024,
                'rx 580': 8 * 1024 * 1024 * 1024,
                'rx 570': 8 * 1024 * 1024 * 1024,
                'rx 6700': 12 * 1024 * 1024 * 1024,
                'arc a770': 16 * 1024 * 1024 * 1024,
            }
            vram = 6 * 1024 * 1024 * 1024
            for key, value in vram_map.items():
                if key in device_name:
                    vram = value
                    break
            _directml_vram_cache[dev] = vram
            if DEBUG_ENABLED:
                logging.debug(f"DirectML VRAM for {device_name}: {vram / (1024**3):.0f} GB")
            return vram
        except Exception as e:
            logging.warning(f"Failed to get DirectML device name: {e}")

    # Fallback to safe default
    vram = 6 * 1024 * 1024 * 1024
    _directml_vram_cache[dev] = vram
    if DEBUG_ENABLED:
        logging.debug(f"DirectML VRAM fallback: {vram / (1024**3):.0f} GB")
    return vram

FLOAT8_TYPES = get_supported_float8_types()
XFORMERS_IS_AVAILABLE = False
XFORMERS_ENABLED_VAE = True
ENABLE_PYTORCH_ATTENTION = True  # Enable PyTorch attention for better performance
FORCE_FP32 = args.force_fp32
DISABLE_SMART_MEMORY = args.disable_smart_memory

# Async offload setup
STREAMS = {}
NUM_STREAMS = 1
stream_counters = {}
if args.async_offload:
    logging.info(f"Using async weight offloading with {NUM_STREAMS} streams")
    # Protection for older GPUs
    if is_nvidia():
        props = torch.cuda.get_device_properties(get_torch_device())
        if props.major < 8:  # Turing (7.5) or Pascal (6.x)
            args.async_offload = False
            NUM_STREAMS = 1
            logging.warning("Async offload disabled for GPUs with SM < 8.0 to prevent memory leaks")

# Device initialization
xpu_available = False
npu_available = False
mlu_available = False
directml_enabled = args.directml is not None
torch_version_numeric = (0, 0)
try:
    torch_version = torch.__version__
    temp = torch_version.split(".")
    torch_version_numeric = (int(temp[0]), int(temp[1]))
    xpu_available = (torch_version_numeric[0] < 2 or (torch_version_numeric[0] == 2 and torch_version_numeric[1] <= 4)) and hasattr(
        torch, "xpu") and torch.xpu.is_available()
except:
    pass
if directml_enabled:
    import torch_directml
    device_index = args.directml if args.directml >= 0 else 0
    directml_device = torch_directml.device(device_index)
    logging.info(f"Using DirectML with device: {torch_directml.device_name(device_index)}")
try:
    import intel_extension_for_pytorch as ipex
    xpu_available = xpu_available or torch.xpu.is_available()
except:
    xpu_available = xpu_available or (hasattr(torch, "xpu") and torch.xpu.is_available())
try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass
try:
    import torch_npu
    npu_available = torch.npu.is_available()
except:
    npu_available = False
try:
    import torch_mlu
    mlu_available = torch.mlu.is_available()
except:
    mlu_available = False
if args.cpu:
    cpu_state = CPUState.CPU

# Device and memory utilities
def is_nvidia():
    """Check if the device is NVIDIA GPU."""
    return cpu_state == CPUState.GPU and torch.version.cuda

def is_amd():
    """Check if the device is AMD GPU."""
    return cpu_state == CPUState.GPU and torch.version.hip

def is_intel_xpu():
    """Check if the device is Intel XPU."""
    return cpu_state == CPUState.GPU and xpu_available

def is_ascend_npu():
    """Check if the device is Ascend NPU."""
    return npu_available

def is_mlu():
    """Check if the device is MLU."""
    return mlu_available

def is_device_cuda(device):
    """Check if the device is CUDA."""
    return hasattr(device, 'type') and device.type == 'cuda'

def is_device_type(device, device_type):
    """Check if the device matches the given type."""
    return hasattr(device, 'type') and device.type == device_type

def get_torch_device():
    """Get the current PyTorch device."""
    if directml_enabled:
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    if is_intel_xpu():
        return torch.device("xpu", torch.xpu.current_device())
    if is_ascend_npu():
        return torch.device("npu", torch.npu.current_device())
    if is_mlu():
        return torch.device("mlu", torch.mlu.current_device())
    return torch.device(torch.cuda.current_device())


def get_total_memory(dev=None, torch_total_too=False):
    """
    Get total memory available on the device.

    Args:
        dev: Torch device (optional, defaults to current device).
        torch_total_too: If True, return (total, torch_total).

    Returns:
        int or tuple: Total memory in bytes (or tuple with torch_total).
    """
    if dev is None:
        dev = get_torch_device()
    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        if directml_enabled:
            mem_total = get_directml_vram(dev)
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            mem_total_torch = mem_reserved
            mem_total = torch.xpu.get_device_properties(dev).total_memory
        elif is_ascend_npu():
            stats = torch.npu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            mem_total_torch = mem_reserved
            _, mem_total_npu = torch.npu.mem_get_info(dev)
            mem_total = mem_total_npu
        elif is_mlu():
            stats = torch.mlu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            mem_total_torch = mem_reserved
            _, mem_total_mlu = torch.mlu.mem_get_info(dev)
            mem_total = mem_total_mlu
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda
    return (mem_total, mem_total_torch) if torch_total_too else mem_total

# Initialize VRAM state
total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
logging.info(f"Total VRAM {total_vram:.0f} MB, total RAM {total_ram:.0f} MB")
logging.info(f"Pytorch version: {torch_version}")

def get_extra_reserved_vram():
    """
    Determine extra VRAM to reserve based on total VRAM and args.

    Returns:
        int: Reserved VRAM in bytes.
    """
    total_vram = get_total_memory(get_torch_device()) / (1024 * 1024 * 1024)  # VRAM in GB
    if args.reserve_vram is not None:
        return args.reserve_vram * 1024 * 1024 * 1024
    if total_vram < 7.9:
        return 150 * 1024 * 1024  # 150 MB for low VRAM (<7.9 GB)
    return 200 * 1024 * 1024  # 200 MB for high VRAM (?7.9 GB)

EXTRA_RESERVED_VRAM = get_extra_reserved_vram()
logging.info(f"EXTRA_RESERVED_VRAM set to {EXTRA_RESERVED_VRAM / (1024 * 1024):.0f} MB")
if args.lowvram:
    set_vram_to = VRAMState.LOW_VRAM
elif args.novram:
    set_vram_to = VRAMState.NO_VRAM
elif args.highvram or args.gpu_only:
    vram_state = VRAMState.HIGH_VRAM
if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED
elif cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED
if directml_enabled:
    lowvram_available = False
else:
    lowvram_available = True
if lowvram_available and set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
    vram_state = set_vram_to
logging.info(f"Set VRAM state to: {vram_state.name}")
if DISABLE_SMART_MEMORY:
    logging.info("Disabling smart memory management")

# XFormers and attention settings
XFORMERS_VERSION = ""
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
        XFORMERS_VERSION = xformers.version.__version__
        logging.info(f"xformers version: {XFORMERS_VERSION}")
        if XFORMERS_VERSION.startswith("0.0.18"):
            logging.warning(
                "WARNING: xformers 0.0.18 has a bug causing black images at high resolutions. Please downgrade or upgrade.")
            XFORMERS_ENABLED_VAE = False
    except:
        XFORMERS_IS_AVAILABLE = False

def xformers_enabled():
    """Check if xformers is enabled and available."""
    global directml_enabled, cpu_state
    if cpu_state != CPUState.GPU or is_intel_xpu() or is_ascend_npu() or is_mlu() or directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE and not args.disable_xformers and not args.use_pytorch_cross_attention

def xformers_enabled_vae():
    """Check if xformers is enabled for VAE."""
    enabled = xformers_enabled()
    if not enabled:
        return False
    return XFORMERS_ENABLED_VAE

def sage_attention_enabled():
    """Check if Sage Attention is enabled."""
    global directml_enabled, cpu_state
    if cpu_state != CPUState.GPU or is_intel_xpu() or is_ascend_npu() or is_mlu() or directml_enabled:
        return False
    return hasattr(args, 'use_sage_attention') and args.use_sage_attention

def flash_attention_enabled():
    """Check if Flash Attention is enabled."""
    global directml_enabled, cpu_state
    if cpu_state != CPUState.GPU or is_intel_xpu() or is_ascend_npu() or is_mlu() or directml_enabled:
        return False
    return hasattr(args, 'use_flash_attention') and args.use_flash_attention

def pytorch_attention_enabled():
    """Check if PyTorch attention is enabled."""
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION or not (xformers_enabled() or sage_attention_enabled() or flash_attention_enabled())

def pytorch_attention_enabled_vae():
    """Check if PyTorch attention is enabled for VAE."""
    if is_amd():
        return False  # Enabling PyTorch attention on AMD causes crashes at high resolutions
    return pytorch_attention_enabled()

def pytorch_attention_flash_attention():
    """Check if PyTorch Flash Attention is supported."""
    if pytorch_attention_enabled():
        if is_nvidia() or is_intel_xpu() or is_ascend_npu() or is_mlu() or is_amd():
            return True
        return False
    return False

def force_upcast_attention_dtype():
    """Check if attention dtype should be upcast (e.g., FP16 to FP32)."""
    upcast = args.force_upcast_attention
    macos_version = mac_version()
    if macos_version is not None and ((14, 5) <= macos_version < (16,)):
        upcast = True  # Workaround for macOS black image bug
    if upcast:
        return {torch.float16: torch.float32}
    return None

def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None):
    """Cast tensor to specified dtype and device, compatible with comfy.ops."""
    if device is None or weight.device == device:
        if not copy:
            if dtype is None or weight.dtype == dtype:
                return weight
        if stream is not None:
            with stream:
                return weight.to(dtype=dtype, copy=copy)
        return weight.to(dtype=dtype, copy=copy)
    if stream is not None:
        with stream:
            r = torch.empty_like(weight, dtype=dtype, device=device)
            r.copy_(weight, non_blocking=non_blocking)
    else:
        r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight, non_blocking=non_blocking)
    return r

def get_torch_device_name(device=None):
    """Get the name of the torch device."""
    if device is None:
        device = get_torch_device()
    if isinstance(device, str):
        return device
    if isinstance(device, torch.device):
        if device.type == "cuda":
            try:
                allocator = torch.cuda.get_allocator_backend()
            except:
                allocator = ""
            return f"{device.type}:{device.index if device.index is not None else 0} {allocator}"
        return device.type
    return str(device)

class OOM_EXCEPTION(Exception):
    """Exception raised for out-of-memory errors."""
    pass

if args.use_pytorch_cross_attention:
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False
MIN_WEIGHT_MEMORY_RATIO = 0.4 if is_nvidia() else 0.0
if is_nvidia() and torch_version_numeric[0] >= 2:
    if not (ENABLE_PYTORCH_ATTENTION or args.use_split_cross_attention or args.use_quad_cross_attention):
        ENABLE_PYTORCH_ATTENTION = True
elif is_intel_xpu() or is_ascend_npu() or is_mlu():
    if not (args.use_split_cross_attention or args.use_quad_cross_attention):
        ENABLE_PYTORCH_ATTENTION = True
elif is_amd() and torch_version_numeric[0] >= 2 and torch_version_numeric[1] >= 7:
    arch = torch.cuda.get_device_properties(get_torch_device()).gcnArchName
    logging.info(f"AMD arch: {arch}")
    if any(a in arch for a in ["gfx1100", "gfx1101"]) and not (args.use_split_cross_attention or args.use_quad_cross_attention):
        ENABLE_PYTORCH_ATTENTION = True
if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
if torch_version_numeric[0] == 2 and torch_version_numeric[1] >= 5:
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
else:
    logging.warning("Could not set allow_fp16_bf16_reduction_math_sdp")

def get_free_memory(dev=None, torch_free_too=False):
    """
    Get free memory available on the device.

    Args:
        dev: Torch device (optional, defaults to current device).
        torch_free_too: If True, return (free_total, free_torch).

    Returns:
        int or tuple: Free memory in bytes (or tuple with free_torch).
    """
    if dev is None:
        dev = get_torch_device()
    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            total_vram = get_directml_vram(dev)
            cache_key = (dev, 'active_models')
            if cache_key not in _directml_active_memory_cache:
                active_models = sum(m.model_loaded_memory() for m in current_loaded_models if m.device == dev)
                _directml_active_memory_cache[cache_key] = active_models
            active_models = _directml_active_memory_cache[cache_key]
            mem_free_total = max(1024 * 1024 * 1024, total_vram - active_models * 1.2)
            mem_free_torch = mem_free_total
            if DEBUG_ENABLED:
                logging.debug(f"DirectML: total_vram={total_vram / (1024**3):.0f} GB, active_models={active_models / (1024**3):.2f} GB, free={mem_free_total / (1024**3):.2f} GB")
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_torch = mem_reserved - mem_active
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_total = mem_free_xpu + mem_free_torch
        elif is_ascend_npu():
            stats = torch.npu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_npu, _ = torch.npu.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_npu + mem_free_torch
        elif is_mlu():
            stats = torch.mlu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_mlu, _ = torch.mlu.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_mlu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch
    return (mem_free_total, mem_free_torch) if torch_free_too else mem_free_total

def get_adaptive_min_free(mem_total, memory_required=None):
    """
    Calculate adaptive min_free VRAM based on GPU memory and model requirements.
    
    Args:
        mem_total (float): Total GPU VRAM in GB.
        memory_required (float, optional): Estimated memory required by the model in GB.
    
    Returns:
        float: Minimum free VRAM required in GB.
    """
    # Base min_free as a fraction of total VRAM
    base_min_free = mem_total * 0.25  # 25% of total VRAM as baseline
    
    if memory_required is not None:
        min_free = max(base_min_free, memory_required)  # Use memory_required directly, no extra multiplier
    else:
        min_free = base_min_free
    
    # Cap min_free to avoid excessive requirements
    min_free = min(min_free, mem_total * 0.5)  # Never exceed 50% of total VRAM
    
    # Minimum threshold for very small GPUs
    min_free = max(min_free, 1.0 if mem_total < 6.0 else 1.5)
    
    if PROFILING_ENABLED:
        memory_required_str = f"{memory_required:.2f}" if memory_required is not None else "None"
        logging.debug(f"get_adaptive_min_free: mem_total={mem_total:.2f} GB, memory_required={memory_required_str} GB, min_free={min_free:.2f} GB")
    
    return min_free

def memory_monitor(device, interval=5.0):
    """Monitor memory usage in a background thread."""
    if not DEBUG_ENABLED:
        return

    def monitor():
        while True:
            log_vram_state(device)
            time.sleep(interval)
    threading.Thread(target=monitor, daemon=True).start()

memory_monitor(get_torch_device())

def soft_empty_cache(clear=False, device=None, caller="unknown"):
    """
    Clear PyTorch memory cache efficiently with VRAM check.

    Args:
        clear (bool): Force cache clearing regardless of memory state.
        device (torch.device): Device to clear cache for. Defaults to current device.
        caller (str): Source of the call for debugging.
    """
    if device is None:
        device = get_torch_device()

    if PROFILING_ENABLED:
        start_time = time.time()
        logging.debug(f"soft_empty_cache called with clear={clear}, device={device}, caller={caller}")

    # Fixed threshold in bytes (100 MB)
    MEMORY_THRESHOLD = 100 * 1024 * 1024  # 100 MB
    cache_key = (device, 'free_memory')

    mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
    total_vram = get_total_memory(device)
    if not clear and (mem_free_torch <= MEMORY_THRESHOLD or mem_free_total > 0.4 * total_vram):
        if PROFILING_ENABLED:
            logging.debug(f"soft_empty_cache: Skipped (free_vram={mem_free_total/1024**3:.2f} GB, free_torch={mem_free_torch/1024**3:.2f} GB)")
        return

    try:
        if is_device_cuda(device):
            torch.cuda.empty_cache()
            if clear and mem_free_torch < min(0.1 * total_vram, 1.0 * 1024**3):
                gc.collect()
                if torch.distributed.is_initialized():
                    torch.cuda.ipc_collect()
        elif cpu_state == CPUState.MPS:
            torch.mps.empty_cache()
        elif is_intel_xpu():
            torch.xpu.empty_cache()
        elif is_ascend_npu():
            torch.npu.empty_cache()
        elif is_mlu():
            torch.mlu.empty_cache()

        if PROFILING_ENABLED:
            free_vram_after, free_torch_after = get_free_memory(device, torch_free_too=True)
            logging.debug(f"After clear: free_vram={free_vram_after/1024**3:.2f} GB, free_torch={free_torch_after/1024**3:.2f} GB")
            logging.debug(f"soft_empty_cache took {time.time() - start_time:.3f} s, gained={(free_vram_after - mem_free_total)/1024**3:.2f} GB")
    except Exception as e:
        if PROFILING_ENABLED:
            logging.warning(f"Failed to clear cache for {device}: {str(e)}")
            
def unload_all_models():
    """
    Unload all models from memory and clear cache if necessary.
    """
    if PROFILING_ENABLED:
        start_time = time.time()
        logging.debug("unload_all_models called")

    for model in list(current_loaded_models):
        model.model_unload()
    
    current_loaded_models.clear()
    
    device = get_torch_device()
    free_vram = get_free_memory(device)[0]
    total_vram = get_total_memory(device)
    clear_aggressive = free_vram < 0.4 * total_vram
    if PROFILING_ENABLED:
        logging.debug(f"unload_all_models: free_vram={free_vram/1024**3:.2f} GB, aggressive_clear={clear_aggressive}")
    
    soft_empty_cache(clear=clear_aggressive, caller="unload_all_models")
    
    if PROFILING_ENABLED:
        new_free_vram = get_free_memory(device)[0]
        logging.debug(f"unload_all_models done: free={new_free_vram/1024**3:.2f} GB, "
                      f"gained={(new_free_vram - free_vram)/1024**3:.2f} GB, took={time.time() - start_time:.3f} s")

def get_offload_stream(device):
    """Get a stream for asynchronous weight offloading."""
    stream_counter = stream_counters.get(device, 0)
    if NUM_STREAMS <= 1 or not is_device_cuda(device):
        return None
    if device in STREAMS:
        ss = STREAMS[device]
        s = ss[stream_counter]
        stream_counter = (stream_counter + 1) % len(ss)
        if is_device_cuda(device):
            ss[stream_counter].wait_stream(torch.cuda.current_stream())
        stream_counters[device] = stream_counter
        return s
    elif is_device_cuda(device):
        ss = [torch.cuda.Stream(device=device, priority=0) for _ in range(NUM_STREAMS)]
        STREAMS[device] = ss
        s = ss[stream_counter]
        stream_counter = (stream_counter + 1) % len(ss)
        stream_counters[device] = stream_counter
        return s
    return None

def sync_stream(device, stream):
    """Synchronize the given stream with the current CUDA stream."""
    if stream is None or not is_device_cuda(device):
        return
    torch.cuda.current_stream().wait_stream(stream)

def cast_to_device(tensor, device, dtype, copy=False):
    """Cast tensor to specified device and dtype with non-blocking support."""
    non_blocking = device_supports_non_blocking(device)
    return cast_to(tensor, dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)

def register_vram_optimizer(optimizer):
    """Register a VRAM optimizer."""
    _vram_optimizers.append(optimizer)

# Model management
current_loaded_models = []

class LoadedModel:
    def __init__(self, model):
        self._set_model(model)
        self.device = model.load_device
        self.real_model = None
        self.currently_used = True
        self.model_offloaded = False
        self.model_finalizer = None
        self._patcher_finalizer = None

    def _set_model(self, model):
        self._model = weakref.ref(model)
        if hasattr(model, 'parent') and model.parent is not None:
            self._parent_model = weakref.ref(model.parent)
            self._patcher_finalizer = weakref.finalize(model, self._switch_parent)

    def _switch_parent(self):
        if hasattr(self, '_parent_model'):
            model = self._parent_model()
            if model is not None:
                self._set_model(model)

    @property
    def model(self):
        return self._model()

    def model_memory(self):
        return self.model.model_size() if hasattr(self.model, 'model_size') else module_size(self.model)

    def model_loaded_memory(self):
        return self.model.loaded_size() if hasattr(self.model, 'loaded_size') else module_size(self.model)

    def model_offloaded_memory(self):
        return self.model_memory() - self.model_loaded_memory()

    def model_memory_required(self, device):
        """
        Estimate memory required for the model on the specified device.

        Args:
            device (torch.device): Target device for memory estimation.

        Returns:
            int: Memory required in bytes.
        """
        # Fast path: use size if available
        if hasattr(self.model, 'size') and self.model.size > 0:
            return self.model.size

        # Check if model is already on the target device
        if hasattr(self.model, 'current_loaded_device') and device == self.model.current_loaded_device():
            return self.model_offloaded_memory()

        # Handle AutoencoderKL
        if self.model.model is not None and isinstance(self.model.model, AutoencoderKL):
            shape = getattr(self.model, 'last_shape', (1, 4, 64, 64))
            dtype = getattr(self.model, 'model_dtype', torch.float32)()
            return estimate_vae_decode_memory(self.model.model, shape, dtype)

        # Sum memory for additional models
        loaded_memory = 0
        if hasattr(self.model, 'additional_models'):
            model_device = device
            if hasattr(self.model.model, 'device'):
                model_device = self.model.model.device
                if DEBUG_ENABLED:
                    logging.debug(f"[DEBUG_CLONES] Model {self.model.__class__.__name__} using device {model_device}")
            for m in self.model.additional_models:
                try:
                    loaded_memory += m.model_memory_required(model_device)
                except Exception as e:
                    if DEBUG_ENABLED:
                        logging.warning(f"[DEBUG_CLONES] Error calculating memory for additional model: {e}")

        return self.model_memory() + loaded_memory
	    
    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        with profile_section("Model load"):
            self.model.model_patches_to(self.device)
            self.model.model_patches_to(self.model.model_dtype())
            use_more_vram = lowvram_model_memory if lowvram_model_memory > 0 else float('inf')
            self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)
            real_model = self.model.model
            if is_intel_xpu() and not args.disable_ipex_optimize and 'ipex' in globals():
                with torch.no_grad():
                    real_model = ipex.optimize(real_model.eval(), inplace=True, graph_mode=True, concat_linear=True)
            self.real_model = weakref.ref(real_model)
            self.model_finalizer = weakref.finalize(real_model, cleanup_models)
            return real_model

    def should_reload_model(self, force_patch_weights=False):
        return force_patch_weights and self.model.lowvram_patch_counter() > 0

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        """
        Unload the model, freeing memory on both CPU and GPU.
        Clears CUDA cache if needed, logs critical information.

        Args:
            memory_to_free: Amount of memory to free (bytes), if partial unloading is needed.
            unpatch_weights: Whether to unpatch model weights during unloading.

        Returns:
            float: Estimated memory freed (in bytes).
        """
        with profile_section("Model unload"):
            if self.is_dead() or self.real_model is None:
                if DEBUG_ENABLED:
                    logging.debug("[DEBUG_CLONES] Model is dead or real_model is None, skipping unload")
                return 0

            mem_freed = getattr(self.model, 'model_loaded_weight_memory', 0) if self.model is not None else 0
            is_cuda = is_device_cuda(self.device)

            try:
                model_name = self.model.__class__.__name__ if self.model is not None else "None"
                model_type = self.model.model.__class__.__name__ if self.model is not None and hasattr(self.model, 'model') else "Unknown"
                if DEBUG_ENABLED:
                    logging.debug(f"[DEBUG_CLONES] Starting unload for {model_name}(type={model_type})")

                # Partial unload if requested and supported
                if memory_to_free is not None and memory_to_free < mem_freed and hasattr(self.model, 'partially_unload'):
                    freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
                    if freed >= memory_to_free:
                        if PROFILING_ENABLED:
                            logging.debug(f"[DEBUG_CLONES] Partial unload freed {freed / 1024**3:.2f} GB")
                        return freed

                # Full unload
                if self.model is not None and hasattr(self.model, 'detach'):
                    self.model.detach(unpatch_all=unpatch_weights)
                if self.model_finalizer is not None:
                    self.model_finalizer.detach()
                self.model_finalizer = None
                self.real_model = None
                self._model = lambda: None

                # Garbage collection for non-CUDA devices
                if not is_cuda:
                    gc.collect()

                # Clear CUDA cache if on CUDA device
                if is_cuda:
                    device = self.device
                    free_vram = get_free_memory(device)[0]
                    total_vram = get_total_memory(device)
                    clear_aggressive = free_vram < 0.4 * total_vram
                    soft_empty_cache(clear=clear_aggressive, caller="model_unload")

                if PROFILING_ENABLED:
                    logging.debug(f"[DEBUG_CLONES] Unload complete for {model_name}")
                return mem_freed

            except Exception as e:
                if DEBUG_ENABLED:
                    logging.warning(f"[DEBUG_CLONES] Error during model_unload for {model_name}(type={model_type}): {e}")
                return mem_freed

    def model_use_more_vram(self, use_more_vram, force_patch_weights=False):
        if not use_more_vram:
            if PROFILING_ENABLED:
                logging.debug(
                    "model_use_more_vram: use_more_vram=False, returning 0")
            return 0
        mem_required = self.model_memory_required(self.device)
        extra_memory = min(mem_required * 0.3, 50 * 1024 * 1024 * 1024)  # Reduced to 50 MB chunks
        return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)

    def __eq__(self, other):
        return self.model is other.model

    def __del__(self):
        if hasattr(self, '_patcher_finalizer') and self._patcher_finalizer is not None:
            self._patcher_finalizer.detach()
        if hasattr(self, '_model_finalizer') and self._model_finalizer is not None:
            self._model_finalizer.detach()

    def is_dead(self):
        """
        Check if the model is dead (real_model exists but model is garbage collected).
        Returns True if the model is dead, False otherwise.
        """
        if self.real_model is None:
            return False  # Model was never loaded or already unloaded
        return self.real_model() is not None and self.model is None

def module_size(model, shape=None, dtype=None):
    """
    Estimate memory size of a module by summing parameter and buffer sizes,
    or using VAE-specific estimation if shape and dtype are provided.
    """
    from diffusers import AutoencoderKL

    module_mem = 0
    if shape is not None and dtype is not None and isinstance(model, AutoencoderKL):
        try:
            batch, channels, height, width = shape
            # Adjusted memory estimate for VAE: reduced multiplier from 64*1.1 to 32*1.05 to avoid overestimation
            base_memory = height * width * channels * 32 * 1.05
            size_of_dtype = dtype_size(dtype)
            module_mem = base_memory * size_of_dtype
            # Add parameter memory for VAE to account for model weights
            param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
            module_mem += param_mem
            if DEBUG_ENABLED:
                logging.debug(
                    f"Estimated VAE memory: shape={shape}, dtype={dtype}, "
                    f"params={param_mem / (1024**3):.2f} GB, total={module_mem / (1024**3):.2f} GB"
                )
        except Exception as e:
            logging.warning(f"Failed to estimate VAE memory for {model.__class__.__name__}: {str(e)}")

    if module_mem == 0:
        try:
            # Sum memory of state dict (parameters and buffers)
            module_mem = sum(p.numel() * p.element_size() for p in model.state_dict().values())
        except AttributeError:
            # Fallback: sum parameters and buffers separately
            if hasattr(model, 'parameters'):
                module_mem += sum(p.numel() * p.element_size() for p in model.parameters())
            if hasattr(model, 'buffers'):
                module_mem += sum(b.numel() * b.element_size() for b in model.buffers())
            if module_mem == 0:
                model_name = model.__class__.__name__.lower()
                if 'vae' in model_name or isinstance(model, AutoencoderKL):
                    # Reduced fallback from 3.5 GB to 2.5 GB for VAE
                    module_mem = 2.5 * 1024**3
                    logging.warning(
                        f"Could not estimate module size for {model.__class__.__name__}, "
                        f"assuming 2.5 GB for VAE"
                    )
                else:
                    # Minimal memory assumption for unknown models
                    module_mem = 1024 * 1024
                    logging.warning(
                        f"Could not estimate module size for {model.__class__.__name__}, "
                        f"assuming minimal memory (1 MB)"
                    )

    if VERBOSE_ENABLED:
        logging.debug(f"Module size for {model.__class__.__name__}: {module_mem / (1024**3):.2f} GB")
    return module_mem

def dtype_size(dtype):
    """Get the size of a data type in bytes."""
    dtype_size = 4
    if dtype in (torch.float16, torch.bfloat16):
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    elif dtype in FLOAT8_TYPES:
        dtype_size = 1
    else:
        try:
            dtype_size = dtype.itemsize
        except:
            pass
    return dtype_size

def get_adaptive_buffer(device):
    """
    Calculate adaptive memory buffer based on VRAM size and available memory.

    Args:
        device: Device to calculate buffer for.

    Returns:
        Buffer size in bytes.
    """
    mem_total = get_total_memory(device) if is_device_cuda(device) else 6 * 1024**3
    mem_free_total, _ = get_free_memory(device, torch_free_too=True)
    # Use 2% for low VRAM (<50% free) or small GPUs, 5% otherwise
    fraction = 0.02 if mem_free_total < mem_total * 0.5 or mem_total < 8 * 1024**3 else 0.05
    buffer = min(max(fraction * mem_total, 0.05 * 1024**3), 0.2 * 1024**3)  # 0.05-0.2 GB
    if PROFILING_ENABLED:
        logging.debug(f"get_adaptive_buffer: mem_total={mem_total / 1024**3:.2f} GB, "
                      f"mem_free_total={mem_free_total / 1024**3:.2f} GB, buffer={buffer / 1024**3:.2f} GB")
    return buffer

def estimate_vae_decode_memory(model, shape, dtype):
    """
    Estimate memory required for VAE decoding.
    Uses module_size with shape and dtype for accurate estimation.
    """
    total_memory = module_size(model, shape=shape, dtype=dtype)
    if PROFILING_ENABLED:
        logging.debug(
            f"Estimated VAE decode memory: shape={shape}, dtype={dtype}, "
            f"total={total_memory / (1024**3):.2f} GB"
        )
    return total_memory


def use_more_memory(extra_memory, loaded_models, device):
    """Use additional VRAM for loaded models."""
    for m in loaded_models:
        if m.device == device:
            extra_memory -= m.model_use_more_vram(extra_memory)
            if extra_memory <= 0:
                break

def offloaded_memory(loaded_models, device):
    """Calculate offloaded memory for loaded models."""
    offloaded_mem = 0
    for m in loaded_models:
        if m.device == device:
            offloaded_mem += m.model_offloaded_memory()
    return offloaded_mem

def extra_reserved_memory():
    """Get extra reserved VRAM."""
    return EXTRA_RESERVED_VRAM

def minimum_inference_memory():
    """Get minimum memory required for inference."""
    return (1024 * 1024 * 1024) * 0.6 + extra_reserved_memory()  # Reduced to 600 MB


def cleanup_models_gc():
    """Clean up dead models and collect garbage if significant memory is freed."""
    dead_memory = 0
    for cur in current_loaded_models:
        if cur.is_dead():
            dead_memory += cur.model_memory()
    
    if dead_memory > 50 * 1024 * 1024:  # 50 MB threshold
        if PROFILING_ENABLED:
            device = get_torch_device()
            free_vram = get_free_memory(device)[0]
            total_vram = get_total_memory(device)
            logging.debug(f"cleanup_models_gc: dead_memory={dead_memory/1024**2:.2f} MB, "
                          f"free_vram={free_vram/1024**3:.2f} GB")
                          
        soft_empty_cache(clear=False, caller="cleanup_models_gc")
    
    i = len(current_loaded_models) - 1
    while i >= 0:
        if current_loaded_models[i].is_dead():
            logging.warning(f"Removing dead model {current_loaded_models[i].real_model().__class__.__name__}")
            current_loaded_models.pop(i)
        i -= 1

def free_memory(memory_required, device, keep_loaded=None, loaded_models=None, caller="unknown"):
    """
    Free memory on the device by unloading models efficiently, prioritizing unused models.

    Args:
        memory_required (int): Memory needed in bytes.
        device (torch.device): Device to free memory on.
        keep_loaded (list, optional): Models to keep loaded. Defaults to [].
        loaded_models (list, optional): List of models to consider for unloading. Defaults to current_loaded_models.
        caller (str): Source of the call for debugging.

    Returns:
        list: Unloaded models.
    """
    with profile_section("free_memory"):
        # Initialize defaults
        if keep_loaded is None:
            keep_loaded = []
        if loaded_models is None:
            loaded_models = current_loaded_models

        # Cache memory state to avoid redundant calls
        cache_key = (device, 'free_memory')
        mem_free = _device_cache.get(cache_key, None)
        if mem_free is None:
            mem_free = get_free_memory(device, torch_free_too=True)
            _device_cache[cache_key] = mem_free
        mem_free_total = mem_free[0] if isinstance(mem_free, tuple) else mem_free
        total_vram = get_total_memory(device)

        # Log initial state if profiling is enabled
        if PROFILING_ENABLED:
            logging.debug(
                f"free_memory: requested={memory_required / 1024**3:.2f} GB, "
                f"free={mem_free_total / 1024**3:.2f} GB, models={len(loaded_models)}, "
                f"device={device}, caller={caller}"
            )

        # Skip if enough VRAM (>20% of total or required memory available)
        if mem_free_total > max(memory_required, 0.4 * total_vram):
            if PROFILING_ENABLED:
                logging.debug(f"free_memory: Skipped (free_vram={mem_free_total / 1024**3:.2f} GB)")
            return []

        # Apply VRAM optimizers
        for optimizer in _vram_optimizers:
            memory_required = optimizer(memory_required, device, keep_loaded)

        # Ensure minimum inference memory
        memory_required = max(memory_required, minimum_inference_memory())

        # Clean up dead models
        cleanup_models_gc()

        unloaded_models = []
        can_unload = []

        # Collect models that can be unloaded
        for i in range(len(loaded_models) - 1, -1, -1):
            model = loaded_models[i]
            if model.device == device and model not in keep_loaded and not model.is_dead() and not model.currently_used:
                mem_required = model.model_memory_required(device)
                can_unload.append((mem_required, model, i))
        can_unload.sort(reverse=True)  # Prioritize models using more memory

        # Calculate memory to free
        memory_to_free = memory_required - mem_free_total + extra_reserved_memory()

        # Unload models to free required memory
        for mem, model, index in can_unload:
            try:
                model_id = getattr(model.model, 'model_id', id(model.model) if hasattr(model.model, 'model') else model.model.__class__.__name__)
                model_type = model.model.__class__.__name__ if hasattr(model.model, 'model') else 'Unknown'
                mem_freed = model.model_unload(memory_to_free=memory_to_free)
                loaded_models.pop(index)
                unloaded_models.append(model)
                if model.model is not None and hasattr(model.model, 'detach'):
                    model.model.detach(unpatch_all=True)
                mem_free_total += mem_freed
                _device_cache[cache_key] = (mem_free_total, mem_free[1] if isinstance(mem_free, tuple) else 0)
                if PROFILING_ENABLED:
                    logging.debug(
                        f"Unloaded model: id={model_id}, type={model_type}, "
                        f"freed={mem_freed / 1024**3:.2f} GB, free_vram={mem_free_total / 1024**3:.2f} GB"
                    )
                if mem_free_total >= memory_required:
                    break
            except Exception as e:
                if DEBUG_ENABLED:
                    logging.warning(f"Failed to unload model at index {index}: {e}")

        # Utilize excess memory if available
        use_more_memory(mem_free_total - memory_required, loaded_models, device)

        if PROFILING_ENABLED:
            logging.debug(
                f"free_memory done: free={mem_free_total / 1024**3:.2f} GB, unloaded={len(unloaded_models)} models"
            )

        return unloaded_models

def load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
    """
    Load multiple models to GPU, managing VRAM efficiently.

    Args:
        models: List of models to load.
        memory_required: Estimated memory needed (bytes).
        force_patch_weights: Force re-patching model weights.
        minimum_memory_required: Minimum memory needed for inference.
        force_full_load: Force full model loading regardless of VRAM state.
    """
    cleanup_models_gc()
    with profile_section("load_models_gpu"):
        # Memory cache for efficient memory queries
        memory_cache = {}
        def get_cached_memory(device, torch_free_too=False):
            cache_key = (device, torch_free_too)
            if cache_key not in memory_cache:
                try:
                    memory_cache[cache_key] = get_free_memory(device, torch_free_too)
                except Exception as e:
                    logging.error(f"Failed to get memory for {device}: {e}")
                    memory_cache[cache_key] = (0, 0) if torch_free_too else 0
            return memory_cache[cache_key]
        
        if minimum_memory_required is None:
            minimum_memory_required = minimum_inference_memory()
        device = get_torch_device()
        if vram_state in (VRAMState.DISABLED, VRAMState.SHARED):
            return
            
        model_lookup = {m.model: m for m in current_loaded_models if m.model is not None}
            
        # Reset currently_used flag for all loaded models
        for loaded_model in current_loaded_models:
            loaded_model.currently_used = False

        loaded = []
         # Prepare models to load
        for model in models:
            if not hasattr(model, "model"):
                continue
            loaded_model = model_lookup.get(model)
            if loaded_model is None:
                loaded_model = LoadedModel(model)
                model_lookup[model] = loaded_model
            loaded_model.currently_used = True
            loaded.append(loaded_model)
            
        # Unload unused models only if necessary
        device = get_torch_device()
        to_remove = []
        if len(current_loaded_models) > 10 or (is_device_cuda(device) and get_cached_memory(device) < 1 * 1024 * 1024 * 1024):  # >10 models or <1GB VRAM
            for i, loaded_model in enumerate(current_loaded_models):
                if not loaded_model.currently_used:
                    model = loaded_model.model
                    if model is None:
                        to_remove.append(i)
                        continue
                    try:
                        mem_freed = loaded_model.model_unload()
                        to_remove.append(i)
                        if hasattr(model, 'detach') and hasattr(model, 'patched_weights') and model.patched_weights:
                            model.detach(unpatch_all=True)
                    except Exception as e:
                        logging.error(f"Failed to unload model at index {i}: {e}")
            for i in reversed(to_remove):
                current_loaded_models.pop(i)

        lowvram_model_memory = 0
        if vram_state == VRAMState.LOW_VRAM and not force_full_load:
            lowvram_model_memory = max(
                int(get_total_memory(device) * MIN_WEIGHT_MEMORY_RATIO), 400 * 1024 * 1024)
        elif vram_state == VRAMState.NO_VRAM:
            lowvram_model_memory = 1

        for l in loaded:
            l.currently_used = True
            if l.should_reload_model(force_patch_weights=force_patch_weights) or l.real_model is None:
                mem_needed = l.model_memory_required(device)
                mem_free = get_free_memory(device)
                if DEBUG_ENABLED:
                    logging.debug(
                        f"Loading {l.model.__class__.__name__}: mem_needed={mem_needed / 1024**3:.2f} GB, free={mem_free / 1024**3:.2f} GB")

                if mem_free < mem_needed + minimum_memory_required:
                    free_memory(mem_needed + minimum_memory_required,
                                device, keep_loaded=loaded)
                    mem_free = get_free_memory(device)

                stream = get_offload_stream(device)
                with torch.cuda.stream(stream) if stream is not None else torch.no_grad():
                    l.model_load(lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights)
                if loaded_model not in current_loaded_models:
                    current_loaded_models.append(l)  # append for efficiency
                sync_stream(device, stream)
                if DEBUG_ENABLED:
                    logging.debug(
                        f"Loaded {l.model.__class__.__name__}: free={get_free_memory(device) / 1024**3:.2f} GB")

    return

def load_model_gpu(model):
    """Load a single model to GPU, wrapper around load_models_gpu."""
    return load_models_gpu([model])


def loaded_models(only_currently_used=False):
    """Return list of loaded models, optionally only those currently used."""
    output = []
    for m in current_loaded_models:
        if only_currently_used and not m.currently_used:
            continue
        output.append(m.model)
    return output

# Data type selection
def supports_fp8_compute(device=None):
    """Check if the device supports FP8 computation."""
    if not is_nvidia():
        return False
    if device is None:
        device = get_torch_device()
    props = torch.cuda.get_device_properties(device)
    if props.major >= 9:  # Ada Lovelace
        return True
    if props.major == 8 and props.minor >= 9 and torch_version_numeric >= (2, 3):
        if any(platform.win32_ver()) and torch_version_numeric < (2, 4):
            return False
        return True
    return False

def supports_dtype(dtype, device):
    """Check if the device supports the given data type."""
    if dtype == torch.bfloat16:
        if is_nvidia():
            return torch.cuda.get_device_properties(device).major >= 8
        elif is_amd():
            arch = torch.cuda.get_device_properties(device).gcnArchName
            return any(a in arch for a in ["gfx941", "gfx942"])
        return False
    elif dtype in (torch.float16, torch.float32):
        return True
    elif dtype in FLOAT8_TYPES:
        return supports_fp8_compute(device)
    return False

def supports_cast(dtype, device):
    """Check if the device supports casting to the given data type."""
    if dtype == torch.bfloat16:
        return True
    return supports_dtype(dtype, device)

def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    """Determine if FP16 should be used for the device."""
    if device is None:
        device = get_torch_device()
    if FORCE_FP32:
        return False
    if args.force_fp16:
        return supports_cast(torch.float16, device)
    if is_intel_xpu():
        return True
    if is_mlu():
        props = torch.mlu.get_device_properties(device)
        return props.major >= 3
    if is_ascend_npu():
        return False
    if is_amd():
        arch = torch.cuda.get_device_properties(device).gcnArchName
        if any(a in arch for a in ["gfx1030", "gfx1031", "gfx1010", "gfx1011", "gfx1012", "gfx906", "gfx900", "gfx803"]):
            return manual_cast
        return True
    props = torch.cuda.get_device_properties(device)
    if is_nvidia():
        # Prefer FP32 for low VRAM or older GPUs
        total_vram = get_total_memory(device) / (1024**3)
        if total_vram < 5.9 or props.major <= 7:  # Turing (7.5) or Pascal (6.x)
            return False
        if any(platform.win32_ver()) and props.major <= 7:
            return manual_cast and torch.cuda.is_bf16_supported()
    if props.major >= 8:
        return True
    return torch.cuda.is_bf16_supported() and manual_cast and (not prioritize_performance or model_params * 4 > get_total_memory(device))

def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    """Determine if BF16 should be used for the device."""
    if device is None:
        device = get_torch_device()
    if args.force_fp16 or FORCE_FP32:
        return False
    if not is_device_cuda(device):
        return False  # BF16 not supported on CPU or MPS
    props = torch.cuda.get_device_properties(device)
    if is_nvidia():
        return props.major >= 8 and supports_cast(torch.bfloat16, device)
    elif is_amd():
        arch = props.gcnArchName
        return any(a in arch for a in ["gfx941", "gfx942"]) and supports_cast(torch.bfloat16, device)
    return False

def vae_dtype(device=None, model=None):
    """
    Select appropriate data type for VAE.

    Args:
        device: PyTorch device (e.g., 'cuda', 'cpu'). Defaults to get_torch_device().
        model: Optional model to check compatibility (not used in this implementation).

    Returns:
        torch.dtype: Appropriate data type for VAE (e.g., torch.float32, torch.float16, torch.bfloat16).
    """
    if device is None:
        device = get_torch_device()
    
    # Handle CPU case explicitly to avoid CUDA calls
    if device.type == 'cpu':
        logging.debug(f"VAE dtype: torch.float32 (CPU device)")
        return torch.float32

    # Handle forced FP32/FP16 via command-line arguments
    if args.force_fp32_vae:
        if DEBUG_ENABLED:
            logging.debug(f"VAE dtype: torch.float32 (forced via --force-fp32-vae)")
        return torch.float32
    if args.force_fp16_vae:
        if supports_cast(torch.float16, device):
            if DEBUG_ENABLED:
                logging.debug(f"VAE dtype: torch.float16 (forced via --force-fp16-vae)")
            return torch.float16
        if DEBUG_ENABLED:
            logging.debug(f"VAE dtype: torch.float32 (FP16 not supported on {device})")
        return torch.float32

    # Handle NVIDIA GPUs
    if is_nvidia():
        props = torch.cuda.get_device_properties(device)
        total_vram = get_total_memory(device) / (1024**3)
        if total_vram < 5.9 or props.major <= 7:  # Turing (7.5) or Pascal (6.x)
            # Try FP16 with fallback to FP32 if unstable
            if supports_cast(torch.float16, device) and total_vram >= 3.9:
                if DEBUG_ENABLED:
                    logging.debug(f"VAE dtype: torch.float16 (Turing SM {props.major}.{props.minor}, VRAM {total_vram:.1f} GB)")
                return torch.float16
            if DEBUG_ENABLED:
                logging.debug(f"VAE dtype: torch.float32 (Turing SM {props.major}.{props.minor}, low VRAM {total_vram:.1f} GB)")
            return torch.float32

    # Handle bfloat16 and FP16 for other devices
    if should_use_bf16(device=device, prioritize_performance=False):
        if DEBUG_ENABLED:
            logging.debug(f"VAE dtype: torch.bfloat16 (device supports BF16)")
        return torch.float16
    if should_use_fp16(device=device, prioritize_performance=False):
        if DEBUG_ENABLED:
            logging.debug(f"VAE dtype: torch.float16 (device supports FP16)")
        return torch.float16

    # Default fallback
    if DEBUG_ENABLED:
        logging.debug(f"VAE dtype: torch.float32 (default fallback)")
    return torch.float32

def unet_dtype(device=None, model=None, model_params=None, supported_dtypes=None, weight_dtype=None):
    """Select appropriate data type for UNet."""
    if device is None:
        device = get_torch_device()
    model_params = module_size(model) // 4 if model is not None else 0

    # FP8 support
    if args.fp8_e4m3fn_unet and supports_fp8_compute(device):
        return torch.float8_e4m3fn
    if args.fp8_e5m2_unet and supports_fp8_compute(device):
        return torch.float8_e5m2
    fp8_dtype = None
    if weight_dtype in FLOAT8_TYPES:
        fp8_dtype = weight_dtype
    if fp8_dtype is not None:
        if supports_fp8_compute(device):
            return fp8_dtype
        free_model_memory = maximum_vram_for_weights(device)
        if model_params * 2 > free_model_memory:
            return fp8_dtype

    # Check supported_dtypes and weight_dtype
    if supported_dtypes is not None and weight_dtype is not None:
        for dtype in supported_dtypes:
            if dtype == weight_dtype:
                return dtype

    # Fallback to bf16/fp16/fp32 based on device and args
    if args.force_fp16 and supports_cast(torch.float16, device):
        return torch.float16
    if args.force_fp32:
        return torch.float32
    if should_use_bf16(device, model_params, prioritize_performance=True):
        return torch.bfloat16
    if should_use_fp16(device, model_params, prioritize_performance=True):
        return torch.float16
    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params, manual_cast=True):
            return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params, manual_cast=True):
            return torch.bfloat16
    return torch.float32

def unet_offload_device():
    """Determine device for UNet offloading (GPU or CPU)."""
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    return torch.device("cpu")

def unet_inital_load_device(parameters, dtype):
    """Determine initial load device for UNet based on model size and dtype."""
    torch_dev = get_torch_device()
    if vram_state in [VRAMState.HIGH_VRAM, VRAMState.SHARED]:
        return torch_dev
    cpu_dev = torch.device("cpu")
    if DISABLE_SMART_MEMORY:
        return cpu_dev
    model_size = dtype_size(dtype) * parameters
    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev * 0.8:  # 80% threshold
        return torch_dev
    return cpu_dev

def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    """Determine if manual casting is needed for UNet dtype."""
    # No cast needed for fp32/fp64
    if weight_dtype in [torch.float32, torch.float64]:
        return None

    # Check FP8 support
    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and supports_fp8_compute(inference_device):
        return None

    # Check FP16 support
    fp16_supported = should_use_fp16(inference_device, prioritize_performance=True)
    if fp16_supported and weight_dtype == torch.float16:
        return None

    # Check BF16 support
    bf16_supported = should_use_bf16(inference_device)
    if bf16_supported and weight_dtype == torch.bfloat16:
        return None

    # Prioritize FP16 if supported and in supported_dtypes
    if fp16_supported and torch.float16 in supported_dtypes:
        return torch.float16

    # Check other supported dtypes
    for dt in supported_dtypes:
        if dt == torch.float16 and fp16_supported:
            return torch.float16
        if dt == torch.bfloat16 and bf16_supported:
            return torch.bfloat16
        if dt in [torch.float8_e4m3fn, torch.float8_e5m2] and supports_fp8_compute(inference_device):
            return dt
    # Fallback to FP32
    return torch.float32


def text_encoder_offload_device():
    """Determine device for offloading text encoder."""
    return torch.device("cpu")  # Keep offload on CPU to save VRAM


def text_encoder_device():
    """Determine device for text encoder (prefer GPU)."""
    if vram_state in (VRAMState.HIGH_VRAM, VRAMState.NORMAL_VRAM, VRAMState.LOW_VRAM):
        return get_torch_device()  # Prefer GPU for low VRAM
    return torch.device("cpu")

def text_encoder_initial_device(load_device, offload_device, model_size=0):
    """Determine initial device for text encoder."""
    if load_device == offload_device or model_size <= 512 * 1024 * 1024:
        return load_device
    if is_device_mps(load_device):
        return load_device
    mem_l = get_free_memory(load_device)
    mem_o = get_free_memory(offload_device)
    if mem_l > (mem_o * 0.5) and model_size * 1.2 < mem_l:
        return load_device
    return offload_device

def unet_inital_load_device(parameters, dtype):
    """Determine initial load device for UNet based on model size and dtype."""
    torch_dev = get_torch_device()
    if vram_state in [VRAMState.HIGH_VRAM, VRAMState.SHARED]:
        return torch_dev

    cpu_dev = torch.device("cpu")
    if DISABLE_SMART_MEMORY:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters  # Size in bytes
    mem_dev = get_free_memory(torch_dev)  # Free VRAM
    mem_cpu = get_free_memory(cpu_dev)  # Free RAM

    # Prefer GPU if VRAM > RAM and model fits in VRAM
    if mem_dev > mem_cpu and model_size < mem_dev * 0.8:  # 80% threshold
        return torch_dev
    return cpu_dev

def maximum_vram_for_weights(device=None):
    """Calculate maximum VRAM available for model weights."""
    if device is None:
        device = get_torch_device()
    return (get_total_memory(device) * 0.9 - minimum_inference_memory())


def force_channels_last():
    """
    Check if channels_last format should be used for tensors.
    Safe for Turing GPUs with FP32 VAE.
    """
    if args.force_channels_last:
        if DEBUG_ENABLED:
            logging.debug("force_channels_last: Enabled via --force-channels-last")
        return True
    if cpu_state == CPUState.GPU and is_nvidia() and torch.cuda.is_available():
        if DEBUG_ENABLED:
            total_vram = get_total_memory(get_torch_device()) / (1024 * 1024 * 1024)  # VRAM in GB
            logging.debug(
                f"force_channels_last: Enabled for NVIDIA GPU with {total_vram:.1f} GB VRAM")
        return True
    logging.debug("force_channels_last: Disabled")
    return False

def intermediate_device():
    """Determine device for intermediate computations (GPU or CPU)."""
    if args.gpu_only:
        return get_torch_device()
    return torch.device("cpu")

def get_autocast_device(dev):
    """Determine device type for autocast (e.g., cuda, cpu, mps)."""
    if hasattr(dev, 'type'):
        return dev.type
    return "cuda"

def vae_offload_device():
    """Determine device for VAE offloading (GPU or CPU)."""
    if args.gpu_only:
        return get_torch_device()
    return torch.device("cpu")

def vae_device():
    """Determine device for VAE (GPU or CPU)."""
    if args.cpu_vae:
        return torch.device("cpu")
    return get_torch_device()

def pick_weight_dtype(dtype, fallback_dtype, device=None):
    """Select appropriate dtype for model weights, using fallback if needed."""
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype
    if not supports_cast(device, dtype):
        dtype = fallback_dtype
    return dtype

def is_device_mps(device):
    """Check if device is MPS (Apple Silicon)."""
    return isinstance(device, torch.device) and device.type == "mps"

def device_supports_non_blocking(device):
    """Check if device supports non-blocking data transfers."""
    if is_device_mps(device):
        return False  # pytorch bug? mps doesn't support non blocking
    if is_intel_xpu():
        return False
    if args.deterministic:  # TODO: figure out why deterministic breaks non blocking from gpu to cpu (previews)
        return False
    if directml_enabled:
        return False
    return True

def device_should_use_non_blocking(device):
    """Determine if non-blocking transfers should be used (disabled due to memory issues)."""
    if not device_supports_non_blocking(device):
        return False
    return False
    # return True #TODO: figure out why this causes memory issues on Nvidia and possibly others

def text_encoder_dtype(device=None, model=None):
    """Select appropriate data type for text encoder."""
    if device is None:
        device = get_torch_device()
    model_params = module_size(model) // 4 if model is not None else 0
    # FP8 support (only for CUDA devices)
    if is_device_cuda(device):
        if getattr(args, 'fp8_e4m3fn_text_enc', False) and supports_fp8_compute(device):
            return torch.float8_e4m3fn
        if getattr(args, 'fp8_e5m2_text_enc', False) and supports_fp8_compute(device):
            return torch.float8_e5m2
    # Check model_dtype safely
    model_dtype = getattr(args, 'model_dtype', None)
    if model_dtype is not None:
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        if model_dtype in dtype_map:
            # Only allow BF16 on supported devices
            if model_dtype == "bf16" and not (is_device_cuda(device) and should_use_bf16(device)):
                return torch.float16
            if supports_cast(dtype_map[model_dtype], device):
                return dtype_map[model_dtype]
    # CPU/MPS fallback to FP32
    if not is_device_cuda(device):
        return torch.float32
    # CUDA devices: BF16/FP16 based on device support
    if should_use_bf16(device, model_params, prioritize_performance=True):
        return torch.bfloat16
    if should_use_fp16(device, model_params, prioritize_performance=True):
        return torch.float16
    return torch.float16  # Default to FP16 for GPU


def mac_version():
    """Get macOS version as a tuple."""
    try:
        return tuple(int(n) for n in platform.mac_ver()[0].split("."))
    except:
        return None

# Interrupt handling
class InterruptProcessingException(Exception):
    pass

interrupt_processing_mutex = threading.RLock()
interrupt_processing = False

def interrupt_current_processing(value=True):
    """Set interrupt flag for processing."""
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        interrupt_processing = value

def lowvram_enabled():
    """Check if low VRAM mode is enabled."""
    return vram_state == VRAMState.LOW_VRAM

def noram_enabled():
    """Check if no VRAM mode is enabled."""
    return vram_state == VRAMState.NO_VRAM


def processing_interrupted():
    """Check if processing is interrupted."""
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        return interrupt_processing

def throw_exception_if_processing_interrupted():
    """Throw exception if processing is interrupted."""
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()

def controlnet_device():
    """Determine device for ControlNet (GPU or CPU)."""
    if args.gpu_only:
        return get_torch_device()
    return torch.device("cpu")

def controlnet_dtype(device=None, model=None):
    """Select appropriate data type for ControlNet."""
    if device is None:
        device = get_torch_device()
    model_params = module_size(model) // 4 if model is not None else 0
    if args.force_fp16:
        if supports_cast(torch.float16, device):
            logging.debug(f"ControlNet dtype: torch.float16 (forced via --force-fp16)")
            return torch.float16
        logging.debug(f"ControlNet dtype: torch.float32 (FP16 not supported)")
        return torch.float32
    if args.force_fp32:
        logging.debug(f"ControlNet dtype: torch.float32 (forced via --force-fp32)")
        return torch.float32
    if should_use_bf16(device=device, model_params=model_params, prioritize_performance=False):
        logging.debug(f"ControlNet dtype: torch.bfloat16 (device supports BF16)")
        return torch.bfloat16
    if should_use_fp16(device=device, model_params=model_params, prioritize_performance=False):
        logging.debug(f"ControlNet dtype: torch.float16 (device supports FP16)")
        return torch.float16
    logging.debug(f"ControlNet dtype: torch.float32 (default fallback)")
    return torch.float32

def cleanup_models():
    """Clean up models on finalization."""
    soft_empty_cache(clear=False, caller="cleanup_models")
    # Check memory state after soft clear
    device = get_torch_device()
    cache_key = (device, 'free_memory')
    mem_free_total, _ = _device_cache.get(cache_key, (0, 0))
    if mem_free_total == 0:
        mem_free_total, _ = get_free_memory(device, torch_free_too=True)
    total_vram = get_total_memory(device)
    if mem_free_total < 0.4 * total_vram:
        if PROFILING_ENABLED:
            logging.debug(f"cleanup_models: Insufficient VRAM ({mem_free_total/1024**3:.2f} GB < 20% of {total_vram/1024**3:.2f} GB), forcing aggressive clear")
        soft_empty_cache(clear=True, caller="cleanup_models_aggressive")

# Profiling context manager
@contextlib.contextmanager
def profile_section(name):
    """Context manager for profiling code sections."""
    if PROFILING_ENABLED:
        start = time.time()
        if DEBUG_ENABLED:
            stack = [frame for frame in traceback.format_stack(
                limit=10) if "model_management" in frame]
            logging.debug(f"Starting {name}, stack: {''.join(stack)}")
        try:
            yield
        finally:
            logging.debug(f"{name}: {time.time() - start:.3f} s")
    else:
        yield

def mac_version():
    """Get macOS version if running on macOS."""
    if platform.system() == "Darwin":
        try:
            version = platform.mac_ver()[0]
            version_parts = version.split(".")
            return (int(version_parts[0]), int(version_parts[1]))
        except:
            return None
    return None

# Additional utilities for memory management
def get_device_memory_info(device=None):
    """Get detailed memory information for a device."""
    if device is None:
        device = get_torch_device()
    mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
    mem_total = get_total_memory(device)
    return {
        "free_total": mem_free_total,
        "free_torch": mem_free_torch,
        "total": mem_total,
        "used": mem_total - mem_free_total
    }

def optimize_memory_for_device(device=None):
    """Optimize memory settings based on device capabilities."""
    if device is None:
        device = get_torch_device()
    total_vram = get_total_memory(device) / (1024 * 1024 * 1024)  # VRAM in GB
    global vram_state
    if total_vram < 3.9:
        vram_state = VRAMState.NO_VRAM
        logging.info(f"Low VRAM ({total_vram:.1f} GB), enabling NO_VRAM mode")
    elif total_vram < 7.9:
        vram_state = VRAMState.LOW_VRAM
        logging.info(f"Moderate VRAM ({total_vram:.1f} GB), enabling LOW_VRAM mode")
    else:
        vram_state = VRAMState.NORMAL_VRAM
        logging.info(f"Sufficient VRAM ({total_vram:.1f} GB), using NORMAL_VRAM mode")

# Initialize device and memory settings
try:
    optimize_memory_for_device()
    if PROFILING_ENABLED:
        logging.debug("Memory optimization completed")
except Exception as e:
    logging.error(f"Failed to optimize memory: {e}")
    vram_state = VRAMState.DISABLED

def get_device_cache_state():
    """Return the current state of _device_cache for logging."""
    return _device_cache
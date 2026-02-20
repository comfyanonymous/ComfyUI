"""
MPS (Apple Silicon) compatibility patches for ComfyUI.

Consolidates all MPS-specific workarounds into a single module that
monkey-patches at startup. Only activates when the device is MPS.

Patches applied:
  A. Memory reporting — use mach_task_info phys_footprint for accurate memory
  B. torch.compile interception — prevent dynamo recompilation storms on MPS
  C. Model unload — skip partial unloading on unified memory (zero-copy full reload is faster)
  D. Unified memory unloading — free CPU-resident models when MPS needs space
"""

import ctypes
import logging
import torch
import psutil

log = logging.getLogger(__name__)

_patches_applied = False


def _is_mps_available():
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


# ---------------------------------------------------------------------------
# macOS mach_task_info for accurate physical footprint
# ---------------------------------------------------------------------------
# On Apple Silicon unified memory, neither torch.mps.current_allocated_memory()
# (misses CPU-side tensors) nor psutil.Process().memory_info().rss (misses
# Metal/GPU allocations) gives accurate total memory usage.
#
# mach_task_info(TASK_VM_INFO).phys_footprint is what Activity Monitor reports
# as "Memory" and correctly includes both CPU and GPU on unified memory.
# ---------------------------------------------------------------------------

class _TaskVMInfo(ctypes.Structure):
    """Minimal task_vm_info_data_t up to phys_footprint (macOS/ARM64)."""
    _fields_ = [
        ('virtual_size', ctypes.c_uint64),
        ('region_count', ctypes.c_int32),
        ('page_size', ctypes.c_int32),
        ('resident_size', ctypes.c_uint64),
        ('resident_size_peak', ctypes.c_uint64),
        ('device', ctypes.c_uint64),
        ('device_peak', ctypes.c_uint64),
        ('internal', ctypes.c_uint64),
        ('internal_peak', ctypes.c_uint64),
        ('external', ctypes.c_uint64),
        ('external_peak', ctypes.c_uint64),
        ('reusable', ctypes.c_uint64),
        ('reusable_peak', ctypes.c_uint64),
        ('purgeable_volatile_pmap', ctypes.c_uint64),
        ('purgeable_volatile_resident', ctypes.c_uint64),
        ('purgeable_volatile_virtual', ctypes.c_uint64),
        ('compressed', ctypes.c_uint64),
        ('compressed_peak', ctypes.c_uint64),
        ('compressed_lifetime', ctypes.c_uint64),
        ('phys_footprint', ctypes.c_uint64),
    ]

_TASK_VM_INFO = 22
_TASK_VM_INFO_COUNT = ctypes.sizeof(_TaskVMInfo) // ctypes.sizeof(ctypes.c_uint32)

try:
    _libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
except OSError:
    _libc = None


def _get_phys_footprint():
    """Get the process physical memory footprint via mach_task_info.

    Returns the same value Activity Monitor shows as 'Memory'.
    Includes CPU allocations, GPU/Metal allocations, and compressed pages.
    Returns None if the syscall fails.
    """
    if _libc is None:
        return None
    try:
        info = _TaskVMInfo()
        count = ctypes.c_uint32(_TASK_VM_INFO_COUNT)
        ret = _libc.task_info(
            _libc.mach_task_self(),
            _TASK_VM_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if ret == 0:
            return info.phys_footprint
    except Exception:
        pass
    return None


def _patch_memory_reporting():
    """Patch A: Total memory cap + free memory = system available (no under-report).

    Total memory: we cap at effective_max (0.95 * recommended_max) so ComfyUI
    doesn't assume 48GB is all usable for MPS.

    Free memory: we use min(psutil.virtual_memory().available, effective_max).
    Stock ComfyUI already uses psutil.available for MPS; we only cap it. We
    previously used effective_max - footprint, which under-reported free (e.g.
    9.5 GB when the system had 22 GB), causing slice_attention to use 2–4x
    more steps and turning a 44 min run into 2+ hours. Using actual system
    available restores speed while still capping at our safe budget.
    """
    import comfy.model_management as mm

    recommended_max = torch.mps.recommended_max_memory()
    # 48 GB unified should behave like "at least 24 GB VRAM" — workflows that run on
    # 24 GB NVIDIA run here. We must not report a budget LOWER than 24 GB, or
    # get_free_memory() stays too small after Patch D and we never have "enough" free.
    # Use ~0.95 of recommended_max so we report a usable budget (e.g. 35–37 GB on
    # 48 GB Mac); Patch D unloads CPU models, then we have room for 14B + activations.
    effective_max = int(recommended_max * 0.95)

    # Verify phys_footprint works at startup
    test_footprint = _get_phys_footprint()
    if test_footprint is not None:
        log.info("[MPS] recommended_max: {:.1f} GB, effective budget: {:.1f} GB, "
                 "current footprint: {:.1f} GB (via mach_task_info)".format(
                     recommended_max / (1024**3),
                     effective_max / (1024**3),
                     test_footprint / (1024**3)))
    else:
        log.warning("[MPS] mach_task_info failed — falling back to driver_allocated_memory")
        log.info("[MPS] recommended_max: {:.1f} GB, effective budget: {:.1f} GB".format(
            recommended_max / (1024**3), effective_max / (1024**3)))

    _orig_get_total_memory = mm.get_total_memory

    def patched_get_total_memory(dev=None, torch_total_too=False):
        if dev is None:
            dev = mm.get_torch_device()
        if hasattr(dev, 'type') and dev.type == 'mps':
            mem_total = effective_max
            mem_total_torch = mem_total
            if torch_total_too:
                return (mem_total, mem_total_torch)
            return mem_total
        return _orig_get_total_memory(dev, torch_total_too)

    mm.get_total_memory = patched_get_total_memory

    _orig_get_free_memory = mm.get_free_memory

    def patched_get_free_memory(dev=None, torch_free_too=False):
        if dev is None:
            dev = mm.get_torch_device()
        if hasattr(dev, 'type') and dev.type == 'mps':
            # Use actual system RAM available so we don't under-report. Before this
            # fix we used effective_max - footprint (e.g. 9.5 GB), which made
            # slice_attention use 2–4x more steps → 44 min became 2+ hours. Stock
            # ComfyUI uses psutil.virtual_memory().available for MPS. Cap at
            # effective_max so we never claim more than our safe budget.
            system_available = psutil.virtual_memory().available
            mem_free = min(system_available, effective_max)
            if torch_free_too:
                return (mem_free, mem_free)
            return mem_free
        return _orig_get_free_memory(dev, torch_free_too)

    mm.get_free_memory = patched_get_free_memory

    # Update the module-level total_vram that was already computed with the old function
    mm.total_vram = patched_get_total_memory(mm.get_torch_device()) / (1024 * 1024)
    log.info("[MPS] Corrected total VRAM to {:.0f} MB".format(mm.total_vram))

    return 2  # number of patches applied


def _patch_torch_compile():
    """Patch B: Prevent torch.compile/dynamo from activating on MPS.

    sub_quad attention uses variable-size chunks, causing 64+ dynamo recompiles
    that exhaust memory. Custom nodes (TorchCompileModelWanVideoV2) can bypass
    TORCHDYNAMO_DISABLE=1 by setting torch._dynamo.config.cache_size_limit = 64.

    We intercept at three levels:
    1. torch.compile itself — return model unmodified on MPS
    2. set_torch_compile_wrapper — no-op on MPS
    3. torch._dynamo.config — reset cache_size_limit to 0
    """
    patch_count = 0
    _compile_skip_count = [0]  # mutable container for closure

    # 1. Wrap torch.compile to be a no-op on MPS
    _orig_torch_compile = torch.compile

    def patched_torch_compile(model=None, *args, **kwargs):
        if model is not None and hasattr(model, 'parameters'):
            try:
                device = next(model.parameters()).device
                if device.type == 'mps':
                    _compile_skip_count[0] += 1
                    if _compile_skip_count[0] <= 1:
                        log.info("[MPS] torch.compile skipped — not supported on MPS backend")
                    return model
            except StopIteration:
                pass
        # Also check if default device is MPS
        if _is_mps_available():
            try:
                dev = torch.device('mps')
                if torch.mps.is_available():
                    _compile_skip_count[0] += 1
                    if _compile_skip_count[0] <= 1:
                        log.info("[MPS] torch.compile skipped — MPS is the active backend "
                                 "(further skips will be silent)")
                    return model if model is not None else lambda m: m
            except Exception:
                pass
        return _orig_torch_compile(model, *args, **kwargs)

    torch.compile = patched_torch_compile
    patch_count += 1

    # 2. Patch set_torch_compile_wrapper to no-op on MPS
    try:
        import comfy_api.torch_helpers.torch_compile as tc
        _orig_set_wrapper = tc.set_torch_compile_wrapper

        def patched_set_torch_compile_wrapper(model, *args, **kwargs):
            log.info("[MPS] set_torch_compile_wrapper skipped — torch.compile disabled on MPS")
            return

        tc.set_torch_compile_wrapper = patched_set_torch_compile_wrapper
        patch_count += 1
    except ImportError:
        log.debug("[MPS] comfy_api.torch_helpers.torch_compile not found, skipping patch")

    # 3. Reset dynamo cache_size_limit to prevent custom nodes from re-enabling
    try:
        import torch._dynamo.config as dynamo_config
        dynamo_config.cache_size_limit = 0
        patch_count += 1
    except (ImportError, AttributeError):
        log.debug("[MPS] torch._dynamo.config not available, skipping cache_size_limit reset")

    return patch_count


def _patch_model_unload():
    """Patch C: Skip partial model unloading on MPS unified memory.

    On unified memory (Apple Silicon), partial unloading creates per-layer
    lowvram patches with synchronous CPU<->MPS copies. Full unload + reload
    is faster because CPU<->MPS is essentially zero-copy (same physical memory).
    """
    import comfy.model_management as mm

    _OrigLoadedModel = mm.LoadedModel

    _orig_model_unload = _OrigLoadedModel.model_unload

    def patched_model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None and hasattr(self.device, 'type') and self.device.type == 'mps':
            # On MPS unified memory, skip partial unload — go straight to full unload.
            # CPU<->MPS is zero-copy so full reload has no transfer cost.
            log.debug("[MPS] Skipping partial unload, using full unload (zero-copy unified memory)")
            self.model.detach(unpatch_weights)
            self.model_finalizer.detach()
            self.model_finalizer = None
            self.real_model = None
            return True
        return _orig_model_unload(self, memory_to_free, unpatch_weights)

    _OrigLoadedModel.model_unload = patched_model_unload
    return 1


def _patch_unified_memory_unloading():
    """Patch D: Free CPU-resident models when MPS needs space on unified memory.

    On Apple Silicon, CPU and MPS share the same physical memory pool. ComfyUI's
    free_memory() only considers models on the *same* device (line 617:
    ``if shift_model.device == device``). When loading a 13.6GB MPS model, it only
    finds other MPS-resident models to unload — the 10.8GB CPU text encoder is
    invisible, even though it consumes the same physical RAM.

    This patch wraps free_memory() to also consider CPU-resident models when the
    target device is MPS and the initial unloading pass didn't free enough.
    """
    import gc
    import comfy.model_management as mm

    _orig_free_memory = mm.free_memory

    def patched_free_memory(memory_required, device, keep_loaded=[], for_dynamic=False, ram_required=0):
        original_requested_gb = memory_required / (1024**3) if memory_required else 0
        total_budget = None

        # On MPS, ComfyUI often asks for activation memory (e.g. 68 GB for WAN 14B) on top of
        # model size. Cap to effective budget so we only unload until we have headroom;
        # 48 GB unified with 0.95 * recommended_max gives ~35 GB budget so Patch D can free enough.
        if hasattr(device, 'type') and device.type == 'mps':
            total_budget = mm.get_total_memory(device)
            total_budget_gb = total_budget / (1024**3)

            if memory_required > total_budget:
                log.warning(
                    "[MPS] *** IMPOSSIBLE REQUEST: ComfyUI asked to free %.1f GB on a machine with "
                    "unified memory budget ~%.1f GB. That can never be satisfied. Without capping, "
                    "the loader would still run, total usage (model + text encoder + activations) would "
                    "exceed RAM, and the OS would swap heavily → THRASHING. Capping request to %.1f GB.",
                    original_requested_gb, total_budget_gb, total_budget_gb
                )
                log.info("[MPS] Capping free_memory request {:.1f} GB -> {:.1f} GB (effective budget)".format(
                    original_requested_gb, total_budget_gb))
                memory_required = total_budget

            footprint_before = _get_phys_footprint()
            loaded = [(m.model.model.__class__.__name__, m.device.type, m.model_memory() / (1024**3))
                      for m in mm.current_loaded_models if not m.is_dead()]
            on_cpu = [x for x in loaded if x[1] == 'cpu']
            on_mps = [x for x in loaded if x[1] == 'mps']
            log.info(
                "[MPS] free_memory called: need {:.1f} GB, process footprint {:.1f} GB, "
                "loaded models: {} (MPS: {}, CPU: {})".format(
                    memory_required / (1024**3), (footprint_before or 0) / (1024**3),
                    loaded, len(on_mps), len(on_cpu)
                )
            )
            if on_cpu and total_budget:
                log.warning(
                    "[MPS] *** SAME-DEVICE ONLY: ComfyUI's built-in free_memory() only unloads models on "
                    "the *same* device (MPS). The %s CPU-resident model(s) above (e.g. text encoder ~10 GB) "
                    "use the same physical RAM but are INVISIBLE to that pass. Patch D will unload them "
                    "after the first pass if MPS still needs more.",
                    len(on_cpu)
                )

        # First run the original (handles same-device unloading only)
        result = _orig_free_memory(memory_required, device, keep_loaded, for_dynamic, ram_required)

        # On MPS unified memory, also consider unloading CPU-resident models
        if not (hasattr(device, 'type') and device.type == 'mps'):
            return result

        mem_free = mm.get_free_memory(device)
        if mem_free >= memory_required:
            log.info("[MPS] free_memory: sufficient after same-device pass ({:.1f} GB free)".format(
                mem_free / (1024**3)))
            return result

        # Still not enough — same-device pass couldn't satisfy. Unload CPU-resident models (Patch D).
        log.warning(
            "[MPS] *** SAME-DEVICE PASS INSUFFICIENT: After unloading MPS models we have %.1f GB free, "
            "need %.1f GB. Unloading CPU-resident models now (unified memory) — e.g. text encoder.",
            mem_free / (1024**3), memory_required / (1024**3)
        )
        cpu_device = torch.device('cpu')
        can_unload_cpu = []
        for i in range(len(mm.current_loaded_models) - 1, -1, -1):
            shift_model = mm.current_loaded_models[i]
            if shift_model.device == cpu_device:
                if shift_model not in keep_loaded and not shift_model.is_dead():
                    can_unload_cpu.append((shift_model.model_memory(), i))

        unloaded_indices = []
        for mem_size, i in sorted(can_unload_cpu, reverse=True):
            mem_free = mm.get_free_memory(device)
            if mem_free >= memory_required:
                break
            model_name = mm.current_loaded_models[i].model.model.__class__.__name__
            log.info("[MPS] Unloading CPU-resident {} ({:.1f} GB) to free unified memory".format(
                model_name, mem_size / (1024**3)))
            if mm.current_loaded_models[i].model_unload():
                unloaded_indices.append(i)

        for i in sorted(unloaded_indices, reverse=True):
            result.append(mm.current_loaded_models.pop(i))

        if unloaded_indices:
            gc.collect()
            mm.soft_empty_cache()
            if hasattr(torch.mps, 'empty_cache') and callable(getattr(torch.mps, 'empty_cache', None)):
                torch.mps.empty_cache()
            gc.collect()
            new_free = mm.get_free_memory(device)
            log.info("[MPS] After unified memory cleanup: {:.1f} GB free (needed {:.1f} GB)".format(
                new_free / (1024**3), memory_required / (1024**3)))

        # Warn only when genuinely short; we cap memory_required to total_budget, so we often
        # "need" 35 GB but only have 22 GB — that's still enough for a 13.6 GB model. Warn only
        # when free is below a safe threshold (e.g. 15 GB) so we don't alarm when headroom exists.
        final_free = mm.get_free_memory(device)
        final_free_gb = final_free / (1024**3)
        if final_free < memory_required:
            if final_free_gb < 15.0:
                log.warning(
                    "[MPS] *** THRASHING RISK: After all unloading we have %.1f GB free but need %.1f GB. "
                    "ComfyUI will load anyway; heavy swap likely. Close other apps or reduce resolution/frames.",
                    final_free_gb, memory_required / (1024**3)
                )
            else:
                log.info(
                    "[MPS] After unloading: %.1f GB free (capped request was %.1f GB). Headroom sufficient for load.",
                    final_free_gb, memory_required / (1024**3)
                )
        if total_budget and original_requested_gb > (total_budget / (1024**3)):
            log.info(
                "[MPS] (Original request was %.1f GB; capped to %.1f GB to avoid impossible "
                "target and reduce thrashing risk.)",
                original_requested_gb, total_budget / (1024**3)
            )

        return result

    mm.free_memory = patched_free_memory
    return 1


def apply_patches():
    """Apply all MPS compatibility patches. Only activates on MPS devices.

    Call this after comfy.options.enable_args_parsing() but before model loading.
    """
    global _patches_applied
    if _patches_applied:
        return

    if not _is_mps_available():
        return

    log.info("[MPS] Apple Silicon detected — applying MPS compatibility patches")

    total_patches = 0
    total_patches += _patch_memory_reporting()
    total_patches += _patch_torch_compile()
    total_patches += _patch_model_unload()
    total_patches += _patch_unified_memory_unloading()

    _patches_applied = True
    log.info("[MPS] Applied {} patches".format(total_patches))

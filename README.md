# ComfyUI Global Memory Trim

Global native heap trimming for ComfyUI on Linux/WSL.

This custom node repo installs a small global execution patch when ComfyUI loads custom nodes. The patch can call Python `gc.collect()` and glibc `malloc_trim(0)` before and/or after node execution. It is meant for workflows that repeatedly create large CPU image/video buffers through PyTorch, NumPy, OpenCV, Pillow, or native custom nodes and then stall or wedge under WSL2 memory pressure.

It also provides two optional workflow nodes:

- **Global Memory Trim Now**: manually run a trim and return RSS metrics.
- **Global Memory Trim Status**: return current config and last trim result.

The global patch does **not** require adding either node to your workflow.

## Why this exists

Some WSL2 workloads can stall when native libraries repeatedly allocate and free large CPU buffers. Python objects may be gone, but glibc arenas can retain pages. Under a WSL memory cap, that can trigger heavy reclaim or a hard-looking VM stall. `malloc_trim(0)` asks glibc to return free heap pages to the OS.

This repo is intentionally CPU/native-heap focused. It does **not** directly free CUDA VRAM, unload ComfyUI models, delete ComfyUI caches, or change workflow outputs.

## Installation

From your ComfyUI directory:

```bash
git clone https://github.com/xmarre/ComfyUI-Global-Memory-Trim custom_nodes/ComfyUI-Global-Memory-Trim
```

Or copy this folder into:

```text
ComfyUI/custom_nodes/ComfyUI-Global-Memory-Trim
```

Restart ComfyUI. On startup you should see a log line similar to:

```text
Installed global memory trim patch: enabled=True before=False after=True ...
```

## Performance-oriented WSL setup

This is the current practical setup I use for a large WSL2 ComfyUI workflow with heavy model switching, Flux/SDXL/SeedVR2/detailer passes, and large CPU image buffers.

The important parts are:

- Keep ComfyUI on `--highvram` for performance.
- Disable async weight offload and pinned memory on WSL.
- Do **not** force `--disable-cuda-malloc` here; the normal CUDA allocator path avoids the VRAM over-reservation/overflow seen with the native allocator path in this workflow.
- Keep `PYTORCH_CUDA_ALLOC_CONF` unset.
- Use glibc trim thresholds and the global trim hook to reduce CPU/native heap retention.
- Keep SeedVR2 BF16 forced on if using the patched SeedVR2 import probe workaround and wanting the higher-quality 7B path.

```bash
#!/usr/bin/env bash
set -e

_hold_terminal_on_failure() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '\nComfyUI launcher exited with status %d\n' "$rc" >&2
    printf 'Dropping into interactive shell so the terminal stays open.\n' >&2
    exec bash -i
  fi
}
trap _hold_terminal_on_failure EXIT

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comfy312

# Native/CPU heap behavior. These do not free CUDA VRAM directly.
export MALLOC_MMAP_THRESHOLD_=65536
export MALLOC_TRIM_THRESHOLD_=65536

# Global trim hook.
# BEFORE=1 is more aggressive and can help before large model/node transitions.
# LOG=1 is useful while validating. Set it to 0 once stable.
export COMFYUI_GLOBAL_TRIM=1
export COMFYUI_GLOBAL_TRIM_AFTER=1
export COMFYUI_GLOBAL_TRIM_BEFORE=1
export COMFYUI_GLOBAL_TRIM_GC=1
export COMFYUI_GLOBAL_TRIM_INTERVAL=1
export COMFYUI_GLOBAL_TRIM_LOG=1
export COMFYUI_GLOBAL_TRIM_MIN_RSS_MB=8192

# Optional, workflow-specific: keep SeedVR2 on BF16 without running an import-time CUDA probe.
export SEEDVR2_FORCE_BFLOAT16=1
unset SEEDVR2_IMPORT_BFLOAT16_PROBE

# Do not force PyTorch's allocator through the environment.
unset PYTORCH_CUDA_ALLOC_CONF

# Optional, workflow-specific memory reduction for SuperBeasts.
export SUPERBEASTS_SPCA_RETURN_RESIDUALS=false
export SUPERBEASTS_HDR_MALLOC_TRIM=true

export PYTHONFAULTHANDLER=1

cd ~/ComfyUI

set +e
python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --fast fp16_accumulation \
  --highvram \
  --use-pytorch-cross-attention \
  --disable-async-offload \
  --disable-pinned-memory \
  "$@"
status=$?
set -e

exit "$status"
```

### After validating stability

Once the workflow is stable, reduce log overhead first:

```bash
export COMFYUI_GLOBAL_TRIM_LOG=0
```

Then, if performance still needs tuning, test one change at a time:

```bash
export COMFYUI_GLOBAL_TRIM_BEFORE=0
```

or:

```bash
export COMFYUI_GLOBAL_TRIM_INTERVAL=2
```

If wedges return, restore the previous value.

## Conservative diagnostic WSL setup

For reproducing or isolating CPU/native heap stalls, use the more conservative version below. It clamps native CPU thread pools and limits glibc arenas, which can improve WSL stability but may slow CPU-heavy nodes.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled

export MALLOC_ARENA_MAX=1
export MALLOC_MMAP_THRESHOLD_=65536
export MALLOC_TRIM_THRESHOLD_=65536

export COMFYUI_GLOBAL_TRIM=1
export COMFYUI_GLOBAL_TRIM_AFTER=1
export COMFYUI_GLOBAL_TRIM_BEFORE=0
export COMFYUI_GLOBAL_TRIM_GC=1
export COMFYUI_GLOBAL_TRIM_INTERVAL=1
export COMFYUI_GLOBAL_TRIM_LOG=0
export COMFYUI_GLOBAL_TRIM_MIN_RSS_MB=8192
```

Use this when the problem is clearly CPU/native memory pressure rather than VRAM pressure.

## Configuration

All configuration is via environment variables.

| Variable | Default | Meaning |
|---|---:|---|
| `COMFYUI_GLOBAL_TRIM` | `1` | Enable/disable the global patch. |
| `COMFYUI_GLOBAL_TRIM_AFTER` | `1` | Trim after node execution. |
| `COMFYUI_GLOBAL_TRIM_BEFORE` | `0` | Also trim before node execution. More aggressive, useful for testing or fragile WSL setups. |
| `COMFYUI_GLOBAL_TRIM_GC` | `1` | Run `gc.collect()` before `malloc_trim(0)`. |
| `COMFYUI_GLOBAL_TRIM_INTERVAL` | `1` | Trim every N trim opportunities. Use `2`, `4`, etc. to reduce overhead. |
| `COMFYUI_GLOBAL_TRIM_MIN_RSS_MB` | `0` | Only trim when process RSS is at least this value. `0` means always. |
| `COMFYUI_GLOBAL_TRIM_LOG` | `0` | Log every trim with RSS before/after. Very noisy; enable only while diagnosing. |
| `COMFYUI_GLOBAL_TRIM_WARN_NO_LIBC` | `1` | Warn when glibc `malloc_trim` cannot be loaded. |

## Notes

- Linux/WSL only. On non-Linux platforms the patch becomes a no-op.
- `malloc_trim(0)` only returns already-free native heap pages. It does not free live tensors, ComfyUI outputs, model weights, or Python objects that are still referenced.
- This is **not** a VRAM fixer. It targets CPU/native heap retention.
- `--disable-cuda-malloc` can change CUDA allocator behavior and may increase VRAM reservation/fragmentation in some workflows. Do not assume it is safer unless you specifically need it.
- `--disable-async-offload` and `--disable-pinned-memory` can be useful on WSL when async offload/pinned-memory paths cause wedges.
- `COMFYUI_GLOBAL_TRIM_LOG=1` is diagnostic only. Turn it off for normal use.

## License

MIT

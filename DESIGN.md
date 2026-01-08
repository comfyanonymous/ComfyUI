# Disk tier safetensors streaming design audit (ComfyUI)

## Mandatory research audit (verified call sites)

### ComfyUI load path + eager materialization sites
- `comfy/utils.py:load_torch_file` currently uses `safetensors.safe_open` and iterates all keys to build a full `sd` dict (eager tensor materialization). It also returns metadata only after reading all tensors.【F:comfy/utils.py†L58-L93】
- `comfy/utils.py:calculate_parameters` and `weight_dtype` iterate `sd.keys()` and then access `sd[k]` to compute `nelement()`/`dtype` (loads tensors).【F:comfy/utils.py†L109-L128】
- `comfy/utils.py:state_dict_prefix_replace` mutates dicts by `pop`+assignment (materializes if used on a streaming mapping).【F:comfy/utils.py†L135-L144】
- `comfy/model_base.py:BaseModel.load_model_weights` builds `to_load = {}` by iterating keys and popping tensors, then passes a fully materialized dict to `load_state_dict` (RAM spike).【F:comfy/model_base.py†L301-L318】
- `comfy/model_detection.py` reads `state_dict[key].shape` in many branches for detection (must be metadata-only). Example: `calculate_transformer_depth` and numerous `detect_unet_config` branches read shapes directly from `state_dict` values.【F:comfy/model_detection.py†L21-L200】
- `comfy/sd.py` loads checkpoints, then slices, renames, and computes parameters/dtypes by reading tensors (e.g., `calculate_parameters`, `weight_dtype`, `process_*_state_dict`, and special scaled-FP8 conversion that builds new dicts).【F:comfy/sd.py†L1304-L1519】
- Direct safetensors load outside `load_torch_file`: `comfy/sd1_clip.py:load_embed` and `nodes.py:LoadLatent.load` use `safetensors.torch.load_file`, bypassing the core loader.【F:comfy/sd1_clip.py†L432-L434】【F:nodes.py†L521-L529】

### FastSageTensors (fastsafetensors) capability audit
- Header parsing and metadata:
  - `fastsafetensors/common.py:SafeTensorsMetadata` parses the header and builds per-tensor `TensorFrame` with `dtype`, `shape`, and `data_offsets` (no tensor allocation).【F:../third_party/fastsafetensors-main/fastsafetensors/common.py†L63-L187】
  - `TensorFrame` stores dtype/shape/offsets and supports slicing metadata.【F:../third_party/fastsafetensors-main/fastsafetensors/common.py†L238-L338】
- GDS + no-GDS low-level readers:
  - `fastsafetensors/cpp.pyi` exposes `gds_file_reader`, `gds_file_handle`, `nogds_file_reader`, `cpu_malloc`, `gpu_malloc`, and alignment helpers such as `get_alignment_size()`.【F:../third_party/fastsafetensors-main/fastsafetensors/cpp.pyi†L1-L43】
  - GDS availability checks are in `fastsafetensors/cpp.pyi`: `is_gds_supported`, `is_cufile_found`, `cufile_version`, and `init_gds`.【F:../third_party/fastsafetensors-main/fastsafetensors/cpp.pyi†L36-L43】
- DLPack wrapping:
  - `fastsafetensors/dlpack.py` provides `from_cuda_buffer()` which creates DLPack capsules for both CPU and GPU buffers via a device descriptor and is used for `torch.from_dlpack`.【F:../third_party/fastsafetensors-main/fastsafetensors/dlpack.py†L232-L239】
- Torch framework interop:
  - `fastsafetensors/frameworks/_torch.py:TorchOp` provides `alloc_tensor_memory`/`free_tensor_memory`, dtype mapping, and uses `torch.from_dlpack` for wrapping raw pointers into tensors.【F:../third_party/fastsafetensors-main/fastsafetensors/frameworks/_torch.py†L131-L205】

### VRAM/RAM offload logic (for extension)
- `comfy/model_management.py` handles VRAM/RAM offload via `free_memory` and keeps tracking of loaded/offloaded memory (needs integration for RAM disk tier).【F:comfy/model_management.py†L584-L612】
- `comfy/model_patcher.py` implements module-by-module offload/low-vram weight casting (`comfy_cast_weights`) and partial unload/load (needs to integrate disk tier for RAM eviction).【F:comfy/model_patcher.py†L663-L955】

## Strategy summary (implemented)

### Streaming safetensors mapping (no full dict materialization)
- [x] Introduce a new module `comfy/safetensors_stream.py` with:
  - [x] `TensorMeta` and `SafeTensorIndex` (metadata-only parsing with `fastsafetensors.SafeTensorsMetadata`).
  - [x] `StreamStateDict` as a mapping backed by `SafeTensorIndex`, exposing metadata-only `keys()`/`__iter__` and loading tensors on demand.
  - [x] Lightweight mapping views: `PrefixViewStateDict`, `FilterViewStateDict`, `RenameViewStateDict` for lazy prefix/filter/rename without eager loading.

### Range reads and tiering
- [x] Disk→RAM: use `fastsafetensors.cpp.nogds_file_reader` for range reads and wrap with DLPack.
- [x] Disk→GPU (GDS): use `gds_file_reader` + `gds_file_handle` to read the aligned range directly into GPU memory. If GDS is requested but not supported (e.g., `is_gds_supported==0` or libcufile missing), raise a hard error with instructions to disable GDS.
- [x] Disk→RAM→GPU: read only the tensor range into (optionally pinned) CPU memory, copy to GPU, then release CPU buffer unless RAM cache policy keeps it.

### Disk tier integration
- [x] Represent disk-resident weights as meta tensors (`device='meta'`) plus a `DiskRef` registry that stores `(module, param_name) -> TensorMeta + loader handle`.
- [x] Add an LRU cache for RAM-resident weights loaded from disk with configurable max bytes. Eviction replaces RAM tensors with meta tensors and keeps `DiskRef` for reload.
- [x] Add a general `forward_pre_hook` to materialize any meta+DiskRef weights before compute; this covers modules that bypass `comfy.ops`.
- [x] Add budgeted module materialization (`DiskMaterializationState`) with per-module loaded/deferred tracking, deterministic ordering, and RAM/VRAM free-memory checks (no insufficient-memory exceptions).
- [x] Add dtype-aware disk loads (override based on forward input/manual cast) to avoid matmul dtype mismatches in on-demand materialization.
- [x] Add disk-tier logging with destination, load size, free memory, partial/full state, and per-device byte breakdowns.

### Pipeline refactors
- [x] Update `load_torch_file` to return `StreamStateDict` for `.safetensors`/`.sft` and return metadata without loading.
- [x] Update helpers (`calculate_parameters`, `weight_dtype`, `state_dict_prefix_replace`) to be metadata-aware and lazy.
- [x] Update `BaseModel.load_model_weights` and other load paths to avoid building large dicts; use streaming mappings + view wrappers instead.
- [x] Update model detection (`comfy/model_detection.py`) to use metadata-based shape/dtype access (no tensor reads).
- [x] Update direct safetensors loaders (e.g., `comfy/sd1_clip.py`) to go through `load_torch_file` so everything uses the same streaming loader.
- [x] Add chunked nogds safetensors reads with a configurable staging size and incremental CPU tensor fill to cap staging buffers.
- [x] Restore full `MutableMapping` behavior (including `meta`) for view wrappers like `RenameViewStateDict`.

### Tests and docs
- [x] Add unit tests for metadata correctness, single-tensor loading, and lazy views (no full materialization), plus integration tests for load behavior and GDS failure path.
- [x] Document new flags for RAM cache size and GPUDirect enablement and how to disable GDS when unsupported.

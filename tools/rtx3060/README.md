# MiniMax-H3 RTX 3060 mode

This profile is the local RTX 3060 12 GB fast path for MiniMax-H3 FL2VA.

## Install or restore pinned custom nodes

Run once after a fresh ComfyUI install or after replacing `custom_nodes`:

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/rtx3060/install-custom-nodes.ps1
```

## Start

Run:

```text
tools/rtx3060/start-minimax-h3-rtx3060.cmd
```

The server listens on `http://127.0.0.1:8188` and loads only the custom nodes needed by this workflow:

- `ComfyUI-MiniMax-H3-Turbo`
- `ComfyUI-sol-attn`
- `ComfyMath`

Use `start-minimax-h3-rtx3060.ps1 -IncludeSpectrum` only for a separate quality workflow. Spectrum is intentionally not active in the Turbo6 main path.

## Workflow

```text
user/default/workflows/MiniMax-H3_FL2VA_RTX3060_TURBO6_FFN4.json
```

Defaults:

- MiniMax-H3 FL2VA W4 ConvRot Offload
- Turbo LoRA strength `1.0`
- MiniMax-H3 Turbo sampler
- `6` scheduler steps
- FFN chunking enabled: `4` chunks, `4096` minimum tokens
- `16:9`, `0.2 MP`, `2 seconds`
- PyTorch cross-attention
- async weight offload: `1` stream
- reserve VRAM: `1.0 GB`
- preview disabled and node cache disabled

The warning below is expected on RTX 3060 / Windows:

```text
[Sol-Attn] not loaded (ModuleNotFoundError: No module named 'triton')
```

The Triton Sol attention kernel is not used. The independent `MiniMaxH3ChunkFeedForward` node still loads and was verified through `/object_info`.

## Installed revisions

- ComfyUI: `b1693ecba9f5b65f8c80ab36b195ab963ec92413`
- Turbo node: `96cc1ddc001617da132dd73f31cd43666bf1d8d4`
- Sol-Attn node: `e2fc225f8642585cfa11a31d52fe7b2db7290efa`
- Spectrum node: `85ec1da66277e893079ecd46e32cc865c56cfe53`
- Workflow SHA-256: `6a461a651dbbd3aa019837bfa8c4575ca380cb9092a9af3e9f13ae2b00200183`
- Turbo LoRA SHA-256: `82d0acff583b04ad9a4238a7440b584b56094bfb7c4fdb2981f67c7a4784b62d`

## Startup optimization

`apply-transformers-startup-cache.ps1` applies an idempotent local patch for Transformers 5.14.1. It avoids a full Python environment metadata scan and caches the generated Transformers lazy-import structure.

Measured on this installation:

- cold cache build: `99.38 s`
- cache hit import: `1.44 s`

A ComfyUI or Transformers update may replace the installed package file. The launcher checks and reapplies the patch only when the expected source markers match; otherwise it stops instead of modifying an unknown version.

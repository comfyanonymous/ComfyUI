# Hardware & Software Compatibility

This project is rigorously tested on specific hardware and software configurations to ensure stability and performance.

## Compatibility Matrix

### Linux

| Hardware | Python | CUDA / ROCm | PyTorch | Torch-TensorRT | Container Image | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA RTX 3090** (24GB) | 3.12 | 12.9.1 | Latest | 2.8.0a0 | `nvcr.io/nvidia/pytorch:25.06-py3` | ✅ Automated |
| **NVIDIA RTX 3090** (24GB) | 3.12 | 12.8.1 | Latest | 2.7.0a0 | `nvcr.io/nvidia/pytorch:25.03-py3` (LTS) | ✅ Automated |
| **NVIDIA RTX 3090** (24GB) | 3.10 | 12.6.2 | Latest | 2.5.0a0 | `nvcr.io/nvidia/pytorch:24.10-py3` | ✅ Automated |
| **NVIDIA RTX 3090** (24GB) | 3.10 | 12.3.2 | Latest | 2.2.0a0 | `nvcr.io/nvidia/pytorch:23.12-py3` | ✅ Automated |
| **AMD RX 7600** (8GB) | 3.12 | ROCm 7.0 | 2.7.1 (Nightly) | N/A | `rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.7.1` | ✅ Automated |

**AMD Note:** Automated testing for AMD uses a specific nightly build of PyTorch 2.7.1 optimized for RDNA 3 (`gfx110X`) from `https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/`.

### macOS

| Hardware | Python | Acceleration | Status |
| :--- | :--- | :--- | :--- |
| **Apple Silicon** (M1/M2/M3) | 3.12 | MPS (Metal Performance Shaders) | ✅ Automated (macOS 14 Runner) |

### Windows

Windows support is manually verified.

| Hardware | Python | CUDA | Drivers | PyTorch | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA RTX 3090** | 3.10 - 3.12 | 12.8 | 560+ | 2.7, 2.8, 2.9 | ✅ Manually Verified |

## AMD ROCm Support for Other Architectures

You can install ComfyUI with acceleration on other AMD architectures by pointing `uv` to the correct package index.

### Architecture Table

Find your GPU in the table below to determine the correct index URL.

| Series | Models (Examples) | Architecture | Index URL |
| :--- | :--- | :--- | :--- |
| **RX 9000** | RX 9070 / XT, RX 9060 / XT | RDNA 4 (`gfx1200`, `gfx1201`) | `https://rocm.nightlies.amd.com/v2/gfx120X-all/` |
| **RX 7000** | RX 7900 XTX, 7800 XT, 7600 | RDNA 3 (`gfx1100`, `gfx1101`, `gfx1102`) | `https://rocm.nightlies.amd.com/v2/gfx110X-all/` |
| **RX 7000 (M)** | Radeon 780M (Laptop), 7700S | RDNA 3 (`gfx1103`) | `https://rocm.nightlies.amd.com/v2/gfx110X-all/` |
| **Strix Halo** | Strix Halo iGPU | RDNA 3.5 (`gfx1151`) | `https://rocm.nightlies.amd.com/v2/gfx1151/` |
| **Instinct** | MI300A, MI300X | CDNA 3 (`gfx942`) | `https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/` |
| **Instinct** | MI350X, MI355X | CDNA 4 (`gfx950`) | `https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/` |

### Installation Examples

Use `uv pip install` with the `--index-url` corresponding to your hardware.

**RX 9000 Series (RDNA 4)**
```bash
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ --pre torch torchaudio torchvision
```

**RX 7000 Series (RDNA 3)**
```bash
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ --pre torch torchaudio torchvision
```

**RX 6000 Series (RDNA 2)**
```bash
uv pip install --torch-backend=auto --pre torch torchvision torchaudio
```

**RX 5000 Series (RDNA 1)**
```bash
uv pip install --torch-backend=auto --pre torch torchvision torchaudio
```

**Instinct MI300**
```bash
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ --pre torch torchaudio torchvision
```

## Notes

- **NVIDIA:** Automated testing uses official NVIDIA PyTorch containers to ensure compatibility with the latest deep learning stack.
- **AMD:** Automated testing targets ROCm 7.0 on RDNA 3 architecture (RX 7000 series).
- **macOS:** Tested on macOS 14 runners with Python 3.12 using the `mps` backend for acceleration.
- **Windows:** While not part of the automated CI loop, Windows builds are manually verified against recent PyTorch and CUDA versions on standard consumer hardware.
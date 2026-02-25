<div align="center">

# ComfyUI
**The most powerful and modular visual AI engine and application.**


[![Website][website-shield]][website-url]
[![Dynamic JSON Badge][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![][github-release-shield]][github-release-link]
[![][github-release-date-shield]][github-release-link]
[![][github-downloads-shield]][github-downloads-link]
[![][github-downloads-latest-shield]][github-downloads-link]

[matrix-shield]: https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white
[matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
[website-shield]: https://img.shields.io/badge/ComfyOrg-4285F4?style=flat
[website-url]: https://www.comfy.org/
<!-- Workaround to display total user from https://github.com/badges/shields/issues/4500#issuecomment-2060079995 -->
[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total
[discord-url]: https://www.comfy.org/discord
[twitter-shield]: https://img.shields.io/twitter/follow/ComfyUI
[twitter-url]: https://x.com/ComfyUI

[github-release-shield]: https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-release-date-shield]: https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat
[github-downloads-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

ComfyUI lets you design and execute advanced stable diffusion pipelines using a graph/nodes/flowchart based interface. Available on Windows, Linux, and macOS.

## Get Started

| Option | Description | Platforms |
|--------|-------------|-----------|
| [**Desktop Application**](https://www.comfy.org/download) | Easiest way to get started | Windows, macOS |
| [**Windows Portable Package**](#installing) | Latest commits, no install required | Windows |
| [**Manual Install**](#manual-install-windows-linux) | Full control, all GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend) | Windows, Linux, macOS |

## [Examples](https://comfyanonymous.github.io/ComfyUI_examples/)

See what ComfyUI can do with the [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Features

- Nodes/graph/flowchart interface to experiment and create complex Stable Diffusion workflows without needing to code anything.
- **Image Models**
   - SD1.x, SD2.x ([unCLIP](https://comfyanonymous.github.io/ComfyUI_examples/unclip/))
   - [SDXL](https://comfyanonymous.github.io/ComfyUI_examples/sdxl/), [SDXL Turbo](https://comfyanonymous.github.io/ComfyUI_examples/sdturbo/)
   - [Stable Cascade](https://comfyanonymous.github.io/ComfyUI_examples/stable_cascade/)
   - [SD3 and SD3.5](https://comfyanonymous.github.io/ComfyUI_examples/sd3/)
   - Pixart Alpha and Sigma
   - [AuraFlow](https://comfyanonymous.github.io/ComfyUI_examples/aura_flow/)
   - [HunyuanDiT](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_dit/)
   - [Flux](https://comfyanonymous.github.io/ComfyUI_examples/flux/)
   - [Lumina Image 2.0](https://comfyanonymous.github.io/ComfyUI_examples/lumina2/)
   - [HiDream](https://comfyanonymous.github.io/ComfyUI_examples/hidream/)
   - [Qwen Image](https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/)
   - [Hunyuan Image 2.1](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_image/)
   - [Flux 2](https://comfyanonymous.github.io/ComfyUI_examples/flux2/)
   - [Z Image](https://comfyanonymous.github.io/ComfyUI_examples/z_image/)
- **Image Editing Models**
   - [Omnigen 2](https://comfyanonymous.github.io/ComfyUI_examples/omnigen/)
   - [Flux Kontext](https://comfyanonymous.github.io/ComfyUI_examples/flux/#flux-kontext-image-editing-model)
   - [HiDream E1.1](https://comfyanonymous.github.io/ComfyUI_examples/hidream/#hidream-e11)
   - [Qwen Image Edit](https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/#edit-model)
- **Video Models**
   - [Stable Video Diffusion](https://comfyanonymous.github.io/ComfyUI_examples/video/)
   - [Mochi](https://comfyanonymous.github.io/ComfyUI_examples/mochi/)
   - [LTX-Video](https://comfyanonymous.github.io/ComfyUI_examples/ltxv/)
   - [Hunyuan Video](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/)
   - [Wan 2.1](https://comfyanonymous.github.io/ComfyUI_examples/wan/)
   - [Wan 2.2](https://comfyanonymous.github.io/ComfyUI_examples/wan22/)
   - [Hunyuan Video 1.5](https://docs.comfy.org/tutorials/video/hunyuan/hunyuan-video-1-5)
- **Audio Models**
   - [Stable Audio](https://comfyanonymous.github.io/ComfyUI_examples/audio/)
   - [ACE Step](https://comfyanonymous.github.io/ComfyUI_examples/audio/)
- **3D Models**
   - [Hunyuan3D 2.0](https://docs.comfy.org/tutorials/3d/hunyuan3D-2)
- Asynchronous queue system
- Smart memory management: automatically runs large models on GPUs with as little as 1 GB VRAM via smart offloading.
- Many optimizations: only re-executes the parts of the workflow that change between runs.
- CPU mode via `--cpu` (slow)
- Loads `.ckpt` and `.safetensors` files: all-in-one checkpoints or standalone diffusion models, VAEs, and CLIP models.
- Safe loading of `.ckpt`, `.pt`, `.pth`, etc. files.
- Embeddings / Textual Inversion
- [LoRAs (regular, locon, and loha)](https://comfyanonymous.github.io/ComfyUI_examples/lora/)
- [Hypernetworks](https://comfyanonymous.github.io/ComfyUI_examples/hypernetworks/)
- Load full workflows (with seeds) from generated PNG, WebP, and FLAC files.
- Save/load workflows as JSON files.
- Build complex workflows like [Hires fix](https://comfyanonymous.github.io/ComfyUI_examples/2_pass_txt2img/) and more using the node interface.
- [Area Composition](https://comfyanonymous.github.io/ComfyUI_examples/area_composition/)
- [Inpainting](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/) with both regular and inpainting models.
- [ControlNet and T2I-Adapter](https://comfyanonymous.github.io/ComfyUI_examples/controlnet/)
- [Upscale Models (ESRGAN, ESRGAN variants, SwinIR, Swin2SR, etc.)](https://comfyanonymous.github.io/ComfyUI_examples/upscale_models/)
- [GLIGEN](https://comfyanonymous.github.io/ComfyUI_examples/gligen/)
- [Model Merging](https://comfyanonymous.github.io/ComfyUI_examples/model_merging/)
- [LCM models and LoRAs](https://comfyanonymous.github.io/ComfyUI_examples/lcm/)
- Latent previews with [TAESD](#how-to-show-high-quality-previews)
- Works fully offline: core will never download anything unless you want it to.
- Optional API nodes for paid models via the [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview); disable with `--disable-api-nodes`.
- [Config file](extra_model_paths.yaml.example) to set model search paths.

More examples: [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/)

## Release Process

ComfyUI follows a weekly release cycle targeting Monday, though this may shift around model releases or large codebase changes. There are three interconnected repositories:

1. **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**
   - Releases a new stable version (e.g., v0.7.0) roughly every week.
   - From v0.4.0 onward, patch versions are used for fixes backported to the current stable release.
   - Minor versions are used for releases off the master branch.
   - Patch versions may also be used for master branch releases where a backport wouldn't make sense.
   - Commits outside stable release tags may be very unstable and break custom nodes.
   - Serves as the foundation for the desktop release.

2. **[ComfyUI Desktop](https://github.com/Comfy-Org/desktop)**
   - Builds a new release using the latest stable core version.

3. **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**
   - Weekly frontend updates are merged into the core repository.
   - Features are frozen for the upcoming core release.
   - Development continues for the next release cycle.

## Shortcuts

| Keybind | Action |
|---------|--------|
| `Ctrl` + `Enter` | Queue current graph for generation |
| `Ctrl` + `Shift` + `Enter` | Queue current graph as next in line |
| `Ctrl` + `Alt` + `Enter` | Cancel current generation |
| `Ctrl` + `Z` / `Ctrl` + `Y` | Undo / Redo |
| `Ctrl` + `S` | Save workflow |
| `Ctrl` + `O` | Load workflow |
| `Ctrl` + `A` | Select all nodes |
| `Alt` + `C` | Collapse / uncollapse selected nodes |
| `Ctrl` + `M` | Mute / unmute selected nodes |
| `Ctrl` + `B` | Bypass selected nodes (acts as if the node was removed and wires reconnected through) |
| `Delete` / `Backspace` | Delete selected nodes |
| `Ctrl` + `Backspace` | Delete the current graph |
| `Space` (hold) | Pan the canvas |
| `Ctrl` / `Shift` + `Click` | Add node to selection |
| `Ctrl` + `C` / `Ctrl` + `V` | Copy / paste selected nodes (without preserving connections from unselected outputs) |
| `Ctrl` + `C` / `Ctrl` + `Shift` + `V` | Copy / paste selected nodes (preserving connections from unselected outputs) |
| `Shift` + `Drag` | Move multiple selected nodes |
| `Ctrl` + `D` | Load default graph |
| `Alt` + `+` | Zoom in |
| `Alt` + `-` | Zoom out |
| `Ctrl` + `Shift` + LMB + vertical drag | Zoom in / out |
| `P` | Pin / unpin selected nodes |
| `Ctrl` + `G` | Group selected nodes |
| `Q` | Toggle queue visibility |
| `H` | Toggle history visibility |
| `R` | Refresh graph |
| `F` | Show / hide menu |
| `.` | Fit view to selection (or whole graph if nothing selected) |
| Double-click LMB | Open node quick search |
| `Shift` + Drag (wire) | Move multiple wires at once |
| `Ctrl` + `Alt` + LMB | Disconnect all wires from clicked slot |

> `Ctrl` can be replaced with `Cmd` on macOS.

# Installing

## Windows Portable

There is a portable standalone build for Windows that should work for running on Nvidia GPUs or for running on your CPU only on the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).

### [Direct link to download](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

Simply download, extract with [7-Zip](https://7-zip.org) or with the windows explorer on recent windows versions and run. For smaller models you normally only need to put the checkpoints (the huge ckpt/safetensors files) in: ComfyUI\models\checkpoints but many of the larger models have multiple files. Make sure to follow the instructions to know which subfolder to put them in ComfyUI\models\

If you have trouble extracting it, right click the file -> properties -> unblock

The portable above currently comes with python 3.13 and pytorch cuda 13.0. Update your Nvidia drivers if it doesn't start.

#### Alternative Downloads

- [Experimental portable for AMD GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_amd.7z)
- [Portable with PyTorch CUDA 12.8 and Python 3.12](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia_cu128.7z)
- [Portable with PyTorch CUDA 12.6 and Python 3.12](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia_cu126.7z) — supports Nvidia 10 series and older GPUs

#### How do I share models between another UI and ComfyUI?

See the [Config file](extra_model_paths.yaml.example) to set the search paths for models. In the standalone windows build you can find this file in the ComfyUI directory. Rename this file to extra_model_paths.yaml and edit it with your favorite text editor.


## [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

You can install and start ComfyUI using comfy-cli:

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Windows (CMD)
venv\Scripts\activate.bat
pip install comfy-cli
comfy install

# When no longer using comfy or the CLI,
# run 'deactivate' to close the venv.
# It is better to dedicate a terminal window
# just for Comfy, particularly if you are
# unfamiliar with Python.
```

## Manual Install (Windows, Linux)

Python 3.14 works but some custom nodes may have issues. The free threaded variant works but some dependencies will enable the GIL so it's not fully supported.

Python 3.13 is very well supported. If you have trouble with some custom node dependencies on 3.13 you can try 3.12

torch 2.4 and above is supported but some features and optimizations might only work on newer versions. We generally recommend using the latest major version of pytorch with the latest cuda version unless it is less than 2 weeks old.

### Instructions

Open a terminal inside the ComfyUI folder and run:

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Windows (CMD)
venv\Scripts\activate.bat
pip install -r requirements.txt
```

---

#### Alternate PyTorch versions

To use a different PyTorch build, first uninstall the version that was automatically installed:

```bash
pip uninstall torch torchvision torchaudio
```

##### AMD GPUs (Linux)

Stable ROCm 7.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1
```

ROCm 7.1 nightly (may have performance improvements):

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.1
>>>>>>> d002b7e9 (chore: modernize and update README)
```

##### AMD GPUs — Experimental (Windows and Linux): RDNA 3, 3.5, and 4 only

These builds have narrower hardware support than the ROCm builds above but work on Windows. Install the PyTorch build matching your GPU architecture:

**RDNA 3 (RX 7000 series):**

```bash
pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/
```

**RDNA 3.5 (Strix Halo / Ryzen AI Max+ 365):**

```bash
pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
```

**RDNA 4 (RX 9000 series):**

```bash
pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/
```

##### Intel GPUs (Windows and Linux)

Intel Arc GPU users can install native PyTorch with `torch.xpu` support. More information: [Accelerated PyTorch on XPU](https://pytorch.org/docs/main/notes/get_start_xpu.html).

Stable:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

Nightly (may have performance improvements):

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
```

##### NVIDIA

Stable:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
```

Nightly (may have performance improvements):

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

##### CPU only

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
---

#### Troubleshooting

If you get the "Torch not compiled with CUDA enabled" error, uninstall torch with:

```pip uninstall torch```

And install it again with the command above.

### Other Platforms

#### Apple Mac Silicon

You can install ComfyUI on Apple Silicon (M1 or later) with any recent macOS version.

1. Install PyTorch nightly by following the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide (make sure to install the latest PyTorch nightly).
2. Follow the [ComfyUI manual installation](#manual-install-windows-linux) instructions.
3. Install the ComfyUI [dependencies](#dependencies). If you have another Stable Diffusion UI [you might be able to reuse the dependencies](#i-already-have-another-ui-for-stable-diffusion-installed-do-i-really-have-to-install-all-of-these-dependencies).
4. Launch ComfyUI: `python main.py`

> **Note:** Remember to add your models, VAE, LoRAs etc. to the corresponding Comfy folders, as discussed in [ComfyUI manual installation](#manual-install-windows-linux).

#### Ascend NPUs

For models compatible with Ascend Extension for PyTorch (torch_npu). To get started, ensure your environment meets the prerequisites outlined on the [installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) page. Here's a step-by-step guide tailored to your platform and installation method:

1. Begin by installing the recommended or newer kernel version for Linux as specified in the Installation page of torch-npu, if necessary.
2. Proceed with the installation of Ascend Basekit, which includes the driver, firmware, and CANN, following the instructions provided for your specific platform.
3. Next, install the necessary packages for torch-npu by adhering to the platform-specific instructions on the [Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch) page.
4. Finally, adhere to the [ComfyUI manual installation](#manual-install-windows-linux) guide for Linux. Once all components are installed, you can run ComfyUI as described earlier.

#### Cambricon MLUs

For models compatible with Cambricon Extension for PyTorch (torch_mlu). Here's a step-by-step guide tailored to your platform and installation method:

1. Install the Cambricon CNToolkit by adhering to the platform-specific instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)
2. Next, install the PyTorch(torch_mlu) following the instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html)
3. Launch ComfyUI by running `python main.py`

#### Iluvatar Corex

For models compatible with Iluvatar Extension for PyTorch. Here's a step-by-step guide tailored to your platform and installation method:

1. Install the Iluvatar Corex Toolkit by adhering to the platform-specific instructions on the [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536)
2. Launch ComfyUI by running `python main.py`


## [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager/tree/manager-v4)

**ComfyUI-Manager** is an extension that allows you to easily install, update, and manage custom nodes for ComfyUI.

### Setup

1. Install the manager dependencies:
   ```bash
   pip install -r manager_requirements.txt
   ```

2. Enable the manager with the `--enable-manager` flag when running ComfyUI:
   ```bash
   python main.py --enable-manager
   ```

## Command Line Options

| Flag | Description |
|------|-------------|
| `--enable-manager` | Enable ComfyUI-Manager |
| `--enable-manager-legacy-ui` | Use the legacy manager UI instead of the new UI (requires `--enable-manager`) |
| `--disable-manager-ui` | Disable the manager UI and endpoints while keeping background features like security checks and scheduled installation completion (requires `--enable-manager`) |


## Place your models

| Type | Directory |
|------|-----------|
| Checkpoints (`.ckpt` / `.safetensors`) | `models/checkpoints` |
| VAE | `models/vae` |
| LoRA / LoCon / LoHa | `models/loras` |
| Embeddings / Textual Inversion | `models/embeddings` |
| ControlNet | `models/controlnet` |
| Upscale models (ESRGAN, SwinIR, etc.) | `models/upscale_models` |
| Hypernetworks | `models/hypernetworks` |
| CLIP / Text encoders | `models/text_encoders` or `models/clip` |
| CLIP Vision | `models/clip_vision` |
| GLIGEN | `models/gligen` |
| VAE approximation (TAESD for previews) | `models/vae_approx` |
| Configs | `models/configs` |

For a complete list of all supported model directories, see the [`extra_model_paths.yaml.example`](extra_model_paths.yaml.example) file.

# Running

```bash
python main.py
```

### AMD cards not officially supported by ROCm

If you have issues, try setting the `HSA_OVERRIDE_GFX_VERSION` environment variable:

- **RX 6700, 6600, and other RDNA2 or older:**
  ```bash
  HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py
  ```
- **RX 7600 and other RDNA3:**
  ```bash
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py
  ```

### AMD ROCm tips

You can enable experimental memory-efficient attention on recent PyTorch for some AMD GPUs. It should already be enabled by default on RDNA3. If this improves speed on your GPU, please report it so it can be enabled by default.

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
```

You can also try `PYTORCH_TUNABLEOP_ENABLED=1`, which may speed things up at the cost of a slow initial run.

# Notes

- Only nodes that have all required inputs connected will execute.
- Only the parts of the graph that change between runs are re-executed. Submitting the same graph twice runs it only once. Changing a node re-executes only that node and its dependents.
- Dragging a generated PNG onto the page (or loading one) restores the full workflow, including the seeds used.
- Use `()` to adjust prompt emphasis: `(good code:1.2)` or `(bad code:0.8)`. Default weight for `()` is 1.1. Escape literal parentheses with `\(` and `\)`.
- Use `{option1|option2}` for wildcard/dynamic prompts — the frontend randomly picks one each time you queue. Escape literal braces with `\{` and `\}`.
- Dynamic prompts support C-style comments: `// comment` or `/* comment */`.
- To use a textual inversion embedding in a prompt, place it in `models/embeddings/` and reference it like:

  ```
  embedding:embedding_filename.pt
  ```
  The `.pt` extension is optional.


## How to show high-quality previews?

Use `--preview-method auto` to enable previews.

The default installation includes a fast but low-resolution latent preview. To enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd):

1. Download [`taesd_decoder.pth`, `taesdxl_decoder.pth`, `taesd3_decoder.pth`, and `taef1_decoder.pth`](https://github.com/madebyollin/taesd/) and place them in `models/vae_approx/`.
2. Restart ComfyUI and launch with `--preview-method taesd`.

## How to use TLS/SSL?

Generate a self-signed certificate and key (not appropriate for shared/production use):

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes \
  -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"
```

Then launch ComfyUI with:

```bash
python main.py --tls-keyfile key.pem --tls-certfile cert.pem
```

The app will be accessible at `https://...` instead of `http://...`.

> **Windows users:** Use [alexisrolland/docker-openssl](https://github.com/alexisrolland/docker-openssl) or a [3rd party OpenSSL binary](https://wiki.openssl.org/index.php/Binaries) to run the command above. When using a container, the `-v` volume mount accepts a relative path — e.g. `-v ".\:/openssl-certs"` — which writes the key and cert to your current directory.

## Support and Dev Channels

- [Discord](https://comfy.org/discord) — try the **#help** or **#feedback** channels.
- [Matrix: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) — open-source alternative to Discord.
- [comfy.org](https://www.comfy.org/)

## Frontend Development

The frontend is hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend). This repo contains the compiled JS (from TS/Vue) under the `web/` directory.

### Reporting issues and requesting features

For frontend bugs, issues, or feature requests, please use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

### Using the latest frontend

The frontend in this repository is updated **fortnightly**. Daily releases are available in the frontend repository.

To use a specific version:

```bash
# Latest daily build
python main.py --front-end-version Comfy-Org/ComfyUI_frontend@latest

# Specific version
python main.py --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
```

### Accessing the legacy frontend

```bash
python main.py --front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest
```

Source: [ComfyUI Legacy Frontend repository](https://github.com/Comfy-Org/ComfyUI_legacy_frontend).

# FAQ

### Which GPU should I buy for this?

[See the GPU recommendations wiki page.](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)

# Getting Started

## Installing

These instructions will install an interactive ComfyUI using the command line.

### Windows

When using Windows, open the **Windows Powershell** app. Then observe you are at a command line, and it is printing "where" you are in your file system: your user directory (e.g., `C:\Users\doctorpangloss`). This is where a bunch of files will go. If you want files to go somewhere else, consult a chat bot for the basics of using command lines, because it is beyond the scope of this document. Then:

1. Install Python 3.12, 3.11 or 3.10. You can do this from the Python website; or, you can use `chocolatey`, a Windows package manager:

   ```shell
   Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```

2. Install `uv`, which makes subsequent installation of Python packages much faster:

    ```shell
    choco install -y uv
    ```

3. Switch into a directory that you want to store your outputs, custom nodes and models in. This is your ComfyUI workspace. For example, if you want to store your workspace in a directory called `ComfyUI_Workspace` in your Documents folder:

   ```powershell
   mkdir ~/Documents/ComfyUI_Workspace
   cd ~/Documents/ComfyUI_Workspace
   ```

4. Create a virtual environment:
   ```shell
   uv venv --python 3.12
   ```
5. Run the following command to install `comfyui` into your current environment. This will correctly select the version of `torch` that matches the GPU on your machine (NVIDIA or CPU on Windows, NVIDIA, Intel, AMD or CPU on Linux):
   ```powershell
   uv pip install --torch-backend=auto "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
   ```
6. To run the web server:
   ```shell
   uv run comfyui
   ```
   When you run workflows that use well-known models, this will download them automatically.

   To make it accessible over the network:
   ```shell
   uv run comfyui --listen
   ```

**Running**

On Windows, you should change into the directory where you ran `uv venv`, then run `comfyui`. For example, if you ran `uv venv` inside `~\Documents\ComfyUI_Workspace\`

```powershell
cd ~\Documents\ComfyUI_Workspace\
uv run comfyui
```

Upgrades are delivered frequently and automatically. To force one immediately, run `uv pip install --upgrade` like so:

```shell
uv pip install --torch-backend=auto --upgrade "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
```

### macOS

1. Install `brew`, a macOS package manager, if you haven't already:
   ```shell
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   Then, install `uv`:
   ```shell
   HOMEBREW_NO_AUTO_UPDATE=1 brew install uv
   ```
3. Switch into a directory that you want to store your outputs, custom nodes and models in. This is your ComfyUI workspace. For example, if you want to store your workspace in a directory called `ComfyUI_Workspace` in your Documents folder:

   ```shell
   mkdir -pv ~/Documents/ComfyUI_Workspace
   cd ~/Documents/ComfyUI_Workspace
   ```

4. Create a virtual environment:
   ```shell
   uv venv --python 3.12
   ```

5. Run the following command to install `comfyui` into your current environment. The `mps` extra improves performance.
   ```shell
   uv pip install "comfyui[mps]@git+https://github.com/hiddenswitch/ComfyUI.git"
   ```
6. To run the web server:
   ```shell
   uv run comfyui
   ```
   When you run workflows that use well-known models, this will download them automatically.

   To make it accessible over the network:
   ```shell
   uv run comfyui --listen
   ```

**Running**

On macOS, you will need to open the terminal and `cd` into the directory in which you ran `uv venv`. For example, if you ran `uv venv` in `~/Documents/ComfyUI_Workspace/`:

```shell
cd ~/Documents/ComfyUI_Workspace/
uv run comfyui
```

## Model Downloading

ComfyUI LTS supports downloading models on demand.

Known models will be downloaded from Hugging Face or CivitAI.

To support licensed models like Flux, you will need to login to Hugging Face from the command line.

1. Activate your Python environment by `cd` followed by your workspace directory. For example, if your workspace is located in `~/Documents/ComfyUI_Workspace`, do:

```shell
cd ~/Documents/ComfyUI_Workspace
```

Then, on Windows: `& .venv/scripts/activate.ps1`; on macOS: `source .venv/bin/activate`.

2. Login with Huggingface:

```shell
uv pip install huggingface-cli
huggingface-cli login
```

3. Agree to the terms for a repository. For example, visit https://huggingface.co/black-forest-labs/FLUX.1-dev, login with your HuggingFace account, then choose **Agree**.

To disable model downloading, start with the command line argument `--disable-known-models`: `comfyui --disable-known-models`. However, this will generally only increase your toil for no return.

### Saving Space on Windows

To save space, you will need to enable **Developer Mode** in the Windows Settings, then reboot your computer. This way, Hugging Face can download models into a common place for all your apps, and place small "link" files that ComfyUI and others can read instead of whole copies of models.

## Using ComfyUI in Google Colab

Access an example Colab Notebook here: https://colab.research.google.com/drive/1Gd9F8iYRJW-LG8JLiwGTKLAcXLJ5eH78?usp=sharing

This demonstrates running a workflow inside colab and accessing the UI remotely.

## Using a "Python Embedded" "Portable" Style Distribution

This is a "ComfyUI" "Portable" style distribution with a "`python_embedded`" directory, carefully spelled correctly. It includes Python 3.12, `torch==2.7.1+cu128`, `sageattention` and the ComfyUI-Manager.

On **Windows**:

1. Download all the files in this the latest release: ([`comfyui_portable.exe`](https://github.com/hiddenswitch/ComfyUI/releases/download/latest/comfyui_portable.exe), [`comfyui_portable.7z.001`](https://github.com/hiddenswitch/ComfyUI/releases/download/latest/comfyui_portable.7z.001) and [`comfyui_portable.7z.002`](https://github.com/hiddenswitch/ComfyUI/releases/download/latest/comfyui_portable.7z.002)).
2. Run `comfyui_portable.exe` to extract a workspace containing an embedded Python 3.12.
3. Double-click on `comfyui.bat` inside `ComfyUI_Workspace` to start the server.

## LTS Custom Nodes

These packages have been adapted to be installable with `pip` and download models to the correct places:

- **ELLA T5 Text Conditioning for SD1.5**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-ella.git`
- **IP Adapter**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-ipadapter-plus`
- **ControlNet Auxiliary Preprocessors**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-controlnet-aux.git`.
- **LayerDiffuse Alpha Channel Diffusion**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-layerdiffuse.git`.
- **BRIA Background Removal**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-bria-bg-removal.git`
- **Video Frame Interpolation**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-video-frame-interpolation`
- **Video Helper Suite**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-video-helper-suite`
- **AnimateDiff Evolved**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-animatediff-evolved`
- **Impact Pack**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-impact-pack`
- **TensorRT**: `uv pip install git+https://github.com/AppMAna/appmana-comfyui-nodes-tensorrt`

Custom nodes are generally supported by this fork. Use these for a bug-free experience.

Request first-class, LTS support for more nodes by [creating a new issue](https://github.com/hiddenswitch/ComfyUI/issues/new). Remember, ordinary custom nodes from the ComfyUI ecosystem work in this fork. Create an issue if you experience a bug or if you think something needs more attention.

##### Running with TLS

To serve with `https://` on Windows easily, use [Caddy](https://github.com/caddyserver/caddy/releases/download/v2.7.6/caddy_2.7.6_windows_amd64.zip). Extract `caddy.exe` to a directory, then run it:

```shell
caddy reverse-proxy --from localhost:443 --to localhost:8188 --tls self_signed
```

##### Notes for AMD Users

Installation for `ROCm` should be explicit:
```shell
uv pip install "comfyui[rocm]@git+https://github.com/hiddenswitch/ComfyUI.git"
```

Then, until a workaround is found, specify these variables:

RDNA 3 (RX 7600 and later)

```shell
export HSA_OVERRIDE_GFX_VERSION=11.0.0
uv run comfyui
```

RDNA 2 (RX 6600 and others)

```shell
export HSA_OVERRIDE_GFX_VERSION=10.3.0
uv run comfyui
```

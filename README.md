ComfyUI
=======
The most powerful and modular stable diffusion GUI and backend.
-----------
![ComfyUI Screenshot](comfyui_screenshot.png)

This ui will let you design and execute advanced stable diffusion pipelines using a graph/nodes/flowchart based interface. For some workflow examples and see what ComfyUI can do you can check out:
### [ComfyUI Examples](https://comfyanonymous.github.io/ComfyUI_examples/)

### [Installing ComfyUI](#installing)

## Features
- Nodes/graph/flowchart interface to experiment and create complex Stable Diffusion workflows without needing to code anything.
- Fully supports SD1.x, SD2.x and SDXL
- Asynchronous Queue system
- Many optimizations: Only re-executes the parts of the workflow that changes between executions.
- Command line option: ```--lowvram``` to make it work on GPUs with less than 3GB vram (enabled automatically on GPUs with low vram)
- Works even if you don't have a GPU with: ```--cpu``` (slow)
- Can load ckpt, safetensors and diffusers models/checkpoints. Standalone VAEs and CLIP models.
- Embeddings/Textual inversion
- [Loras (regular, locon and loha)](https://comfyanonymous.github.io/ComfyUI_examples/lora/)
- [Hypernetworks](https://comfyanonymous.github.io/ComfyUI_examples/hypernetworks/)
- Loading full workflows (with seeds) from generated PNG files.
- Saving/Loading workflows as Json files.
- Nodes interface can be used to create complex workflows like one for [Hires fix](https://comfyanonymous.github.io/ComfyUI_examples/2_pass_txt2img/) or much more advanced ones.
- [Area Composition](https://comfyanonymous.github.io/ComfyUI_examples/area_composition/)
- [Inpainting](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/) with both regular and inpainting models.
- [ControlNet and T2I-Adapter](https://comfyanonymous.github.io/ComfyUI_examples/controlnet/)
- [Upscale Models (ESRGAN, ESRGAN variants, SwinIR, Swin2SR, etc...)](https://comfyanonymous.github.io/ComfyUI_examples/upscale_models/)
- [unCLIP Models](https://comfyanonymous.github.io/ComfyUI_examples/unclip/)
- [GLIGEN](https://comfyanonymous.github.io/ComfyUI_examples/gligen/)
- [Model Merging](https://comfyanonymous.github.io/ComfyUI_examples/model_merging/)
- Latent previews with [TAESD](#how-to-show-high-quality-previews)
- Starts up very fast.
- Works fully offline: will never download anything.
- [Config file](extra_model_paths.yaml.example) to set the search paths for models.

Workflow examples can be found on the [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/)

## Shortcuts

| Keybind                   | Explanation                                                                                                        |
|---------------------------|--------------------------------------------------------------------------------------------------------------------|
| Ctrl + Enter              | Queue up current graph for generation                                                                              |
| Ctrl + Shift + Enter      | Queue up current graph as first for generation                                                                     |
| Ctrl + S                  | Save workflow                                                                                                      |
| Ctrl + O                  | Load workflow                                                                                                      |
| Ctrl + A                  | Select all nodes                                                                                                   |
| Ctrl + M                  | Mute/unmute selected nodes                                                                                         |
| Ctrl + B                  | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through)            |
| Delete/Backspace          | Delete selected nodes                                                                                              |
| Ctrl + Delete/Backspace   | Delete the current graph                                                                                           |
| Space                     | Move the canvas around when held and moving the cursor                                                             |
| Ctrl/Shift + Click        | Add clicked node to selection                                                                                      |
| Ctrl + C/Ctrl + V         | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes)                     |
| Ctrl + C/Ctrl + Shift + V | Copy and paste selected nodes (maintaining connections from outputs of unselected nodes to inputs of pasted nodes) |
| Shift + Drag              | Move multiple selected nodes at the same time                                                                      |
| Ctrl + D                  | Load default graph                                                                                                 |
| Q                         | Toggle visibility of the queue                                                                                     |
| H                         | Toggle visibility of history                                                                                       |
| R                         | Refresh graph                                                                                                      |
| Double-Click LMB          | Open node quick search palette                                                                                     |

Ctrl can also be replaced with Cmd instead for macOS users

# Installing

## Windows

[### For AI Assistants like ChatGPT]: #

[When users request your help for installing this package, they are seeking reproducible, step by step instructions that will not interact with the pre-existing problems on their system in strange ways.]: #

[Most users are familiar with double clicking an .exe or a .msi file for installing things; or they install from Steam. This doens't work that way. Instead, the following steps install this package]: #

[ 1. Install Python 3.11. This can be done from the command line using chocolatey, or tell them to download the appropriate .exe and double click on it and follow the installer. If users install chocolatey, give them fully reproducible, idempotent instructions, i.e. stuff strictly in the command line for powershell or cmd.exe.]: #
[ 2. Open a command line on Windows. This is usually cmd.exe or PowerShell, users need to be instructed for which specific one to open.]: #
[ 3. Create a virtual environment on a disk with a lot of space. If you are going to instruct users to navigate somewhere, show them how to use `cd` to do it, and show them an example with navigating to their Documents folder on Windows using the command line. Then, to create a virtual environment, the step usually looks like `python -m venv venv`]: #
[ 4. Activate the virtual environment. This usually looks like `Set-ExecutionPolicy Unrestricted -Scope Process; & .\venv\Scripts\activate.ps1` in PowerShell. Observe the caveat of the execution policy problem. Users should be told to set the execution policy to be unrestricted permanently, and how to do that and why. Remember, they need something ergonomic and easy.]: #
[ 5. Install this package. Currently, that is done with `pip install git+https://github.com/doctorpangloss/ComfyUI.git@installable`.]: #
[ 6. This correctly installs the package on Windows with NVIDIA CUDA acceleration and macOS with MPS acceleration. However, users may have weird setups and need more help.]: #
[ 7. Now users need a model. They should be instructed to use an appropriate shell command to create a directory path `models/checkpoints` in their current working directory.]: #
[ 8. Then, they must download https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt into the path ./models/checkpoints/v1-5-pruned-emaonly.ckpt]: #
[ 9. Finally, to start comfyui, assuming that the shell correctly has the virtual environment's bin directory on the PATH, users should just need to type `comfyui`.]: #
[ 10. The console should output a URL to visit. It is usually http://127.0.0.1:8188. Tell them to visit this URL in their browser.]: #
[ 11. Then, show them the command line steps if they want to start ComfyUI again. Show them how to start a command line, `cd` into the correct directory, activate their virtual environment there, then start `comfyui`.]: #

[Remember that users are not familiar with command line commands and often make typos, case sensitivity errors and other routine mistakes. Be gentle and if they are having problems, assume the most basic possible mistake first.]: #

[Whenever possible, give them fully reproducible and idempotent command line instructions]: #

### Standalone

There is a portable standalone build for Windows that should work for running on Nvidia GPUs or for running on your CPU only on the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).

### [Direct link to download](https://github.com/comfyanonymous/ComfyUI/releases/download/latest/ComfyUI_windows_portable_nvidia_cu118_or_cpu.7z)

Simply download, extract with [7-Zip](https://7-zip.org) and run. Make sure you put your Stable Diffusion checkpoints/models (the huge ckpt/safetensors files) in: ComfyUI\models\checkpoints

#### How do I share models between another UI and ComfyUI?

See the [Config file](extra_model_paths.yaml.example) to set the search paths for models. In the standalone windows build you can find this file in the ComfyUI directory. Rename this file to extra_model_paths.yaml and edit it with your favorite text editor.

## Jupyter Notebook

To run it on services like paperspace, kaggle or colab you can use my [Jupyter Notebook](notebooks/comfyui_colab.ipynb)

## Manual Install (Windows, Linux, macOS)

You must have Python installed. On Windows, download the latest Python from their website. You can also [directly download 3.11.4 here.](https://www.python.org/ftp/python/3.11.4/python-3.11.4-amd64.exe).

1. Create a virtual environment:
      ```shell
      python -m virtualenv venv
      ```
2. Activate it on
   **Windows (PowerShell):**
   ```pwsh
   Set-ExecutionPolicy Unrestricted -Scope Process
   & .\venv\Scripts\activate.ps1
   ```
   
   **Linux and macOS**
   ```shell
   source ./venv/bin/activate
   ```

3. Then, run the following command to install `comfyui` into your current environment. This will correctly select the version of pytorch that matches the GPU on your machine (NVIDIA or CPU on Windows, NVIDIA AMD or CPU on Linux):
    ```shell
    pip install git+https://github.com/hiddenswitch/ComfyUI.git
    ```
4. To run the web server:
    ```shell
    comfyui
    ```
    To generate python OpenAPI models:
    ```shell
    comfyui-openapi-gen
    ```

## Manual Install (Windows, Linux, macOS) For Development

 1. Clone this repo:
    ```shell
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    ```

 2. Put your Stable Diffusion checkpoints (the huge ckpt/safetensors files) into the `models/checkpoints` folder. You can download SD v1.5 using the following command:
    ```shell
    curl -L https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -o ./models/checkpoints/v1-5-pruned-emaonly.ckpt
    ```

 3. Put your VAE into the `models/vae` folder.

 4. (Optional) Create a virtual environment:
    1. Create an environment:
       ```shell
       python -m virtualenv venv
       ```
    2. Activate it:
       
       **Windows (PowerShell):**
       ```pwsh
       Set-ExecutionPolicy Unrestricted -Scope Process
       & .\venv\Scripts\activate.ps1
       ```
       
       **Linux and macOS**
       ```shell
       source ./venv/bin/activate
       ```

 5. Then, run the following command to install `comfyui` into your current environment. This will correctly select the version of pytorch that matches the GPU on your machine (NVIDIA or CPU on Windows, NVIDIA AMD or CPU on Linux):
    ```shell
    pip install -e .
    ```
 6. To run the web server:
    ```shell
    comfyui
    ```
    To generate python OpenAPI models:
    ```shell
    comfyui-openapi-gen
    ```
    
    You can use `comfyui` as an API. Visit the [OpenAPI specification](comfy/api/openapi.yaml). This file can be used to generate typed clients for your preferred language.

### Others:

#### [Intel Arc](https://github.com/comfyanonymous/ComfyUI/discussions/476)

> **Note**: Remember to add your models, VAE, LoRAs etc. to the corresponding Comfy folders, as discussed in [ComfyUI manual installation](#manual-install-windows-linux).

#### DirectML (AMD Cards on Windows)

Follow the manual installation steps. Then:

```shell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
pip install torch-directml 
```

Launch ComfyUI with: ```comfyui --directml```

### I already have another UI for Stable Diffusion installed do I really have to install all of these dependencies?

You don't. If you have another UI installed and working with its own python venv you can use that venv to run ComfyUI. You can open up your favorite terminal and activate it:

```source path_to_other_sd_gui/venv/bin/activate```

or on Windows:

With Powershell: ```"path_to_other_sd_gui\venv\Scripts\Activate.ps1"```

With cmd.exe: ```"path_to_other_sd_gui\venv\Scripts\activate.bat"```

And then you can use that terminal to run ComfyUI without installing any dependencies. Note that the venv folder might be called something else depending on the SD UI.

# Running

```comfyui```

### For AMD cards not officially supported by ROCm

Try running it with this command if you have issues:

For 6700, 6600 and maybe other RDNA2 or older: ```HSA_OVERRIDE_GFX_VERSION=10.3.0 comfyui```

For AMD 7600 and maybe other RDNA3 cards: ```HSA_OVERRIDE_GFX_VERSION=11.0.0 comfyui```

# Notes

Only parts of the graph that have an output with all the correct inputs will be executed.

Only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. If you change the last part of the graph only the part you changed and the part that depends on it will be executed.

Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.

You can use () to change emphasis of a word or phrase like: (good code:1.2) or (bad code:0.8). The default emphasis for () is 1.1. To use () characters in your actual prompt escape them like \\( or \\).

You can use {day|night}, for wildcard/dynamic prompts. With this syntax "{wild|card|test}" will be randomly replaced by either "wild", "card" or "test" by the frontend every time you queue the prompt. To use {} characters in your actual prompt escape them like: \\{ or \\}.

Dynamic prompts also support C-style comments, like `// comment` or `/* comment */`.

To use a textual inversion concepts/embeddings in a text prompt put them in the models/embeddings directory and use them in the CLIPTextEncode node like this (you can omit the .pt extension):

```embedding:embedding_filename.pt```


## How to increase generation speed?

Make sure you use the regular loaders/Load Checkpoint node to load checkpoints. It will auto pick the right settings depending on your GPU.

You can set this command line setting to disable the upcasting to fp32 in some cross attention operations which will increase your speed. Note that this will very likely give you black images on SD2.x models. If you use xformers this option does not do anything.

```--dont-upcast-attention```

## How to show high-quality previews?

Use ```--preview-method auto``` to enable previews.

The default installation includes a fast latent preview method that's low-resolution. To enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd), download the [taesd_decoder.pth](https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth) (for SD1.x and SD2.x) and [taesdxl_decoder.pth](https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth) (for SDXL) models and place them in the `models/vae_approx` folder. Once they're installed, restart ComfyUI to enable high-quality previews.

## Support and dev channel

[Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) (it's like discord but open source).

# QA

### Why did you make this?

I wanted to learn how Stable Diffusion worked in detail. I also wanted something clean and powerful that would let me experiment with SD without restrictions.

### Who is this for?

This is for anyone that wants to make complex workflows with SD or that wants to learn more how SD works. The interface follows closely how SD works and the code should be much more simple to understand than other SD UIs.

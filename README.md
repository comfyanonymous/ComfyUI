ComfyUI
=======
A powerful and modular stable diffusion GUI and backend.
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

There is a portable standalone build for Windows that should work for running on Nvidia GPUs or for running on your CPU only on the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).

### [Direct link to download](https://github.com/comfyanonymous/ComfyUI/releases/download/latest/ComfyUI_windows_portable_nvidia_cu118_or_cpu.7z)

Simply download, extract with [7-Zip](https://7-zip.org) and run. Make sure you put your Stable Diffusion checkpoints/models (the huge ckpt/safetensors files) in: ComfyUI\models\checkpoints

#### How do I share models between another UI and ComfyUI?

See the [Config file](extra_model_paths.yaml.example) to set the search paths for models. In the standalone windows build you can find this file in the ComfyUI directory. Rename this file to extra_model_paths.yaml and edit it with your favorite text editor.

## Colab Notebook

To run it on colab or paperspace you can use my [Colab Notebook](notebooks/comfyui_colab.ipynb) here: [Link to open with google colab](https://colab.research.google.com/github/comfyanonymous/ComfyUI/blob/master/notebooks/comfyui_colab.ipynb)

## Manual Install (Windows, Linux)

Git clone this repo.

Put your SD checkpoints (the huge ckpt/safetensors files) in: models/checkpoints

Put your VAE in: models/vae

### AMD GPUs (Linux only)
AMD users can install rocm and pytorch with pip if you don't have it already installed, this is the command to install the stable version:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.4.2```

This is the command to install the nightly with ROCm 5.6 that supports the 7000 series and might have some performance improvements:
```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.6```

### NVIDIA

Nvidia users should install torch and xformers using this command:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 xformers```

#### Troubleshooting

If you get the "Torch not compiled with CUDA enabled" error, uninstall torch with:

```pip uninstall torch```

And install it again with the command above.

### Dependencies

Install the dependencies by opening your terminal inside the ComfyUI folder and:

```pip install -r requirements.txt```

After this you should have everything installed and can proceed to running ComfyUI.

### Others:

#### [Intel Arc](https://github.com/comfyanonymous/ComfyUI/discussions/476)

#### Apple Mac silicon

You can install ComfyUI in Apple Mac silicon (M1 or M2) with any recent macOS version.

1. Install pytorch nightly. For instructions, read the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide (make sure to install the latest pytorch nightly).
1. Follow the [ComfyUI manual installation](#manual-install-windows-linux) instructions for Windows and Linux.
1. Install the ComfyUI [dependencies](#dependencies). If you have another Stable Diffusion UI [you might be able to reuse the dependencies](#i-already-have-another-ui-for-stable-diffusion-installed-do-i-really-have-to-install-all-of-these-dependencies).
1. Launch ComfyUI by running `python main.py --force-fp16`. Note that --force-fp16 will only work if you installed the latest pytorch nightly.

> **Note**: Remember to add your models, VAE, LoRAs etc. to the corresponding Comfy folders, as discussed in [ComfyUI manual installation](#manual-install-windows-linux).

#### DirectML (AMD Cards on Windows)

```pip install torch-directml``` Then you can launch ComfyUI with: ```python main.py --directml```

### I already have another UI for Stable Diffusion installed do I really have to install all of these dependencies?

You don't. If you have another UI installed and working with its own python venv you can use that venv to run ComfyUI. You can open up your favorite terminal and activate it:

```source path_to_other_sd_gui/venv/bin/activate```

or on Windows:

With Powershell: ```"path_to_other_sd_gui\venv\Scripts\Activate.ps1"```

With cmd.exe: ```"path_to_other_sd_gui\venv\Scripts\activate.bat"```

And then you can use that terminal to run ComfyUI without installing any dependencies. Note that the venv folder might be called something else depending on the SD UI.

# Running

```python main.py```

### For AMD cards not officially supported by ROCm

Try running it with this command if you have issues:

For 6700, 6600 and maybe other RDNA2 or older: ```HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py```

For AMD 7600 and maybe other RDNA3 cards: ```HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py```

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

<div align="center">

# 4090 ComfyUI
**This is a fork of ComfyUI designed specifically to make the often painful process of installing Comfy easy.**


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

## I. Features
- Nodes/graph/flowchart interface to experiment and create complex Stable Diffusion workflows without needing to code anything.
- Image Models
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
   - [Cosmos Predict2](https://comfyanonymous.github.io/ComfyUI_examples/cosmos_predict2/)
   - [Qwen Image](https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/)
- Image Editing Models
   - [Omnigen 2](https://comfyanonymous.github.io/ComfyUI_examples/omnigen/)
   - [Flux Kontext](https://comfyanonymous.github.io/ComfyUI_examples/flux/#flux-kontext-image-editing-model)
   - [HiDream E1.1](https://comfyanonymous.github.io/ComfyUI_examples/hidream/#hidream-e11)
- Video Models
   - [Stable Video Diffusion](https://comfyanonymous.github.io/ComfyUI_examples/video/)
   - [Mochi](https://comfyanonymous.github.io/ComfyUI_examples/mochi/)
   - [LTX-Video](https://comfyanonymous.github.io/ComfyUI_examples/ltxv/)
   - [Hunyuan Video](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/)
   - [Nvidia Cosmos](https://comfyanonymous.github.io/ComfyUI_examples/cosmos/) and [Cosmos Predict2](https://comfyanonymous.github.io/ComfyUI_examples/cosmos_predict2/)
   - [Wan 2.1](https://comfyanonymous.github.io/ComfyUI_examples/wan/)
   - [Wan 2.2](https://comfyanonymous.github.io/ComfyUI_examples/wan22/)
- Audio Models
   - [Stable Audio](https://comfyanonymous.github.io/ComfyUI_examples/audio/)
   - [ACE Step](https://comfyanonymous.github.io/ComfyUI_examples/audio/)
- 3D Models
   - [Hunyuan3D 2.0](https://docs.comfy.org/tutorials/3d/hunyuan3D-2)
- Asynchronous Queue system
- Many optimizations: Only re-executes the parts of the workflow that changes between executions.
- Smart memory management: can automatically run large models on GPUs with as low as 1GB vram with smart offloading.
- Works even if you don't have a GPU with: ```--cpu``` (slow)
- Can load ckpt and safetensors: All in one checkpoints or standalone diffusion models, VAEs and CLIP models.
- Safe loading of ckpt, pt, pth, etc.. files.
- Embeddings/Textual inversion
- [Loras (regular, locon and loha)](https://comfyanonymous.github.io/ComfyUI_examples/lora/)
- [Hypernetworks](https://comfyanonymous.github.io/ComfyUI_examples/hypernetworks/)
- Loading full workflows (with seeds) from generated PNG, WebP and FLAC files.
- Saving/Loading workflows as Json files.
- Nodes interface can be used to create complex workflows like one for [Hires fix](https://comfyanonymous.github.io/ComfyUI_examples/2_pass_txt2img/) or much more advanced ones.
- [Area Composition](https://comfyanonymous.github.io/ComfyUI_examples/area_composition/)
- [Inpainting](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/) with both regular and inpainting models.
- [ControlNet and T2I-Adapter](https://comfyanonymous.github.io/ComfyUI_examples/controlnet/)
- [Upscale Models (ESRGAN, ESRGAN variants, SwinIR, Swin2SR, etc...)](https://comfyanonymous.github.io/ComfyUI_examples/upscale_models/)
- [GLIGEN](https://comfyanonymous.github.io/ComfyUI_examples/gligen/)
- [Model Merging](https://comfyanonymous.github.io/ComfyUI_examples/model_merging/)
- [LCM models and Loras](https://comfyanonymous.github.io/ComfyUI_examples/lcm/)
- Latent previews with [TAESD](#how-to-show-high-quality-previews)
- Works fully offline: core will never download anything unless you want to.
- Optional API nodes to use paid models from external providers through the online [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).
- [Config file](extra_model_paths.yaml.example) to set the search paths for models.

Workflow examples can be found on the [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/)

## II. Shortcuts

| Keybind                            | Explanation                                                                                                        |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | Queue up current graph for generation                                                                              |
| `Ctrl` + `Shift` + `Enter`              | Queue up current graph as first for generation                                                                     |
| `Ctrl` + `Alt` + `Enter`                | Cancel current generation                                                                                          |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | Undo/Redo                                                                                                          |
| `Ctrl` + `S`                          | Save workflow                                                                                                      |
| `Ctrl` + `O`                          | Load workflow                                                                                                      |
| `Ctrl` + `A`                          | Select all nodes                                                                                                   |
| `Alt `+ `C`                           | Collapse/uncollapse selected nodes                                                                                 |
| `Ctrl` + `M`                          | Mute/unmute selected nodes                                                                                         |
| `Ctrl` + `B`                           | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through)            |
| `Delete`/`Backspace`                   | Delete selected nodes                                                                                              |
| `Ctrl` + `Backspace`                   | Delete the current graph                                                                                           |
| `Space`                              | Move the canvas around when held and moving the cursor                                                             |
| `Ctrl`/`Shift` + `Click`                 | Add clicked node to selection                                                                                      |
| `Ctrl` + `C`/`Ctrl` + `V`                  | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes)                     |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | Copy and paste selected nodes (maintaining connections from outputs of unselected nodes to inputs of pasted nodes) |
| `Shift` + `Drag`                       | Move multiple selected nodes at the same time                                                                      |
| `Ctrl` + `D`                           | Load default graph                                                                                                 |
| `Alt` + `+`                          | Canvas Zoom in                                                                                                     |
| `Alt` + `-`                          | Canvas Zoom out                                                                                                    |
| `Ctrl` + `Shift` + LMB + Vertical drag | Canvas Zoom in/out                                                                                                 |
| `P`                                  | Pin/Unpin selected nodes                                                                                           |
| `Ctrl` + `G`                           | Group selected nodes                                                                                               |
| `Q`                                 | Toggle visibility of the queue                                                                                     |
| `H`                                  | Toggle visibility of history                                                                                       |
| `R`                                  | Refresh graph                                                                                                      |
| `F`                                  | Show/Hide menu                                                                                                      |
| `.`                                  | Fit view to selection (Whole graph when nothing is selected)                                                        |
| Double-Click LMB                   | Open node quick search palette                                                                                     |
| `Shift` + Drag                       | Move multiple wires at once                                                                                        |
| `Ctrl` + `Alt` + LMB                   | Disconnect all wires from clicked slot                                                                             |

`Ctrl` can also be replaced with `Cmd` instead for macOS users

## III. Manual Install on Windows

**NOTE**: Python 3.12 is used in this repository.

1. Git clone this repo.

```
git clone https://github.com/rzasharp79/4090_ComfyUI
```

2. Put your SD checkpoints (the huge ckpt/safetensors files) in: models/checkpoints

3. Put your VAE in: models/vae

4. Run STEP1-INSTALL.bat

5. Run STEP2-INSTALL_CUSTOM_NODES.bat
   
6. Run START_COMFY.bat

## IV. Notes

- Only parts of the graph that have an output with all the correct inputs will be executed.

- Only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. If you change the last part of the graph only the part you changed and the part that depends on it will be executed.

- Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.

- You can use () to change emphasis of a word or phrase like: (good code:1.2) or (bad code:0.8). The default emphasis for () is 1.1. To use () characters in your actual prompt escape them like \\( or \\).

- You can use {day|night}, for wildcard/dynamic prompts. With this syntax "{wild|card|test}" will be randomly replaced by either "wild", "card" or "test" by the frontend every time you queue the prompt. To use {} characters in your actual prompt escape them like: \\{ or \\}.

- Dynamic prompts also support C-style comments, like `// comment` or `/* comment */`.

- To use a textual inversion concepts/embeddings in a text prompt put them in the models/embeddings directory and use them in the CLIPTextEncode node like this (you can omit the .pt extension):

```embedding:embedding_filename.pt```

**How to show high-quality previews?**

- Use ```--preview-method auto``` to enable previews.

*The default installation includes a fast latent preview method that's low-resolution. To enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd), download the [taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth](https://github.com/madebyollin/taesd/) and place them in the `models/vae_approx` folder. Once they're installed, restart ComfyUI and launch it with `--preview-method taesd` to enable high-quality previews.*

**How to use TLS/SSL?**
- Generate a self-signed certificate (not appropriate for shared/production use) and key by running the command: `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

- Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL, the app will now be accessible with `https://...` instead of `http://...`.

> Note: Windows users can use [alexisrolland/docker-openssl](https://github.com/alexisrolland/docker-openssl) or one of the [3rd party binary distributions](https://wiki.openssl.org/index.php/Binaries) to run the command example above.
<br/><br/>If you use a container, note that the volume mount `-v` can be a relative path so `... -v ".\:/openssl-certs" ...` would create the key & cert files in the current directory of your command prompt or powershell terminal.

## V. Support and dev channel

[Discord](https://comfy.org/discord): Try the #help or #feedback channels.

[Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) (it's like discord but open source).

See also: [https://www.comfy.org/](https://www.comfy.org/)

## VI. Changelog

**Rev 0.1.1a (2025-08-23):** </br>
```
- Changed the names of batch files for running and installing
- Updated to latest Comfy version
```

**Rev 0.0.1a (2025-08-23):** </br>
- First commit to test out README markdown.

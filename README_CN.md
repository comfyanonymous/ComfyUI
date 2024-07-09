ComfyUI
=======
一个强大且模块化的稳定扩散 GUI 和后端。
-----------
Show Image

这个 UI 允许您使用基于图形/节点/流程图的界面设计和执行高级稳定扩散管道。要查看一些工作流示例并了解 ComfyUI 的功能,可以查看:
### [ComfyUI 示例](https://comfyanonymous.github.io/ComfyUI_examples/)

### [安装 ComfyUI](#installing)

## 特性
- 节点/图/流程图界面,可以在不需要编程的情况下试验和创建复杂的稳定扩散工作流。
- 完全支持 SD1.x、SD2.x 和 SDXL
- 异步队列系统
- 许多优化:在执行之间只重新执行工作流更改的部分。
- 命令行选项:```--lowvram```可在具有 3GB 或更少 VRAM 的 GPU 上运行(在具有低 VRAM 的 GPU 上自动启用)
- 即使没有 GPU 也可以工作:```--cpu```(慢)
- 可以加载 ckpt、safetensors 和 diffusers models/checkpoints。独立的 VAE 和 CLIP 模型。
- 嵌入(Embeddings)/文本反转(Textual inversion)
- [Loras(regular, locon and loha)](https://comfyanonymous.github.io/ComfyUI_examples/lora/)
- [Hypernetworks](https://comfyanonymous.github.io/ComfyUI_examples/hypernetworks/)
- 从生成的 PNG 文件加载完整工作流(带种子)。
- 将工作流保存/加载为 ```Json``` 文件。
- 节点界面可用于创建复杂的工作流,[Hires fix](https://comfyanonymous.github.io/ComfyUI_examples/2_pass_txt2img/) 修复或更高级的工作流。
- 区域组合[Area Composition](https://comfyanonymous.github.io/ComfyUI_examples/area_composition/)
- 修复 [Inpainting](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/)带常规和修复模型。
- ControlNet 和 T2I-Adapter[ControlNet and T2I-Adapter](https://comfyanonymous.github.io/ComfyUI_examples/controlnet/)
- 上缩放模型(ESRGAN、ESRGAN 变体、SwinIR、Swin2SR 等...)[Upscale Models (ESRGAN, ESRGAN variants, SwinIR, Swin2SR, etc...)](https://comfyanonymous.github.io/ComfyUI_examples/upscale_models/)
- unCLIP 模型[unCLIP Models](https://comfyanonymous.github.io/ComfyUI_examples/unclip/)
- GLIGEN[GLIGEN](https://comfyanonymous.github.io/ComfyUI_examples/gligen/)
- 模型合并 [Model Merging](https://comfyanonymous.github.io/ComfyUI_examples/model_merging/)
- 使用 TAESD 显示潜在预览[TAESD](#how-to-show-high-quality-previews)
- 启动非常快。
- 完全离线工作:绝不会下载任何内容。
- 配置文件 [Config file](extra_model_paths.yaml.example)用于设置模型的搜索路径。

可以在示例页面上找到工作流示例。[Examples page](https://comfyanonymous.github.io/ComfyUI_examples/)

## 快捷键
按键绑定	                    说明
Ctrl + Enter	            将当前图排入生成队列
Ctrl + Shift + Enter	    将当前图作为第一项排入生成队列
Ctrl + S	                保存工作流
Ctrl + O	                加载工作流
Ctrl + A	                选择所有节点
Ctrl + M	                静音/取消静音所选节点
Ctrl + B	                绕过所选节点(表现为从图中删除该节点并重新连接线)
Delete/Backspace	        删除所选节点
Ctrl + Delete/Backspace	    删除当前图
Spacecursor 空格	        按住时移动画布
Ctrl/Shift + 点击	        将单击的节点添加到选择中
Ctrl + C/Ctrl + V	        复制和粘贴所选节点(不保持与未选择节点输出的连接)
Ctrl + C/Ctrl + Shift + V	复制和粘贴所选节点(保持从未选择节点的输出到粘贴节点的输入的连接)
Shift + 拖动	            同时移动多个所选节点
Ctrl + D	                加载默认图
Q	                        切换队列可见性
H	                        切换历史记录可见性
R	                        刷新图
Double-Click LMB	        打开节点快速搜索面板
macOS                       用户可以用 Cmd 键代替 Ctrl

# 安装

## Windows

发布页面上有一个便携式独立 Windows 构建,可用于在 Nvidia GPU 或仅使用 CPU 运行。 [releases page](https://github.com/comfyanonymous/ComfyUI/releases).

### 直接下载链接[Direct link to download](https://github.com/comfyanonymous/ComfyUI/releases/download/latest/ComfyUI_windows_portable_nvidia_cu118_or_cpu.7z)
只需下载、使用 7-Zip 解压、然后运行。请确保将 Stable Diffusion 检查点/模型(大的ckpt/safetensors 文件)放在:ComfyUI\models\checkpoints

#### 我该如何在另一个 UI 和 ComfyUI 之间共享模型?
参见配置文件[Config file](extra_model_paths.yaml.example)以设置模型的搜索路径。在独立的 Windows 构建中,您可以在 ComfyUI 目录中找到此文件。将此文件重命名为 extra_model_paths.yaml,并使用您喜欢的文本编辑器对其进行编辑。

## Colab 笔记本
要在 colab 或 paperspace 上运行它,可以使用我的 [Colab Notebook](notebooks/comfyui_colab.ipynb) here: [Link to open with google colab](https://colab.research.google.com/github/comfyanonymous/ComfyUI/blob/master/notebooks/comfyui_colab.ipynb)在 Google Colab 中打开

## 手动安装(Windows、Linux)

克隆此存储库。

将 SD checkpoints (大的 ckpt/safetensors 文件)放在:models/checkpoints

将你的 VAE 放在:models/vae

### AMD GPU(仅限 Linux)

如果您还没有安装,AMD 用户可以安装 rocm 和使用 pip 安装 pytorch,这是安装稳定版本的命令:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.4.2```

这是使用支持 7000 系列并可能具有一些性能改进的 ROCm 5.6 nightly 的命令:
```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.6```

### NVIDIA
Nvidia 用户应使用此命令安装 torch 和 xformers:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 xformers```

#### 故障排除

如果您收到“Torch 未与 CUDA 一起编译”错误,请使用以下命令卸载 torch:

```pip uninstall torch```

然后再次使用上述命令重新安装它。

### 依赖项

通过在 ComfyUI 文件夹中打开终端并运行以下命令来安装依赖项:

```pip install -r requirements.txt```

之后您应该已经安装了所有内容并可以继续运行 ComfyUI。

### 其他:

#### Intel Arc[Intel Arc](https://github.com/comfyanonymous/ComfyUI/discussions/476)

#### Apple Mac silicon

您可以在任何最近的 macOS 版本的 Apple Mac silicon (M1 或 M2) 上安装 ComfyUI。

1. 安装 pytorch nightly。有关说明,请阅读 [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)指南(确保安装最新的 pytorch nightly)。
1. 按照 [ComfyUI manual installation](#manual-install-windows-linux) 的说明进行操作。
1. 安装 ComfyUI 依赖项 [dependencies](#dependencies).如果您已经安装了另一个 Stable Diffusion UI 您可能能够重用依赖项。
(#i-already-have-another-ui-for-stable-diffusion-installed-do-i-really-have-to-install-all-of-these-dependencies).
1. 通过运行 python main.py `python main.py --force-fp16`启动 ComfyUI。请注意,如果您安装了最新的 pytorch nightly, `python main.py --force-fp16`才能工作。

 > **注意**:请记住按照 ComfyUI 手动安装 中讨论的那样,将模型、VAE、LoRAs 等添加到相应的 Comfy 文件夹中[ComfyUI manual installation](#manual-install-windows-linux).

#### DirectML (Windows 上的 AMD 卡)

```pip install torch-directml``` 然后您可以用此启动 ComfyUI: ```python main.py --directml```

### 我已经安装了另一个稳定扩散 UI,我真的需要安装所有这些依赖项吗?

不需要。如果您已经安装了另一个 UI 及其自己的 python 虚拟环境,则可以使用该
虚拟环境来运行 ComfyUI。您可以打开喜欢的终端并激活它:

```source path_to_other_sd_gui/venv/bin/activate```
例如：我的虚拟环境是 ```B:\stable-diffusion-webui\venv\Scripts```
那么这条激活命令既是：```source B:\stable-diffusion-webui\venv\Scripts\activate```

或在 Windows 上:

使用 Powershell:```"path_to_other_sd_gui\venv\Scripts\Activate.ps1"```
例如：我的虚拟环境是  ```B:\stable-diffusion-webui\venv\Scripts```
在Pwershell里执行的命令为： ```B:\stable-diffusion-webui\venv\Scripts\Activate.ps1```

使用 cmd.exe:```"path_to_other_sd_gui\venv\Scripts\activate.bat"```
例如：我的虚拟环境是  ```B:\stable-diffusion-webui\venv\Scripts```
在 cmd.exe l里执行的命令为：```B:\stable-diffusion-webui\venv\Scripts\activate.bat```

然后您就可以使用该终端运行 ComfyUI,而无需安装任何依赖项。请注意,venv 文件夹的名称可能因 SD UI 的不同而有所不同。

# 运行

```python main.py```

### 对于 ROCm 不正式支持的 AMD 卡

如果您遇到问题,请尝试使用此命令运行它:

对于 6700、6600 和可能的其他：  ```HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py```

对于 AMD 7600 和可能的其他 RDNA3 卡: ```HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py```

# 注意事项

只有输出端具有所有正确输入的图形部分才会被执行。

只有在每次执行之间更改的图形部分才会被执行,如果两次提交相同的图形,只有第一次会被执行。如果您更改了图形的最后一部分,则只会执行您更改的部分以及依赖于它的部分。

在网页上拖放生成的 png 或加载一个将为您提供创建它所用的完整工作流程,包括种子。

您可以使用 () 来更改单词或短语的重点,例如:(good code:1.2) 或 (bad code:0.8)。() 的默认重点是 1.1。要在实际提示中使用 () 字符,请对其进行转义,如\\( 或 \\)。

您可以使用 {day|night} 进行通配符/动态提示。使用此语法“{wild|card|test}”将在每次您排队提示时由前端随机替换为“wild”、“card”或“test”。要在实际提示中使用 {} 字符,请对其进行转义,如:\\{ 或 \\}。

动态提示还支持 C-style 样式注释,如 `// comment` 或 `/* comment */`.

要在文本提示中使用文本反转概念/嵌入,请将它们放在 models/embeddings 目录中,并在 CLIPTextEncode 节点中使用它们,如下所示(可以省略 .pt 扩展名):

```embedding:embedding_filename.pt```

## 如何提高生成速度?

请确保使用常规加载器/加载检查点节点来加载检查点。它将根据您的 GPU 自动选择正确的设置。

您可以设置此命令行设置以禁用某些跨注意力操作中的 fp32 升级,这将提高速度。请注意,这很可能会在 SD2.x 模型上给您带来黑色图像。如果您使用 xformers,此选项不会起任何作用。

```--dont-upcast-attention```

## 如何显示高质量预览?

使用```--preview-method auto``` 来启用预览。

默认安装包括一个低分辨率的快速潜在预览方法。要启用具有 [TAESD](https://github.com/madebyollin/taesd) 的更高质量预览，请下载 [taesd_decoder.pth](https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth)（适用于 SD1.x 和 SD2.x）和 [taesdxl_decoder.pth](https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth)（适用于 SDXL）模型，并将它们放在  `models/vae_approx` 文件夹中。安装完成后，重新启动 ComfyUI 以启用高质量预览。

## 支持和开发频道

Matrix 聊天室: #comfyui_space:matrix.org(它类似 discord 但开源)。[Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) 

# QA常见问题解答

### 你为什么要做这个?

我想详细了解稳定扩散的工作原理。我也想要一些干净且强大的东西,它将允许我在不受限制的情况下使用 SD 进行实验。

### 这个是为谁准备的?
这适用于任何想用 SD 制作复杂工作流的人,或者任何想更深入了解 SD 如何工作的人。该界面紧密遵循 SD 的工作方式,与其他 SD UI 相比,代码应该更容易理解。
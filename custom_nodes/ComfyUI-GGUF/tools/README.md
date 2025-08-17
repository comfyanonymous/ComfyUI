## Converting initial model

To convert your initial safetensors/ckpt model to FP16/BF16 GGUF, run the following command:

```
python convert.py --src E:\models\unet\flux1-dev.safetensors
```
Make sure `gguf>=0.13.0` is installed for this step. Optionally, specify the output gguf file with the `--dst` arg.

> [!NOTE]  
> Do not use the diffusers UNET format for flux, it won't work, use the default/reference checkpoint key format. This is due to q/k/v being merged into one qkv key.
> You can convert it by loading it in ComfyUI and saving it using the built-in "ModelSave" node.

> [!WARNING] 
> For hunyuan video/wan 2.1, you will see a warning about 5D tensors. This means the script will save a **non functional** model to disk first, that you can quantize. I recommend saving these in a separate `raw` folder to avoid confusion.
> 
> After quantization, you will have to run `fix_5d_tensor.py` manually to add back the missing key that was saved by the conversion code.

## Quantizing using custom llama.cpp

Depending on your git settings, you may need to run the following script first in order to make sure the patch file is valid. It will convert Windows (CRLF) line endings to Unix (LF) ones.

```
python fix_lines_ending.py
```

Git clone llama.cpp into the current folder:

```
git clone https://github.com/ggerganov/llama.cpp
```

Check out the correct branch, then apply the custom patch needed to add image model support to the repo you just cloned.

```
cd llama.cpp
git checkout tags/b3962
git apply ..\lcpp.patch
```

Compile the llama-quantize binary. This example uses cmake, on linux you can just use make.

### Visual Studio 2019, Linux, etc...

```
mkdir build
cmake -B build
cmake --build build --config Debug -j10 --target llama-quantize
cd ..
```

### Visual Studio 2022

```
mkdir build
cmake -B build -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_FLAGS="-std=c++17"
```

Edit the `llama.cpp\common\log.cpp` file, inserts two lines after the existing first line:

```
#include "log.h"

#define _SILENCE_CXX23_CHRONO_DEPRECATION_WARNING
#include <chrono>
```

Then you can build the project:
```
cmake --build build --config Debug -j10 --target llama-quantize
cd ..
```

### Quantize your model


Now you can use the newly build binary to quantize your model to the desired format:
```
llama.cpp\build\bin\Debug\llama-quantize.exe E:\models\unet\flux1-dev-BF16.gguf E:\models\unet\flux1-dev-Q4_K_S.gguf Q4_K_S
```

You can extract the patch again with `git diff src\llama.cpp > lcpp.patch` if you wish to change something and contribute back.

> [!WARNING] 
> For hunyuan video/wan 2.1, you will have to run `fix_5d_tensor.py` after the quantization step is done.
>
> Example usage:  `fix_5d_tensors.py --src E:\models\video\raw\wan2.1-t2v-1.3b-Q8_0.gguf --dst E:\models\video\wan2.1-t2v-1.3b-Q8_0.gguf`
>
> By default, this also saves a `fix_5d_tensors_[arch].safetensors` file in the `ComfyUI-GGUF/tools` folder, it's recommended to delete this after all models have been converted.

> [!NOTE]
> Do not quantize SDXL / SD1 / other Conv2D heavy models. If you do, make sure to **extract the UNET model first**.
>This should be obvious, but also don't use the resulting llama-quantize binary with LLMs.

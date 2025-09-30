As of the time of writing this you need this preview driver for best results:
https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-PREVIEW.html

HOW TO RUN:

if you have a AMD gpu:

run_amd_gpu.bat


IF YOU GET A RED ERROR IN THE UI MAKE SURE YOU HAVE A MODEL/CHECKPOINT IN: ComfyUI\models\checkpoints

You can download the stable diffusion XL one from: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors


RECOMMENDED WAY TO UPDATE:
To update the ComfyUI code: update\update_comfyui.bat


TO SHARE MODELS BETWEEN COMFYUI AND ANOTHER UI:
In the ComfyUI directory you will find a file: extra_model_paths.yaml.example
Rename this file to: extra_model_paths.yaml and edit it with your favorite text editor.



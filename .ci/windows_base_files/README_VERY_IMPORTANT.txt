HOW TO RUN:

if you have a NVIDIA gpu:

run_nvidia_gpu.bat



To run it in slow CPU mode:

run_cpu.bat



IF YOU GET A RED ERROR IN THE UI MAKE SURE YOU HAVE A MODEL/CHECKPOINT IN: ccniyUI\models\checkpoints

You can download the stable diffusion 1.5 one from: https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt


RECOMMENDED WAY TO UPDATE:
To update the ccniyUI code: update\update_ccniyui.bat



To update ccniyUI with the python dependencies, note that you should ONLY run this if you have issues with python dependencies.
update\update_ccniyui_and_python_dependencies.bat


TO SHARE MODELS BETWEEN ccniyUI AND ANOTHER UI:
In the ccniyUI directory you will find a file: extra_model_paths.yaml.example
Rename this file to: extra_model_paths.yaml and edit it with your favorite text editor.

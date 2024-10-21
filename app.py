from beam import task_queue, Output, Image, env, Volume
import os
import sys
import requests

if env.is_remote():
    import folder_paths

    sys.path.append(os.getcwd())
    import main


# Function to setup environment and load models
def upload_models():
    # Define the model URLs and their respective directories
    model_urls = [
        # put models here
    ]

    # Download models if not already downloaded
    for model_name, filename, directory in model_urls:
        if not os.path.exists(f"{directory}/{filename}"):
            print(os.getcwd())
            os.system(
                f"wget -c https://huggingface.co/{model_name}/resolve/main/{filename} -P {os.path.join(os.getcwd() , directory)}"
            )


# Import functions or execute code from workflow_api.py
# Step 1: Define environment setup and model loading
@task_queue(
    name="ComfyUI",
    cpu=4,
    memory="32Gi",
    gpu="A10G",
    image=Image(
        python_version="python3.10",
        python_packages="requirements.txt",
        # commands=["pip3 install opencv-python"],
    ),
    volumes=[Volume(name="PUTVOLUMEHERE", mount_path="./VOLUMEPATH")],
    # keep_warm_seconds=0,
    # on_start=upload_models,
)
def generate_video(**inputs):
    main()

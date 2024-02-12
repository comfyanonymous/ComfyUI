# This sample shows how to execute a ComfyUI workflow against a remote ComfyUI server or the server running on your
# local machine. It will return the bytes of the image in the workflow.
#
# This script does not need to run within a ComfyUI directory. Instead, this can be used inside your own
# Python application or located elsewhere. It should **not** be in the Git repository directory.
#
# First, you will need to install ComfyUI. You do not need the ComfyUI repository or all of the ComfyUI dependencies to
# run a script against a server on your machine or elsewhere. You can install the convenient client and types with:
#   pip install --no-deps git+https://github.com/hiddenswitch/ComfyUI.git
#   pip install aiohttp[speedups]
#
# Now you should develop your workflow. Start ComfyUI as normal; navigate to "Settings" in the menu, and check "Enable
# Dev mode Options". Then click "Save (API Format)". Copy and paste the contents of this file here:
_PROMPT_FROM_WEB_UI = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 8,
            "denoise": 1,
            "latent_image": [
                "5",
                0
            ],
            "model": [
                "4",
                0
            ],
            "negative": [
                "7",
                0
            ],
            "positive": [
                "6",
                0
            ],
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": 8566257,
            "steps": 20
        }
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.ckpt"
        }
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": 512,
            "width": 512
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "4",
                1
            ],
            "text": "masterpiece best quality girl"
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "4",
                1
            ],
            "text": "bad hands"
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": [
                "3",
                0
            ],
            "vae": [
                "4",
                2
            ]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": [
                "8",
                0
            ]
        }
    }
}


# Observe this is an ordinary dictionary. The JSON that was saved from the workflow is compatible with Python syntax.
# 
# Because you will be accessing a remote ComfyUI server, it should remain running somewhere. It can be your local machine,
# or some other machine. Do what makes sense for your application.
# 
# We'll now write the entrypoint of our script. This is an `async def main()` because async helps us start and stop the
# code object that will run your workflow, just like pressing the Queue Prompt button.
async def main():
    import copy

    # Let's make some changes to the prompt. First we'll change the input text:
    prompt_dict = copy.deepcopy(_PROMPT_FROM_WEB_UI)
    prompt_dict["6"]["inputs"]["text"] = "masterpiece best quality man"

    # Let's set the seed for our KSampler node:
    prompt_dict["3"]["inputs"]["seed"] = 5

    # Now we will validate the prompt. This Prompt class contains everything we need to validate the prompt.
    from comfy.api.components.schema.prompt import Prompt
    prompt = Prompt.validate(prompt_dict)

    # Your prompt is ready to be processed. You should start your ComfyUI server; or, specify a remote URL for it.
    # If you want to run your own ComfyUI server, follow the **Manual Install (Windows, Linux, macOS)** instructions.
    # Then, as the README specifies, `cd` into the directory that contains your `models/` folder and run:
    #   comfyui --listen
    # Let's create the client we will use to access it:
    from comfy.client.aio_client import AsyncRemoteComfyClient
    client = AsyncRemoteComfyClient(server_address="http://localhost:8188")

    # Now let's get the bytes of the PNG image saved by the SaveImage node:
    png_image_bytes = await client.queue_prompt(prompt)

    # You can save these bytes wherever you need!
    with open("image.png", "rb") as f:
        f.write(png_image_bytes)


# Now let's make this script runnable:
import asyncio

if __name__ == "__main__":
    # Since our main function is async, it must be run as async too.
    asyncio.run(main())

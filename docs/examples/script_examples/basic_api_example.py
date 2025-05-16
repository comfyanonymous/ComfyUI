# This sample shows how to execute a ComfyUI workflow, saving an image file to the location you specify.
# It does not require the server to be run. It runs ComfyUI embedded, as a library. No process is started.
#
# This script does not need to run within a ComfyUI directory. Instead, this can be used inside your own
# Python application or located elsewhere. It should **not** be in the Git repository directory.
#
# First, you will need to install ComfyUI. Follow the **Manual Install (Windows, Linux, macOS)** instructions in the
# README.md. If you are an experienced developer, instead run `pip install git+https://github.com/hiddenswitch/ComfyUI.git`
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
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
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
# Now, QUIT AND CLOSE YOUR COMFYUI SERVER. You don't need it anymore. This script will handle starting and stopping
# the server for you. Actually, it will create an object that does the same thing that pressing Queue Prompt does.
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

    # Your prompt is ready to be processed.
    # You should **not** be running the ComfyUI application (the thing you start with /main.py). You don't need it. You
    # are not making any HTTP requests, you are not running a server, you are not connecting to anything, you are not
    # executing the main.py from the ComfyUI git repository, you don't even need that Git repository located anywhere.

    from comfy.cli_args_types import Configuration

    # Let's specify some settings. Suppose this is the structure of your directories:
    #   C:/Users/comfyanonymous/Documents/models
    #   C:/Users/comfyanonymous/Documents/models/checkpoints
    #   C:/Users/comfyanonymous/Documents/models/loras
    #   C:/Users/comfyanonymous/Documents/outputs
    # Then your "current working directory" (`cwd`) should be set to "C:/Users/comfyanonymous/Documents":
    #   configuration.cwd = "C:/Users/comfyanonymous/Documents/"
    # Or, if your models directory is located in the same directory as this script:
    #   configuration.cwd = os.path.dirname(__file__)
    configuration = Configuration()

    from comfy.client.embedded_comfy_client import Comfy
    async with Comfy(configuration=configuration) as client:
        # This will run your prompt
        outputs = await client.queue_prompt(prompt)

        # At this point, your prompt is finished and all the outputs, like saving images, have been completed.
        # Now the outputs will contain the same thing that the Web UI expresses: a file path for each output.
        # Let's find the node ID of the first SaveImage node. This will work when you change your workflow JSON from
        # the example above.
        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")

        # Now let's print the absolute path to the image.
        print(outputs[save_image_node_id]["images"][0]["abs_path"])
    # At this point, all the models have been unloaded from VRAM, and everything has been cleaned up.


# Now let's make this script runnable:
import asyncio

if __name__ == "__main__":
    # Since our main function is async, it must be run as async too.
    asyncio.run(main())

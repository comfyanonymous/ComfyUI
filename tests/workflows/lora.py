import pytest
import torch

from comfy import model_management
from comfy.api.components.schema.prompt import Prompt
from comfy.model_downloader import add_known_models, KNOWN_LORAS
from comfy.model_downloader_types import CivitFile
from comfy.model_management import CPUState

try:
    has_gpu = torch.device(torch.cuda.current_device()) is not None
except:
    has_gpu = False

model_management.cpu_state = CPUState.GPU if has_gpu else CPUState.CPU
from comfy.client.embedded_comfy_client import EmbeddedComfyClient


@pytest.mark.skipif(not has_gpu, reason="requires gpu for performant testing")
@pytest.mark.asyncio
async def test_lora_workflow():
    prompt = Prompt.validate({
        "3": {
            "inputs": {
                "seed": 851616030078638,
                "steps": 20,
                "cfg": 8,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
                "model": [
                    "10",
                    0
                ],
                "positive": [
                    "6",
                    0
                ],
                "negative": [
                    "7",
                    0
                ],
                "latent_image": [
                    "5",
                    0
                ]
            },
            "class_type": "KSampler",
            "_meta": {
                "title": "KSampler"
            }
        },
        "4": {
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": "Load Checkpoint"
            }
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {
                "title": "Empty Latent Image"
            }
        },
        "6": {
            "inputs": {
                "text": "masterpiece best quality girl",
                "clip": [
                    "10",
                    1
                ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Prompt)"
            }
        },
        "7": {
            "inputs": {
                "text": "bad hands",
                "clip": [
                    "10",
                    1
                ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Prompt)"
            }
        },
        "8": {
            "inputs": {
                "samples": [
                    "3",
                    0
                ],
                "vae": [
                    "4",
                    2
                ]
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "VAE Decode"
            }
        },
        "9": {
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": [
                    "8",
                    0
                ]
            },
            "class_type": "SaveImage",
            "_meta": {
                "title": "Save Image"
            }
        },
        "10": {
            "inputs": {
                "lora_name": "epi_noiseoffset2.safetensors",
                "strength_model": 1,
                "strength_clip": 1,
                "model": [
                    "4",
                    0
                ],
                "clip": [
                    "4",
                    1
                ]
            },
            "class_type": "LoraLoader",
            "_meta": {
                "title": "Load LoRA"
            }
        }
    })

    add_known_models("loras", KNOWN_LORAS, CivitFile(13941, 16576, "epi_noiseoffset2.safetensors"))
    async with EmbeddedComfyClient() as client:
        outputs = await client.queue_prompt(prompt)

        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
        assert outputs[save_image_node_id]["images"][0]["abs_path"] is not None

import pytest

from comfy.api.components.schema.prompt import Prompt
from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfy.model_downloader import add_known_models, KNOWN_LORAS
from comfy.model_downloader_types import CivitFile

_workflows = {
    "hunyuandit_1": {
        "3": {
            "inputs": {
                "seed": 377072733774956,
                "steps": 1,
                "cfg": 6,
                "sampler_name": "euler",
                "scheduler": "sgm_uniform",
                "denoise": 1,
                "model": [
                    "4",
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
                "ckpt_name": "hunyuan_dit_1.2.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": "Load Checkpoint"
            }
        },
        "5": {
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {
                "title": "Empty Latent Image"
            }
        },
        "6": {
            "inputs": {
                "text": "a bottle with a 彩虹星系 inside it on top of a wooden table on a snowy mountain top with the ocean and clouds in the background\n",
                "clip": [
                    "4",
                    1
                ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Positive Prompt)"
            }
        },
        "7": {
            "inputs": {
                "text": "",
                "clip": [
                    "4",
                    1
                ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Negative Prompt)"
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
        }
    },
    "audio_1": {
        "14": {
            "inputs": {
                "ckpt_name": "stable_audio_open_1.0.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": "Load Checkpoint"
            }
        },
        "15": {
            "inputs": {
                "clip_name": "t5_base.safetensors",
                "type": "stable_audio"
            },
            "class_type": "CLIPLoader",
            "_meta": {
                "title": "Load CLIP"
            }
        },
        "16": {
            "inputs": {
                "text": "hard bop, upright bass, slappy bass, low frequencies, drum kit brushed hi-hat, snare with ghost notes, syncopated, groove",
                "clip": [
                    "15",
                    0
                ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Prompt)"
            }
        },
        "17": {
            "inputs": {
                "text": "",
                "clip": [
                    "15",
                    0
                ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Prompt)"
            }
        },
        "19": {
            "inputs": {
                "cfg": 7,
                "model": [
                    "14",
                    0
                ],
                "positive": [
                    "16",
                    0
                ],
                "negative": [
                    "17",
                    0
                ]
            },
            "class_type": "CFGGuider",
            "_meta": {
                "title": "CFGGuider"
            }
        },
        "21": {
            "inputs": {
                "noise_seed": 600769511872395
            },
            "class_type": "RandomNoise",
            "_meta": {
                "title": "RandomNoise"
            }
        },
        "22": {
            "inputs": {
                "noise": [
                    "21",
                    0
                ],
                "guider": [
                    "19",
                    0
                ],
                "sampler": [
                    "29",
                    0
                ],
                "sigmas": [
                    "24",
                    0
                ],
                "latent_image": [
                    "30",
                    0
                ]
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {
                "title": "SamplerCustomAdvanced"
            }
        },
        "24": {
            "inputs": {
                "scheduler": "exponential",
                "steps": 20,
                "denoise": 1,
                "model": [
                    "14",
                    0
                ]
            },
            "class_type": "BasicScheduler",
            "_meta": {
                "title": "BasicScheduler"
            }
        },
        "26": {
            "inputs": {
                "samples": [
                    "22",
                    1
                ],
                "vae": [
                    "14",
                    2
                ]
            },
            "class_type": "VAEDecodeAudio",
            "_meta": {
                "title": "VAEDecodeAudio"
            }
        },
        "27": {
            "inputs": {
                "filename_prefix": "audio/objectobject",
                "audio": [
                    "26",
                    0
                ]
            },
            "class_type": "SaveAudio",
            "_meta": {
                "title": "SaveAudio"
            }
        },
        "29": {
            "inputs": {
                "version": "regular"
            },
            "class_type": "SamplerEulerCFGpp",
            "_meta": {
                "title": "SamplerEulerCFG++"
            }
        },
        "30": {
            "inputs": {
                "seconds": 47.6
            },
            "class_type": "EmptyLatentAudio",
            "_meta": {
                "title": "EmptyLatentAudio"
            }
        },
        "31": {
            "inputs": {
                "filename_prefix": "latents/ComfyUI",
                "samples": [
                    "22",
                    1
                ]
            },
            "class_type": "SaveLatent",
            "_meta": {
                "title": "SaveLatent"
            }
        }
    },
    "auraflow_1": {
        "1": {
            "inputs": {
                "ckpt_name": "aura_flow_0.1.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": "Load Checkpoint"
            }
        },
        "2": {
            "inputs": {
                "shift": 1.73,
                "model": [
                    "1",
                    0
                ]
            },
            "class_type": "ModelSamplingAuraFlow",
            "_meta": {
                "title": "ModelSamplingAuraFlow"
            }
        },
        "3": {
            "inputs": {
                "seed": 232240565010917,
                "steps": 25,
                "cfg": 3.5,
                "sampler_name": "uni_pc",
                "scheduler": "normal",
                "denoise": 1,
                "model": [
                    "2",
                    0
                ],
                "positive": [
                    "4",
                    0
                ],
                "negative": [
                    "5",
                    0
                ],
                "latent_image": [
                    "6",
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
                "text": "close-up portrait of cat",
                "clip": [
                    "1",
                    1
                ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Prompt)"
            }
        },
        "5": {
            "inputs": {
                "text": "",
                "clip": [
                    "1",
                    1
                ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Prompt)"
            }
        },
        "6": {
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {
                "title": "Empty Latent Image"
            }
        },
        "7": {
            "inputs": {
                "samples": [
                    "3",
                    0
                ],
                "vae": [
                    "1",
                    2
                ]
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "VAE Decode"
            }
        },
        "8": {
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": [
                    "7",
                    0
                ]
            },
            "class_type": "SaveImage",
            "_meta": {
                "title": "Save Image"
            }
        }
    },
    "lora_1": {
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
    }
}


@pytest.fixture(scope="module", autouse=False)
@pytest.mark.asyncio
async def client(tmp_path_factory) -> EmbeddedComfyClient:
    async with EmbeddedComfyClient() as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_name, workflow", _workflows.items())
async def test_workflow(workflow_name: str, workflow: dict, has_gpu: bool, client: EmbeddedComfyClient):
    if not has_gpu:
        pytest.skip("requires gpu")

    if "audio" in workflow_name:
        try:
            import torchaudio
        except (ImportError, ModuleNotFoundError):
            pytest.skip("requires torchaudio")

    prompt = Prompt.validate(workflow)
    add_known_models("loras", KNOWN_LORAS, CivitFile(13941, 16576, "epi_noiseoffset2.safetensors"))
    # todo: add all the models we want to test a bit more elegantly
    outputs = await client.queue_prompt(prompt)

    if any(v.class_type == "SaveImage" for v in prompt.values()):
        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
        assert outputs[save_image_node_id]["images"][0]["abs_path"] is not None
    elif any(v.class_type == "SaveAudio" for v in prompt.values()):
        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveAudio")
        assert outputs[save_image_node_id]["audio"][0]["filename"] is not None

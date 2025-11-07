import copy
from typing import TypeAlias, Union

JSON: TypeAlias = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]
_BASE_PROMPT: JSON = {
    "4": {
        "inputs": {
            "ckpt_name": "sd_xl_base_1.0.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    "5": {
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage"
    },
    "6": {
        "inputs": {
            "text": "a photo of a cat",
            "clip": [
                "4",
                1
            ]
        },
        "class_type": "CLIPTextEncode"
    },
    "10": {
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 42,
            "steps": 20,
            "cfg": 7.5,
            "sampler_name": "euler",
            "scheduler": "normal",
            "start_at_step": 0,
            "end_at_step": 32,
            "return_with_leftover_noise": "enable",
            "model": [
                "4",
                0
            ],
            "positive": [
                "6",
                0
            ],
            "negative": [
                "15",
                0
            ],
            "latent_image": [
                "5",
                0
            ]
        },
        "class_type": "KSamplerAdvanced"
    },
    "12": {
        "inputs": {
            "samples": [
                "14",
                0
            ],
            "vae": [
                "4",
                2
            ]
        },
        "class_type": "VAEDecode"
    },
    "13": {
        "inputs": {
            "filename_prefix": "test_inference",
            "images": [
                "12",
                0
            ]
        },
        "class_type": "SaveImage"
    },
    "14": {
        "inputs": {
            "add_noise": "disable",
            "noise_seed": 42,
            "steps": 20,
            "cfg": 7.5,
            "sampler_name": "euler",
            "scheduler": "normal",
            "start_at_step": 32,
            "end_at_step": 10000,
            "return_with_leftover_noise": "disable",
            "model": [
                "16",
                0
            ],
            "positive": [
                "17",
                0
            ],
            "negative": [
                "20",
                0
            ],
            "latent_image": [
                "10",
                0
            ]
        },
        "class_type": "KSamplerAdvanced"
    },
    "15": {
        "inputs": {
            "conditioning": [
                "6",
                0
            ]
        },
        "class_type": "ConditioningZeroOut"
    },
    "16": {
        "inputs": {
            "ckpt_name": "sd_xl_refiner_1.0.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    "17": {
        "inputs": {
            "text": "a photo of a cat",
            "clip": [
                "16",
                1
            ]
        },
        "class_type": "CLIPTextEncode"
    },
    "20": {
        "inputs": {
            "text": "",
            "clip": [
                "16",
                1
            ]
        },
        "class_type": "CLIPTextEncode"
    }
}


def sdxl_workflow_with_refiner(prompt: str,
                               negative_prompt: str = "",
                               inference_steps=25,
                               refiner_steps=5,
                               sdxl_base_checkpoint_name="sd_xl_base_1.0.safetensors",
                               sdxl_refiner_checkpoint_name="sd_xl_refiner_1.0.safetensors",
                               width=1024,
                               height=1024,
                               sampler="euler_ancestral",
                               scheduler="normal",
                               filename_prefix="sdxl_",
                               seed=42) -> dict:
    prompt_dict: JSON = copy.deepcopy(_BASE_PROMPT)
    prompt_dict["17"]["inputs"]["text"] = prompt
    prompt_dict["20"]["inputs"]["text"] = negative_prompt
    prompt_dict["16"]["inputs"]["ckpt_name"] = sdxl_refiner_checkpoint_name
    prompt_dict["4"]["inputs"]["ckpt_name"] = sdxl_base_checkpoint_name
    prompt_dict["5"]["inputs"]["width"] = width
    prompt_dict["5"]["inputs"]["height"] = height

    # base
    prompt_dict["10"]["inputs"]["steps"] = inference_steps + refiner_steps
    prompt_dict["10"]["inputs"]["seed"] = seed
    prompt_dict["10"]["inputs"]["start_at_step"] = 0
    prompt_dict["10"]["inputs"]["end_at_step"] = inference_steps
    prompt_dict["10"]["inputs"]["steps"] = inference_steps + refiner_steps
    prompt_dict["10"]["inputs"]["sampler_name"] = sampler
    prompt_dict["10"]["inputs"]["scheduler"] = scheduler

    # refiner
    prompt_dict["14"]["inputs"]["steps"] = inference_steps + refiner_steps
    prompt_dict["10"]["inputs"]["seed"] = seed
    prompt_dict["14"]["inputs"]["start_at_step"] = inference_steps
    prompt_dict["14"]["inputs"]["end_at_step"] = inference_steps + refiner_steps
    prompt_dict["14"]["inputs"]["sampler_name"] = sampler
    prompt_dict["14"]["inputs"]["scheduler"] = scheduler

    prompt_dict["13"]["inputs"]["filename_prefix"] = filename_prefix
    return prompt_dict

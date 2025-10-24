from contextvars import ContextVar
from typing import Final

import pytest
from pytest_mock import MockerFixture

from comfy.cli_args import args
from comfy.cmd.execution import validate_prompt
from comfy.nodes_context import get_nodes

import uuid

valid_prompt: Final[dict] = {
    "1": {
        "inputs": {
            "ckpt_name": "model1.safetensors",
        },
        "class_type": "CheckpointLoaderSimple",
    },
    "2": {
        "inputs": {
            "text": "a beautiful landscape",
            "clip": ["1", 1],
        },
        "class_type": "CLIPTextEncode",
    },
    "3": {
        "inputs": {
            "text": "ugly, deformed",
            "clip": ["1", 1],
        },
        "class_type": "CLIPTextEncode",
    },
    "4": {
        "inputs": {
            "width": 512,
            "height": 512,
            "batch_size": 1,
        },
        "class_type": "EmptyLatentImage",
    },
    "5": {
        "inputs": {
            "model": ["1", 0],
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "denoise": 1.0,
        },
        "class_type": "KSampler",
    },
    "6": {
        "inputs": {
            "samples": ["5", 0],
            "vae": ["1", 2],
        },
        "class_type": "VAEDecode",
    },
    "7": {
        "inputs": {
            "images": ["6", 0],
            "filename_prefix": "test_output",
        },
        "class_type": "SaveImage",
    },
}

known_models: ContextVar[list[str]] = ContextVar('known_models', default=[])


@pytest.fixture
def mock_nodes(mocker: MockerFixture):
    nodes = get_nodes()
    class MockCheckpointLoaderSimple:
        @staticmethod
        def INPUT_TYPES():
            models = known_models.get()
            return {
                "required": {
                    "ckpt_name": (models if models else ["model1.safetensors", "model2.safetensors"],),
                }
            }

        RETURN_TYPES = ("MODEL", "CLIP", "VAE")

    mocker.patch.dict(nodes.NODE_CLASS_MAPPINGS, {
        "CheckpointLoaderSimple": MockCheckpointLoaderSimple,
        "KSampler": type("KSampler", (), {
            "INPUT_TYPES": staticmethod(lambda: {
                "required": {
                    "model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"],),
                    "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"],),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "latent_image": ("LATENT",),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                }
            }),
            "RETURN_TYPES": ("LATENT",),
        }),
        "CLIPTextEncode": type("CLIPTextEncode", (), {
            "INPUT_TYPES": staticmethod(lambda: {
                "required": {
                    "text": ("STRING", {"multiline": True}),
                    "clip": ("CLIP",),
                }
            }),
            "RETURN_TYPES": ("CONDITIONING",),
        }),
        "VAEDecode": type("VAEDecode", (), {
            "INPUT_TYPES": staticmethod(lambda: {
                "required": {
                    "samples": ("LATENT",),
                    "vae": ("VAE",),
                }
            }),
            "RETURN_TYPES": ("IMAGE",),
        }),
        "SaveImage": type("SaveImage", (), {
            "INPUT_TYPES": staticmethod(lambda: {
                "required": {
                    "images": ("IMAGE",),
                    "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                }
            }),
            "RETURN_TYPES": (),
            "OUTPUT_NODE": True,
        }),
        "EmptyLatentImage": type("EmptyLatentImage", (), {
            "INPUT_TYPES": staticmethod(lambda: {
                "required": {
                    "width": ("INT", {"default": 512, "min": 16, "max": 8192}),
                    "height": ("INT", {"default": 512, "min": 16, "max": 8192}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                }
            }),
            "RETURN_TYPES": ("LATENT",),
        }),
    })


@pytest.fixture
def disable_known_models():
    original_value = args.disable_known_models
    args.disable_known_models = False
    yield
    args.disable_known_models = original_value


async def test_validate_prompt_valid(mock_nodes):
    prompt = valid_prompt
    result = await validate_prompt(str(uuid.uuid4()), prompt)
    assert result.valid
    assert result.error is None
    assert set(result.good_output_node_ids) == {"7"}


async def test_validate_prompt_invalid_node(mock_nodes):
    prompt = {
        "1": {
            "inputs": {},
            "class_type": "NonExistentNode",
        },
    }

    result = await validate_prompt(str(uuid.uuid4()), prompt)
    assert not result.valid
    assert result.error["type"] == "invalid_prompt"
    assert "NonExistentNode" in result.error["message"]


async def test_prompt_has_no_output(mock_nodes):
    prompt = {
        "1": {
            "inputs": {},
            "class_type": "CheckpointLoaderSimple",
        },
    }

    result = await validate_prompt(str(uuid.uuid4()), prompt)
    assert not result.valid
    assert result.error["type"] == "prompt_no_outputs"


async def test_validate_prompt_invalid_input_type(mock_nodes):
    prompt = valid_prompt.copy()
    prompt["1"] = {
        "inputs": {
            "ckpt_name": 123,
        },
        "class_type": "CheckpointLoaderSimple",
    }

    result = await validate_prompt(str(uuid.uuid4()), prompt)
    assert not result.valid
    assert result.error["type"] == "prompt_outputs_failed_validation"
    assert result.node_errors["1"]["errors"][0]["type"] == "value_not_in_list"


@pytest.mark.parametrize("ckpt_name, known_model", [
    ("model\\with\\backslash.safetensors", "model/with/backslash.safetensors"),
    ("model/with/forward/slash.safetensors", "model/with/forward/slash.safetensors"),
    ("mixed\\slash/path.safetensors", "mixed/slash/path.safetensors"),
    ("model with spaces.safetensors", "model with spaces.safetensors"),
    ("model_with_underscores.safetensors", "model_with_underscores.safetensors"),
    ("C:\\Windows\\Temp\\model.safetensors", "C:/Windows/Temp/model.safetensors"),
    ("/home/user/models/model.safetensors", "/home/user/models/model.safetensors"),
])
async def test_validate_prompt_path_variations(mock_nodes, disable_known_models, ckpt_name, known_model):
    token = known_models.set([known_model])

    try:
        prompt = valid_prompt.copy()
        prompt["1"] = {
            "inputs": {
                "ckpt_name": ckpt_name,
            },
            "class_type": "CheckpointLoaderSimple",
        }

        result = await validate_prompt(str(uuid.uuid4()), prompt)
        assert result.valid, f"Failed for ckpt_name: {ckpt_name}, known_model: {known_model}"
        assert result.error is None, f"Error for ckpt_name: {ckpt_name}, known_model: {known_model}"
    finally:
        known_models.reset(token)


async def test_validate_prompt_default_models(mock_nodes, disable_known_models):
    prompt = valid_prompt.copy()
    prompt["1"]["inputs"]["ckpt_name"] = "model1.safetensors"

    result = await validate_prompt(str(uuid.uuid4()), prompt)
    assert result.valid, "Failed for default model list"
    assert result.error is None, "Error for default model list"


async def test_validate_prompt_no_outputs(mock_nodes):
    prompt = {
        "1": {
            "inputs": {
                "ckpt_name": "model1.safetensors",
            },
            "class_type": "CheckpointLoaderSimple",
        },
    }

    result = await validate_prompt(str(uuid.uuid4()), prompt)
    assert not result.valid
    assert result.error["type"] == "prompt_no_outputs"

import pytest
import torch

from comfy import model_management
from comfy.model_base import Flux
from comfy.model_patcher import ModelPatcher
from comfy.nodes.base_nodes import UNETLoader
from comfy_extras.nodes.nodes_torch_compile import QuantizeModel

has_torchao = True
try:
    from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
except (ImportError, ModuleNotFoundError):
    has_torchao = False

has_tensorrt = True
try:
    from comfyui_tensorrt import STATIC_TRT_MODEL_CONVERSION
except (ImportError, ModuleNotFoundError):
    has_tensorrt = False


@pytest.mark.parametrize("checkpoint_name", ["flux1-dev.safetensors"])
@pytest.mark.skipif(not has_torchao, reason="torchao not installed")
async def test_unit_torchao(checkpoint_name):
    # Downloads FLUX.1-dev and loads it using ComfyUI's models
    model, = UNETLoader().load_unet(checkpoint_name, weight_dtype="default")
    model: ModelPatcher = model.clone()

    transformer: Flux = model.get_model_object("diffusion_model")
    quantize_(transformer, int8_dynamic_activation_int8_weight(), device=model_management.get_torch_device())
    assert transformer is not None
    del transformer
    model_management.unload_all_models()


@pytest.mark.parametrize("checkpoint_name", ["flux1-dev.safetensors"])
@pytest.mark.parametrize("strategy", ["torchao", "torchao-autoquant"])
@pytest.mark.skipif(not has_torchao, reason="torchao not installed")
async def test_torchao_node(checkpoint_name, strategy):
    model, = UNETLoader().load_unet(checkpoint_name, weight_dtype="default")
    model: ModelPatcher = model.clone()

    quantized_model, = QuantizeModel().execute(model, strategy=strategy)

    transformer = quantized_model.get_model_object("diffusion_model")
    del transformer
    model_management.unload_all_models()


@pytest.mark.parametrize("checkpoint_name", ["flux1-dev.safetensors"])
@pytest.mark.parametrize("strategy", ["torchao", "torchao-autoquant"])
@pytest.mark.skipif(True, reason="not yet supported")
async def test_torchao_into_tensorrt(checkpoint_name, strategy):
    model, = UNETLoader().load_unet(checkpoint_name, weight_dtype="default")
    model: ModelPatcher = model.clone()
    model_management.load_models_gpu([model], force_full_load=True)
    model.diffusion_model = model.diffusion_model.to(memory_format=torch.channels_last)
    model.diffusion_model = torch.compile(model.diffusion_model, mode="max-autotune", fullgraph=True)

    quantized_model, = QuantizeModel().execute(model, strategy=strategy)

    STATIC_TRT_MODEL_CONVERSION().convert(quantized_model, "test", 1, 1024, 1024, 1, 14)
    model_management.unload_all_models()

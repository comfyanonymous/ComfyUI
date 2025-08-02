import pytest

from comfy import model_management
from comfy.model_patcher import ModelPatcher
from comfy.nodes.base_nodes import UNETLoader, CheckpointLoaderSimple
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


@pytest.fixture(scope="function", params=["flux1-dev.safetensors"])
def model_patcher_obj(request) -> ModelPatcher:
    checkpoint_name = request.param
    model_obj = None
    try:
        if "flux" in checkpoint_name:
            model_obj, = UNETLoader().load_unet(checkpoint_name, weight_dtype="default")
            yield model_obj
        else:
            objs = CheckpointLoaderSimple().load_checkpoint(checkpoint_name)
            model_obj = objs[0]
            yield model_obj
    finally:
        model_management.unload_all_models()
        if model_obj is not None:
            model_obj.unpatch_model()
            del model_obj

        model_management.soft_empty_cache(force=True)


# @pytest.mark.forked
@pytest.mark.skipif(not has_torchao, reason="torchao not installed")
@pytest.mark.skipif(True, reason="wip")
async def test_unit_torchao(model_patcher_obj):
    quantize_(model_patcher_obj.diffusion_model, int8_dynamic_activation_int8_weight(), device=model_management.get_torch_device())


# @pytest.mark.forked
@pytest.mark.parametrize("strategy", ["torchao", "torchao-autoquant"])
@pytest.mark.skipif(True, reason="wip")
async def test_torchao_node(model_patcher_obj, strategy):
    QuantizeModel().execute(model_patcher_obj, strategy=strategy)


# @pytest.mark.forked
@pytest.mark.skipif(True, reason="wip")
async def test_tensorrt(model_patcher_obj):
    STATIC_TRT_MODEL_CONVERSION().convert(model_patcher_obj, "test", 1, 1024, 1024, 1, 14)

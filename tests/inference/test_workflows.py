import importlib.resources
import json
import logging
from importlib.abc import Traversable

import pytest

from comfy.api.components.schema.prompt import Prompt
from comfy.client.embedded_comfy_client import Comfy
from comfy.model_downloader import add_known_models, KNOWN_LORAS
from comfy.model_downloader_types import CivitFile, HuggingFile
from comfy_extras.nodes.nodes_audio import TorchAudioNotFoundError
from . import workflows

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function", autouse=False)
async def client(tmp_path_factory) -> Comfy:
    async with Comfy() as client:
        yield client


def _prepare_for_workflows() -> dict[str, Traversable]:
    add_known_models("loras", KNOWN_LORAS, CivitFile(13941, 16576, "epi_noiseoffset2.safetensors"))
    add_known_models("checkpoints", HuggingFile("autismanon/modeldump", "cardosAnime_v20.safetensors"))

    return {f.name: f for f in importlib.resources.files(workflows).iterdir() if f.is_file() and f.name.endswith(".json")}


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_name, workflow_file", _prepare_for_workflows().items())
async def test_workflow(workflow_name: str, workflow_file: Traversable, has_gpu: bool, client: Comfy):
    if not has_gpu:
        pytest.skip("requires gpu")

    workflow = json.loads(workflow_file.read_text(encoding="utf8"))

    prompt = Prompt.validate(workflow)
    # todo: add all the models we want to test a bit m2ore elegantly
    outputs = {}
    try:
        outputs = await client.queue_prompt(prompt)
    except TorchAudioNotFoundError:
        pytest.skip("requires torchaudio")

    if any(v.class_type == "SaveImage" for v in prompt.values()):
        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
        assert outputs[save_image_node_id]["images"][0]["abs_path"] is not None
    elif any(v.class_type == "SaveAudio" for v in prompt.values()):
        save_audio_node_id = next(key for key in prompt if prompt[key].class_type == "SaveAudio")
        assert outputs[save_audio_node_id]["audio"][0]["filename"] is not None
    elif any(v.class_type == "SaveAnimatedWEBP" for v in prompt.values()):
        save_video_node_id = next(key for key in prompt if prompt[key].class_type == "SaveAnimatedWEBP")
        assert outputs[save_video_node_id]["images"][0]["filename"] is not None
    elif any(v.class_type == "PreviewString" for v in prompt.values()):
        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "PreviewString")
        output_str = outputs[save_image_node_id]["string"][0]
        assert output_str is not None
        assert len(output_str) > 0
    else:
        assert len(outputs) > 0
        logger.warning(f"test {workflow_name} did not have a node that could be checked for output")

import os.path
import tempfile
from importlib.resources import files

import pytest

from comfy.api.components.schema.prompt import Prompt
from comfy.cli_args import default_configuration
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import Comfy

_TEST_WORKFLOW = {
    "0": {
        "inputs": {},
        "class_type": "TestPath",
        "_meta": {
            "title": ""
        }
    }
}


@pytest.mark.asyncio
async def test_respect_cwd_param():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = str(tmp_dir)
        config = default_configuration()
        config.cwd = cwd

        from comfy.cmd.folder_paths import models_dir
        assert os.path.commonpath([os.getcwd(), models_dir]) == os.getcwd(), "at the time models_dir is accessed, the cwd should be the actual cwd, since there is no other configuration"

        # for finding the custom nodes
        config.base_paths = [str(files(__package__))]

        async with Comfy(config) as client:
            prompt = Prompt.validate(_TEST_WORKFLOW)
            outputs = await client.queue_prompt_api(prompt)
            path_as_imported = outputs.outputs["0"]["path"][0]
            assert os.path.commonpath([path_as_imported, cwd]) == cwd, "at the time the node is imported, the cwd should be the temporary directory"

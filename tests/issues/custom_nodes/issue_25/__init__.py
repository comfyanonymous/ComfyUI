import os

from comfy.cmd import folder_paths
from comfy.nodes.package_typing import CustomNode, InputTypes, FunctionReturnsUIVariables

TEST_PATH = os.path.join(folder_paths.models_dir, "test", "path")


class TestPath(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {"required": {}}

    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self) -> FunctionReturnsUIVariables:
        return {"ui": {"path": [TEST_PATH]}}


NODE_CLASS_MAPPINGS = {
    "TestPath": TestPath
}

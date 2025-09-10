from comfy.node_helpers import export_custom_nodes
from comfy.nodes.package_typing import CustomNode, InputTypes


class OutputTensor(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "tensor": ("IMAGE,AUDIO,VIDEO", {})
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"

    def execute(self, tensor):
        return {"ui": {"tensor": tensor}}


export_custom_nodes()

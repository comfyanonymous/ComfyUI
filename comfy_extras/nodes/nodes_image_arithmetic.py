from torch import Tensor

from comfy.nodes.package_typing import CustomNode, InputTypes, ValidatedNodeResult


class ImageMin(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "image": ("IMAGE", {})
            }
        }

    RETURN_TYPES = ("FLOAT",)
    CATEGORY = "image/postprocessing"
    FUNCTION = "execute"

    def execute(self, image: Tensor) -> ValidatedNodeResult:
        return float(image.min().item()),


class ImageMax(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "image": ("IMAGE", {})
            }
        }

    RETURN_TYPES = ("FLOAT",)
    CATEGORY = "image/postprocessing"
    FUNCTION = "execute"

    def execute(self, image: Tensor) -> ValidatedNodeResult:
        return float(image.max().item()),


NODE_CLASS_MAPPINGS = {
    ImageMin.__name__: ImageMin,
    ImageMax.__name__: ImageMax,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    ImageMin.__name__: "Image Minimum Value",
    ImageMax.__name__: "Image Maximum Value"
}

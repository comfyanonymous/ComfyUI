from comfy.nodes.package_typing import CustomNode, InputTypes, FunctionReturnsUIVariables


class ShouldNotExist(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {"required": {}}

    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self) -> tuple[...,]:
        return None,


NODE_CLASS_MAPPINGS = {
    "ShouldNotExist": ShouldNotExist
}

class _CONSTANT_BASE:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s) -> dict:
        raise NotImplementedError

    RETURN_TYPES: tuple = ()
    RETURN_NAMES: tuple = ()

    CATEGORY: str = "constants"

    FUNCTION: str = "process"

    OUTPUT_NODE: bool = False

    def process(self, value) -> tuple:
        return (value,)


class ConstantFloat(_CONSTANT_BASE):
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0})
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)


class ConstantInteger(_CONSTANT_BASE):
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "value": ("INT", {"default": 0})
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)


class ConstantString(_CONSTANT_BASE):
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required" : {
                "value": ("STRING", {"multiline": False, "dynamicPrompts": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)


class ConstantStringMultiline(_CONSTANT_BASE):
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "value": ("STRING", {"multiline": True, "dynamicPrompts": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)


class ConstantNonDynamicString(_CONSTANT_BASE):
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required" : {
                "value": ("STRING", {"multiline": False, "dynamicPrompts": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)


class ConstantNonDynamicStringMultiline(_CONSTANT_BASE):
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "value": ("STRING", {"multiline": True, "dynamicPrompts": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)


NODE_CLASS_MAPPINGS = {
    "ConstantFloat": ConstantFloat,
    "ConstantInteger": ConstantInteger,
    "ConstantString": ConstantString,
    "ConstantStringMultiline": ConstantStringMultiline,
    "ConstantNonDynamicString": ConstantNonDynamicString,
    "ConstantNonDynamicStringMultiline": ConstantNonDynamicStringMultiline
}

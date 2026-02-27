from __future__ import annotations

import time


class LongComboDropdown:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"option": ([f"Option {i}" for i in range(1_000)],)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "long_combo_dropdown"
    CATEGORY = "DevTools"
    DESCRIPTION = "A long combo dropdown"

    def long_combo_dropdown(self, option: str):
        print(option)


class NodeWithOptionalInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"required_input": ("IMAGE",)},
            "optional": {"optional_input": ("IMAGE", {"default": None})},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "node_with_optional_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with an optional input"

    def node_with_optional_input(self, required_input, optional_input=None):
        print(
            f"Calling node with required_input: {required_input} and optional_input: {optional_input}"
        )
        return (required_input,)


class NodeWithOptionalComboInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "optional_combo_input": (
                    [f"Random Unique Option {time.time()}" for _ in range(8)],
                    {"default": None},
                )
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "node_with_optional_combo_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with an optional combo input that returns unique values every time INPUT_TYPES is called"

    def node_with_optional_combo_input(self, optional_combo_input=None):
        print(f"Calling node with optional_combo_input: {optional_combo_input}")
        return (optional_combo_input,)


class NodeWithOnlyOptionalInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "clip": ("CLIP", {}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "node_with_only_optional_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with only optional input"

    def node_with_only_optional_input(self, clip=None, text=None):
        pass


class NodeWithOutputList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = (
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "INTEGER OUTPUT",
        "INTEGER LIST OUTPUT",
    )
    OUTPUT_IS_LIST = (
        False,
        True,
    )
    FUNCTION = "node_with_output_list"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with an output list"

    def node_with_output_list(self):
        return (1, [1, 2, 3])


class NodeWithForceInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int_input": ("INT", {"forceInput": True}),
                "int_input_widget": ("INT", {"default": 1}),
            },
            "optional": {"float_input": ("FLOAT", {"forceInput": True})},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "node_with_force_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with a forced input"

    def node_with_force_input(
        self, int_input: int, int_input_widget: int, float_input: float = 0.0
    ):
        print(
            f"int_input: {int_input}, int_input_widget: {int_input_widget}, float_input: {float_input}"
        )


class NodeWithDefaultInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int_input": ("INT", {"defaultInput": True}),
                "int_input_widget": ("INT", {"default": 1}),
            },
            "optional": {"float_input": ("FLOAT", {"defaultInput": True})},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "node_with_default_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with a default input"

    def node_with_default_input(
        self, int_input: int, int_input_widget: int, float_input: float = 0.0
    ):
        print(
            f"int_input: {int_input}, int_input_widget: {int_input_widget}, float_input: {float_input}"
        )


class NodeWithStringInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string_input": ("STRING",)}}

    RETURN_TYPES = ()
    FUNCTION = "node_with_string_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with a string input"

    def node_with_string_input(self, string_input: str):
        print(f"string_input: {string_input}")


class NodeWithUnionInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "string_or_int_input": ("STRING,INT",),
                "string_input": ("STRING", {"forceInput": True}),
                "int_input": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "node_with_union_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with a union input"

    def node_with_union_input(
        self,
        string_or_int_input: str | int = "",
        string_input: str = "",
        int_input: int = 0,
    ):
        print(
            f"string_or_int_input: {string_or_int_input}, string_input: {string_input}, int_input: {int_input}"
        )
        return {
            "ui": {
                "text": string_or_int_input,
            }
        }


class NodeWithBooleanInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"boolean_input": ("BOOLEAN",)}}

    RETURN_TYPES = ()
    FUNCTION = "node_with_boolean_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with a boolean input"

    def node_with_boolean_input(self, boolean_input: bool):
        print(f"boolean_input: {boolean_input}")


class SimpleSlider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "FLOAT",
                    {
                        "display": "slider",
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "DevTools"

    def execute(self, value):
        return (value,)


class NodeWithSeedInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0})}}

    RETURN_TYPES = ()
    FUNCTION = "node_with_seed_input"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node with a seed input"
    OUTPUT_NODE = True

    def node_with_seed_input(self, seed: int):
        print(f"seed: {seed}")


class NodeWithValidation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"int_input": ("INT",)},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, int_input: int):
        if int_input < 0:
            raise ValueError("int_input must be greater than 0")
        return True

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "DevTools"
    DESCRIPTION = "A node that validates an input"
    OUTPUT_NODE = True

    def execute(self, int_input: int):
        print(f"int_input: {int_input}")
        return tuple()


class NodeWithV2ComboInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "combo_input": (
                    "COMBO",
                    {"options": ["A", "B"]},
                ),
            }
        }

    RETURN_TYPES = ("COMBO",)
    FUNCTION = "node_with_v2_combo_input"
    CATEGORY = "DevTools"
    DESCRIPTION = (
        "A node that outputs a combo type that adheres to the v2 combo input spec"
    )

    def node_with_v2_combo_input(self, combo_input: str):
        return (combo_input,)


NODE_CLASS_MAPPINGS = {
    "DevToolsLongComboDropdown": LongComboDropdown,
    "DevToolsNodeWithOptionalInput": NodeWithOptionalInput,
    "DevToolsNodeWithOptionalComboInput": NodeWithOptionalComboInput,
    "DevToolsNodeWithOnlyOptionalInput": NodeWithOnlyOptionalInput,
    "DevToolsNodeWithOutputList": NodeWithOutputList,
    "DevToolsNodeWithForceInput": NodeWithForceInput,
    "DevToolsNodeWithDefaultInput": NodeWithDefaultInput,
    "DevToolsNodeWithStringInput": NodeWithStringInput,
    "DevToolsNodeWithUnionInput": NodeWithUnionInput,
    "DevToolsNodeWithBooleanInput": NodeWithBooleanInput,
    "DevToolsSimpleSlider": SimpleSlider,
    "DevToolsNodeWithSeedInput": NodeWithSeedInput,
    "DevToolsNodeWithValidation": NodeWithValidation,
    "DevToolsNodeWithV2ComboInput": NodeWithV2ComboInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DevToolsLongComboDropdown": "Long Combo Dropdown",
    "DevToolsNodeWithOptionalInput": "Node With Optional Input",
    "DevToolsNodeWithOptionalComboInput": "Node With Optional Combo Input",
    "DevToolsNodeWithOnlyOptionalInput": "Node With Only Optional Input",
    "DevToolsNodeWithOutputList": "Node With Output List",
    "DevToolsNodeWithForceInput": "Node With Force Input",
    "DevToolsNodeWithDefaultInput": "Node With Default Input",
    "DevToolsNodeWithStringInput": "Node With String Input",
    "DevToolsNodeWithUnionInput": "Node With Union Input",
    "DevToolsNodeWithBooleanInput": "Node With Boolean Input",
    "DevToolsSimpleSlider": "Simple Slider",
    "DevToolsNodeWithSeedInput": "Node With Seed Input",
    "DevToolsNodeWithValidation": "Node With Validation",
    "DevToolsNodeWithV2ComboInput": "Node With V2 Combo Input",
}

__all__ = [
    "LongComboDropdown",
    "NodeWithOptionalInput",
    "NodeWithOptionalComboInput",
    "NodeWithOnlyOptionalInput",
    "NodeWithOutputList",
    "NodeWithForceInput",
    "NodeWithDefaultInput",
    "NodeWithStringInput",
    "NodeWithUnionInput",
    "NodeWithBooleanInput",
    "SimpleSlider",
    "NodeWithSeedInput",
    "NodeWithValidation",
    "NodeWithV2ComboInput",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

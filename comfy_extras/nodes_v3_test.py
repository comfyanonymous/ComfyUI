import torch
from comfy_api.v3.io import (
    ComfyNodeV3, SchemaV3, InputBehavior, NumberDisplay,
    IntegerInput, MaskInput, ImageInput, ComboInput, CustomInput,
    IntegerOutput, ImageOutput,
    NodeOutput,
)


class V3TestNode(ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return SchemaV3(
            node_id="V3TestNode1",
            display_name="V3 Test Node",
            description="This is a funky V3 node test.",
            category="v3 nodes",
            inputs=[
                ImageInput("image", display_name="new_image"),
                CustomInput("xyz", "XYZ"),
                MaskInput("mask", behavior=InputBehavior.optional),
                IntegerInput("some_int", display_name="new_name", min=0, max=127, default=42,
                             tooltip="My tooltip 😎", display_mode=NumberDisplay.slider),
                ComboInput("combo", options=["a", "b", "c"], tooltip="This is a combo input"),
                # ComboInput("combo", image_upload=True, image_folder=FolderType.output,
                #             remote=RemoteOptions(
                #                 route="/internal/files/output",
                #                 refresh_button=True,
                #             ),
                #             tooltip="This is a combo input"),
                # IntegerInput("some_int", display_name="new_name", min=0, tooltip="My tooltip 😎", display=NumberDisplay.slider, ),
                # ComboDynamicInput("mask", behavior=InputBehavior.optional),
                # IntegerInput("some_int", display_name="new_name", min=0, tooltip="My tooltip 😎", display=NumberDisplay.slider,
                #              dependent_inputs=[ComboDynamicInput("mask", behavior=InputBehavior.optional)],
                #              dependent_values=[lambda my_value: IO.STRING if my_value < 5 else IO.NUMBER],
                #              ),
                # ["option1", "option2". "option3"]
                # ComboDynamicInput["sdfgjhl", [ComboDynamicOptions("option1", [IntegerInput("some_int", display_name="new_name", min=0, tooltip="My tooltip 😎", display=NumberDisplay.slider, ImageInput(), MaskInput(), String()]),
                #                              CombyDynamicOptons("option2", [])
                #                                                   ]]
            ],
            outputs=[
                IntegerOutput("int_output"),
                ImageOutput("img_output", display_name="img🖼️", tooltip="This is an image"),
            ],
            is_output_node=True,
        )

    def execute(image: torch.Tensor, xyz, some_int: int, combo: str, mask: torch.Tensor=None):
        return NodeOutput(some_int, image)


NODES_LIST: list[ComfyNodeV3] = [
    V3TestNode,
]

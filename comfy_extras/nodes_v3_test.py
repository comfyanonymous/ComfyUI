import torch
from comfy_api.v3.io import (
    ComfyNodeV3, SchemaV3, NumberDisplay,
    IntegerInput, MaskInput, ImageInput, ComboInput, CustomInput, StringInput, CustomType,
    IntegerOutput, ImageOutput, MultitypedInput, InputV3, OutputV3,
    NodeOutput, Hidden
)
import logging


class XYZInput(InputV3, io_type="XYZ"):
    Type = tuple[int,str]

class XYZOutput(OutputV3, io_type="XYZ"):
    ...


class V3TestNode(ComfyNodeV3):

    def __init__(self):
        self.hahajkunless = ";)"

    @classmethod
    def DEFINE_SCHEMA(cls):
        return SchemaV3(
            node_id="V3TestNode1",
            display_name="V3 Test Node",
            description="This is a funky V3 node test.",
            category="v3 nodes",
            inputs=[
                ImageInput("image", display_name="new_image"),
                XYZInput("xyz", optional=True),
                #CustomInput("xyz", "XYZ", optional=True),
                MaskInput("mask", optional=True),
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
            hidden=[
                
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image: ImageInput.Type, some_int: IntegerInput.Type, combo: ComboInput.Type, xyz: XYZInput.Type=None, mask: MaskInput.Type=None):
        if hasattr(cls, "hahajkunless"):
            raise Exception("The 'cls' variable leaked instance state between runs!")
        if hasattr(cls, "doohickey"):
            raise Exception("The 'cls' variable leaked state on class properties between runs!")
        cls.doohickey = "LOLJK"
        return NodeOutput(some_int, image)


NODES_LIST: list[ComfyNodeV3] = [
    V3TestNode,
]

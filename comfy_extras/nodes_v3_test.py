import torch

from comfy_api.v3.io import (
    ComfyNodeV3, SchemaV3, CustomType, CustomInput, CustomOutput, InputBehavior, NumberDisplay,
    IntegerInput, MaskInput, ImageInput, ComboDynamicInput,
)





class V3TestNode(ComfyNodeV3):


    @classmethod
    def DEFINE_SCHEMA(cls):
        schema = SchemaV3(
            node_id="V3TestNode1",
            display_name="V3 Test Node (1djekjd)",
            description="This is a funky V3 node test.",
            category="v3 nodes",
            inputs=[
                IntegerInput("some_int", display_name="new_name", min=0, tooltip="My tooltip 😎"),
                MaskInput("mask", behavior=InputBehavior.optional),
                ImageInput("image", display_name="new_image"),
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
            is_output_node=True,
        )
        return schema

    def execute(self, some_int: int, image: torch.Tensor, mask: torch.Tensor=None, **kwargs):
        return (None,)






NODES: list[ComfyNodeV3] = [
    V3TestNode,
]


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for node in NODES:
    schema = node.GET_SCHEMA()
    NODE_CLASS_MAPPINGS[schema.node_id] = node
    if schema.display_name:
        NODE_DISPLAY_NAME_MAPPINGS[schema.node_id] = schema.display_name

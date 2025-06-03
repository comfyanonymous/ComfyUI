import torch

from comfy_api.v3.io import (
    ComfyNodeV3, SchemaV3, CustomType, CustomInput, CustomOutput, InputBehavior, NumberDisplay,
    IntegerInput, MaskInput, ImageInput, ComboInput, NodeOutput, FolderType, RemoteOptions
)





class V3TestNode(ComfyNodeV3):


    @classmethod
    def DEFINE_SCHEMA(cls):
        return SchemaV3(
            node_id="V3TestNode1",
            display_name="V3 Test Node (1djekjd)",
            description="This is a funky V3 node test.",
            category="v3 nodes",
            inputs=[
                IntegerInput("some_int", display_name="new_name", min=0, tooltip="My tooltip 😎", display_mode=NumberDisplay.slider),
                MaskInput("mask", behavior=InputBehavior.optional),
                ImageInput("image", display_name="new_image"),
                ComboInput("combo", image_upload=True, image_folder=FolderType.output,
                            remote=RemoteOptions(
                                route="/internal/files/output",
                                refresh_button=True,
                            ),
                            tooltip="This is a combo input"),
                # ComboInput("combo", options=["a", "b", "c"], tooltip="This is a combo input"),
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

    def execute(self, some_int: int, image: torch.Tensor, mask: torch.Tensor=None, **kwargs):
        a  = NodeOutput(1)
        aa = NodeOutput(1, "hellothere")
        ab = NodeOutput(1, "hellothere", ui={"lol": "jk"})
        b  = NodeOutput()
        c  = NodeOutput(ui={"lol": "jk"})
        return NodeOutput()
        return NodeOutput(1)
        return NodeOutput(1, block_execution="Kill yourself")
        return ()






NODES_LIST: list[ComfyNodeV3] = [
    V3TestNode,
]

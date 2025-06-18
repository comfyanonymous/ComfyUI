import torch
from comfy_api.v3_01 import io
import logging


@io.comfytype(io_type="XYZ")
class XYZ:
    Type = tuple[int,str]
    class Input(io.InputV3):
        ...
    class Output(io.OutputV3):
        ...

class MyState(io.NodeState):
    my_str: str
    my_int: int


class V3TestNode(io.ComfyNodeV3):

    state: MyState

    def __init__(self):
        super().__init__()
        self.hahajkunless = ";)"

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="V3_01_TestNode1",
            display_name="V3_01 Test Node",
            description="This is a funky V3_01 node test.",
            category="v3 nodes",
            inputs=[
                io.Image.Input("image", display_name="new_image"),
                XYZ.Input("xyz", optional=True),
                io.Custom("JKL").Input("jkl", optional=True),
                #JKL.Input("jkl", optional=True),
                #CustomInput("xyz", "XYZ", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Int.Input("some_int", display_name="new_name", min=0, max=127, default=42,
                             tooltip="My tooltip 😎", display_mode=io.NumberDisplay.slider),
                io.Combo.Input("combo", options=["a", "b", "c"], tooltip="This is a combo input"),
                io.MultiCombo.Input("combo2", options=["a","b","c"]),
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
                io.Int.Output("int_output"),
                io.Image.Output("img_output", display_name="img🖼️", tooltip="This is an image"),
            ],
            hidden=[
                
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image: io.Image.Type, some_int: int, combo: io.Combo.Type, combo2: io.MultiCombo.Type, xyz: XYZ.Type=None, mask: io.Mask.Type=None, **kwargs):
        zzz = cls.hidden.prompt
        cls.state.my_str = "LOLJK"
        expected_int = 123
        cls.state.my_int = expected_int
        if cls.state.my_int is None:
            cls.state.my_int = expected_int
        else:
            if cls.state.my_int != expected_int:
                raise Exception(f"Explicit state object did not maintain expected value: {cls.state.my_int} != {expected_int}")
        #some_int
        if hasattr(cls, "hahajkunless"):
            raise Exception("The 'cls' variable leaked instance state between runs!")
        if hasattr(cls, "doohickey"):
            raise Exception("The 'cls' variable leaked state on class properties between runs!")
        cls.doohickey = "LOLJK"
        return io.NodeOutput(some_int, image)


NODES_LIST: list[io.ComfyNodeV3] = [
    V3TestNode,
]

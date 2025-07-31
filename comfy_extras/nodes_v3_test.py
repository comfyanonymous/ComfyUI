import torch
import time
from comfy_api.latest import io, ui, _io, ComfyExtension
import logging  # noqa
import comfy.utils
import asyncio
from typing_extensions import override

@io.comfytype(io_type="XYZ")
class XYZ(io.ComfyTypeIO):
    Type = tuple[int,str]


class V3TestNode(io.ComfyNode):
    # NOTE: this is here just to test that state is not leaking
    def __init__(self):
        super().__init__()
        self.hahajkunless = ";)"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="V3_01_TestNode1",
            display_name="V3 Test Node",
            category="v3 nodes",
            description="This is a funky V3 node test.",
            inputs=[
                io.Image.Input("image", display_name="new_image"),
                XYZ.Input("xyz", optional=True),
                io.Custom("JKL").Input("jkl", optional=True),
                io.Mask.Input("mask", display_name="mask haha", optional=True),
                io.Int.Input("some_int", display_name="new_name", min=0, max=127, default=42,
                             tooltip="My tooltip 😎", display_mode=io.NumberDisplay.slider),
                io.Combo.Input("combo", options=["a", "b", "c"], tooltip="This is a combo input"),
                io.MultiCombo.Input("combo2", options=["a","b","c"]),
                io.MultiType.Input(io.Int.Input("int_multitype", display_name="haha"), types=[io.Float]),
                io.MultiType.Input("multitype", types=[io.Mask, io.Float, io.Int], optional=True),
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
                io.Int.Output(),
                io.Image.Output(display_name="img🖼️", tooltip="This is an image"),
            ],
            hidden=[
                io.Hidden.prompt,
                io.Hidden.auth_token_comfy_org,
                io.Hidden.unique_id,
            ],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, image: io.Image.Type, some_int: int, combo: io.Combo.Type, combo2: io.MultiCombo.Type, xyz: XYZ.Type=None, mask: io.Mask.Type=None, **kwargs):
        if some_int < 0:
            raise Exception("some_int must be greater than 0")
        if combo == "c":
            raise Exception("combo must be a or b")
        return True

    @classmethod
    def execute(cls, image: io.Image.Type, some_int: int, combo: io.Combo.Type, combo2: io.MultiCombo.Type, xyz: XYZ.Type=None, mask: io.Mask.Type=None, **kwargs):
        if hasattr(cls, "hahajkunless"):
            raise Exception("The 'cls' variable leaked instance state between runs!")
        if hasattr(cls, "doohickey"):
            raise Exception("The 'cls' variable leaked state on class properties between runs!")
        try:
            cls.doohickey = "LOLJK"
        except AttributeError:
            pass
        return io.NodeOutput(some_int, image, ui=ui.PreviewImage(image, cls=cls))


# class V3LoraLoader(io.ComfyNode):
#     @classmethod
#     def define_schema(cls):
#         return io.Schema(
#             node_id="V3_LoraLoader",
#             display_name="V3 LoRA Loader",
#             category="v3 nodes",
#             description="LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together.",
#             inputs=[
#                 io.Model.Input("model", tooltip="The diffusion model the LoRA will be applied to."),
#                 io.Clip.Input("clip", tooltip="The CLIP model the LoRA will be applied to."),
#                 io.Combo.Input(
#                     "lora_name",
#                     options=folder_paths.get_filename_list("loras"),
#                     tooltip="The name of the LoRA."
#                 ),
#                 io.Float.Input(
#                     "strength_model",
#                     default=1.0,
#                     min=-100.0,
#                     max=100.0,
#                     step=0.01,
#                     tooltip="How strongly to modify the diffusion model. This value can be negative."
#                 ),
#                 io.Float.Input(
#                     "strength_clip",
#                     default=1.0,
#                     min=-100.0,
#                     max=100.0,
#                     step=0.01,
#                     tooltip="How strongly to modify the CLIP model. This value can be negative."
#                 ),
#             ],
#             outputs=[
#                 io.Model.Output(),
#                 io.Clip.Output(),
#             ],
#         )

#     @classmethod
#     def execute(cls, model: io.Model.Type, clip: io.Clip.Type, lora_name: str, strength_model: float, strength_clip: float, **kwargs):
#         if strength_model == 0 and strength_clip == 0:
#             return io.NodeOutput(model, clip)

#         lora = cls.resources.get(resources.TorchDictFolderFilename("loras", lora_name))

#         model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
#         return io.NodeOutput(model_lora, clip_lora)


class NInputsTest(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="V3_NInputsTest",
            display_name="V3 N Inputs Test",
            inputs=[
                _io.AutogrowDynamic.Input("nmock", template_input=io.Image.Input("image"), min=1, max=3),
                _io.AutogrowDynamic.Input("nmock2", template_input=io.Int.Input("int"), optional=True, min=1, max=4),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def validate_inputs(cls, nmock, nmock2):
        return True

    @classmethod
    def fingerprint_inputs(cls, nmock, nmock2):
        return time.time()

    @classmethod
    def check_lazy_status(cls, **kwargs) -> list[str]:
        need = [name for name in kwargs if kwargs[name] is None]
        return need

    @classmethod
    def execute(cls, nmock, nmock2):
        first_image = nmock[0]
        all_images = []
        for img in nmock:
            if img.shape != first_image.shape:
                img = img.movedim(-1,1)
                img = comfy.utils.common_upscale(img, first_image.shape[2], first_image.shape[1], "lanczos", "center")
                img = img.movedim(1,-1)
            all_images.append(img)
        combined_image = torch.cat(all_images, dim=0)
        return io.NodeOutput(combined_image)


class V3TestSleep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="V3_TestSleep",
            display_name="V3 Test Sleep",
            category="_for_testing",
            description="Test async sleep functionality.",
            inputs=[
                io.AnyType.Input("value", display_name="Value"),
                io.Float.Input("seconds", display_name="Seconds", default=1.0, min=0.0, max=9999.0, step=0.01, tooltip="The amount of seconds to sleep."),
            ],
            outputs=[
                io.AnyType.Output(),
            ],
            hidden=[
                io.Hidden.unique_id,
            ],
            is_experimental=True,
        )

    @classmethod
    async def execute(cls, value: io.AnyType.Type, seconds: io.Float.Type, **kwargs):
        logging.info(f"V3TestSleep: {cls.hidden.unique_id}")
        pbar = comfy.utils.ProgressBar(seconds, node_id=cls.hidden.unique_id)
        start = time.time()
        expiration = start + seconds
        now = start
        while now < expiration:
            now = time.time()
            pbar.update_absolute(now - start)
            await asyncio.sleep(0.02)
        return io.NodeOutput(value)


class V3DummyStart(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="V3_DummyStart",
            display_name="V3 Dummy Start",
            category="v3 nodes",
            description="This is a dummy start node.",
            inputs=[],
            outputs=[
                io.Custom("XYZ").Output(),
            ],
        )

    @classmethod
    def execute(cls):
        return io.NodeOutput(None)


class V3DummyEnd(io.ComfyNode):
    COOL_VALUE = 123

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="V3_DummyEnd",
            display_name="V3 Dummy End",
            category="v3 nodes",
            description="This is a dummy end node.",
            inputs=[
                io.Custom("XYZ").Input("xyz"),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def custom_action(cls):
        return 456

    @classmethod
    def execute(cls, xyz: io.Custom("XYZ").Type):
        logging.info(f"V3DummyEnd: {cls.COOL_VALUE}")
        logging.info(f"V3DummyEnd: {cls.custom_action()}")
        return


class V3DummyEndInherit(V3DummyEnd):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "V3_DummyEndInherit"
        schema.display_name = "V3 Dummy End Inherit"
        return schema

    @classmethod
    def execute(cls, xyz: io.Custom("XYZ").Type):
        logging.info(f"V3DummyEndInherit: {cls.COOL_VALUE}")
        return super().execute(xyz)

NODES_LIST: list[type[io.ComfyNode]] = [
    V3TestNode,
    # V3LoraLoader,
    NInputsTest,
    V3TestSleep,
    V3DummyStart,
    V3DummyEnd,
    V3DummyEndInherit,
]

class v3TestExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODES_LIST

async def comfy_entrypoint() -> v3TestExtension:
    return v3TestExtension()

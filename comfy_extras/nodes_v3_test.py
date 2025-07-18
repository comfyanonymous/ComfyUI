import torch
import time
from comfy_api.v3 import io, ui, resources
import logging  # noqa
import folder_paths
import comfy.utils
import comfy.sd
import asyncio


@io.comfytype(io_type="XYZ")
class XYZ:
    Type = tuple[int,str]
    class Input(io.InputV3):
        ...
    class Output(io.OutputV3):
        ...


class V3TestNode(io.ComfyNodeV3):
    class State(io.NodeState):
        my_str: str
        my_int: int
    state: State

    def __init__(self):
        super().__init__()
        self.hahajkunless = ";)"

    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
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
        zzz = cls.hidden.prompt
        cls.state.my_str = "LOLJK"
        expected_int = 123
        if "thing" not in cls.state:
            cls.state["thing"] = "hahaha"
            yyy = cls.state["thing"]    # noqa
            del cls.state["thing"]
        if cls.state.get_value("int2") is None:
            cls.state.set_value("int2", 123)
            zzz = cls.state.get_value("int2")   # noqa
            cls.state.pop("int2")
        if cls.state.my_int is None:
            cls.state.my_int = expected_int
        else:
            if cls.state.my_int != expected_int:
                raise Exception(f"Explicit state object did not maintain expected value (__getattr__/__setattr__): {cls.state.my_int} != {expected_int}")
        #some_int
        if hasattr(cls, "hahajkunless"):
            raise Exception("The 'cls' variable leaked instance state between runs!")
        if hasattr(cls, "doohickey"):
            raise Exception("The 'cls' variable leaked state on class properties between runs!")
        try:
            cls.doohickey = "LOLJK"
        except AttributeError:
            pass
        return io.NodeOutput(some_int, image, ui=ui.PreviewImage(image, cls=cls))


class V3LoraLoader(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="V3_LoraLoader",
            display_name="V3 LoRA Loader",
            category="v3 nodes",
            description="LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together.",
            inputs=[
                io.Model.Input("model", tooltip="The diffusion model the LoRA will be applied to."),
                io.Clip.Input("clip", tooltip="The CLIP model the LoRA will be applied to."),
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                    tooltip="The name of the LoRA."
                ),
                io.Float.Input(
                    "strength_model",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="How strongly to modify the diffusion model. This value can be negative."
                ),
                io.Float.Input(
                    "strength_clip",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="How strongly to modify the CLIP model. This value can be negative."
                ),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
            ],
        )

    @classmethod
    def execute(cls, model: io.Model.Type, clip: io.Clip.Type, lora_name: str, strength_model: float, strength_clip: float, **kwargs):
        if strength_model == 0 and strength_clip == 0:
            return io.NodeOutput(model, clip)

        lora = cls.resources.get(resources.TorchDictFolderFilename("loras", lora_name))

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return io.NodeOutput(model_lora, clip_lora)


class NInputsTest(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="V3_NInputsTest",
            display_name="V3 N Inputs Test",
            inputs=[
                io.AutogrowDynamic.Input("nmock", template_input=io.Image.Input("image"), min=1, max=3),
                io.AutogrowDynamic.Input("nmock2", template_input=io.Int.Input("int"), optional=True, min=1, max=4),
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


class V3TestSleep(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
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


NODES_LIST: list[type[io.ComfyNodeV3]] = [
    V3TestNode,
    V3LoraLoader,
    NInputsTest,
    V3TestSleep,
]

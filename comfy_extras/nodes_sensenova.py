from typing_extensions import override

from comfy.ldm.sensenova.sampling import SenseNovaModelSampling
from comfy_api.latest import ComfyExtension, io


class SenseNovaSamplingOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaSamplingOptions",
            display_name="SenseNova Sampling Options",
            category="model/patch/sensenova",
            description="Set the SenseNova flow shift.",
            inputs=[
                io.Model.Input(id="model"),
                io.Float.Input(id="shift", default=3.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, *, model, shift: float) -> io.NodeOutput:
        patched = model.clone()
        model_sampling = SenseNovaModelSampling(patched.model.model_config)
        model_sampling.set_parameters(shift=shift)
        patched.add_object_patch("model_sampling", model_sampling)
        return io.NodeOutput(patched)


class SenseNovaTextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaTextEncode",
            display_name="SenseNova 1.x Text Encode",
            category="model/conditioning/sensenova",
            description="Encode a SenseNova prompt with optional image-generation reasoning.",
            inputs=[
                io.Clip.Input(id="clip"),
                io.String.Input(id="text", multiline=True, dynamic_prompts=True),
                io.Boolean.Input(id="thinking", default=False),
                io.Int.Input(
                    id="max_think_tokens",
                    default=1024,
                    min=1,
                    advanced=True,
                ),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(
        cls, *, clip, text: str, thinking: bool, max_think_tokens: int
    ) -> io.NodeOutput:
        tokens = clip.tokenize(text, thinking=thinking)
        conditioning = clip.encode_from_tokens_scheduled(
            tokens,
            add_dict={
                "sensenova_thinking": thinking,
                "sensenova_max_think_tokens": max_think_tokens,
            },
        )
        return io.NodeOutput(conditioning)


class SenseNovaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SenseNovaTextEncode,
            SenseNovaSamplingOptions,
        ]


async def comfy_entrypoint() -> SenseNovaExtension:
    return SenseNovaExtension()

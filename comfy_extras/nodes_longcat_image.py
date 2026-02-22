from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class CLIPTextEncodeLongCatImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeLongCatImage",
            display_name="CLIP Text Encode (LongCat-Image)",
            category="advanced/conditioning/longcat",
            description="Text encoding for LongCat-Image with character-level quoted text support. Wrap text in quotes for accurate text rendering.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
                io.Float.Input("guidance", default=4.0, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, text, guidance) -> io.NodeOutput:
        tokens = clip.tokenize(text)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance}))

    encode = execute


class LongCatImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeLongCatImage,
        ]


async def comfy_entrypoint() -> LongCatImageExtension:
    return LongCatImageExtension()

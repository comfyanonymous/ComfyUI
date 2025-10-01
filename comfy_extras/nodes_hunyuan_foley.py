import torch
import comfy.model_management
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

class EmptyLatentHunyuanFoley(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyLatentHunyuanFoley",
            display_name="EmptyLatentHunyuanFoley",
            category="audio/latent",
            inputs = [
                io.Int.Input("length", min = 1, max = 15, default = 12),
                io.Int.Input("batch_size", min = 1, max = 48_000, default = 1),
                io.Video.Input("video", optional=True),
            ],
            outputs=[io.Latent.Output(display_name="latent")]
        )
    @classmethod
    def execute(cls, length, batch_size, video = None):
        if video is not None:
            length = video.size(0)
            length /= 25
        shape = (batch_size, 128, int(50 * length))
        latent = torch.randn(shape, device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent, "type": "hunyuan_foley"}, )

class HunyuanFoleyConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanFoleyConditioning",
            display_name="HunyuanFoleyConditioning",
            category="conditioning/video_models",
            inputs = [
                io.Conditioning.Input("siglip_encoding_1"),
                io.Conditioning.Input("synchformer_encoding_2"),
                io.Conditioning.Input("text_encoding"),
            ],
            outputs=[io.Conditioning.Output(display_name= "positive"), io.Conditioning.Output(display_name="negative")]
        )

    @classmethod
    def execute(cls, siglip_encoding_1, synchformer_encoding_2, text_encoding):

        if isinstance(text_encoding, list):
            text_encoding = text_encoding[0]

        embeds = torch.cat([siglip_encoding_1, synchformer_encoding_2, text_encoding], dim = 0)
        positive = [[embeds, {}]]
        negative = [[torch.zeros_like(embeds), {}]]
        return io.NodeOutput(positive, negative)

class FoleyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            HunyuanFoleyConditioning,
            EmptyLatentHunyuanFoley
        ]

async def comfy_entrypoint() -> FoleyExtension:
    return FoleyExtension()

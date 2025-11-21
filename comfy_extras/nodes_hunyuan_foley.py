import torch
import comfy.model_management
import torch.nn.functional as F
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
                io.Float.Input("length", min = 1.0, max = 15.0, default = 12.0),
                io.Int.Input("batch_size", min = 1, max = 48_000, default = 1),
                io.Video.Input("video", optional=True),
            ],
            outputs=[io.Latent.Output(display_name="latent")]
        )
    @classmethod
    def execute(cls, length, batch_size, video = None):
        if video is not None:
            video = video.images
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
                io.Conditioning.Input("text_encoding_positive"),
                io.Conditioning.Input("text_encoding_negative"),
            ],
            outputs=[io.Conditioning.Output(display_name= "positive"), io.Conditioning.Output(display_name="negative")]
        )

    @classmethod
    def execute(cls, siglip_encoding_1, synchformer_encoding_2, text_encoding_positive, text_encoding_negative):

        text_encoding_positive = text_encoding_positive[0][0]
        text_encoding_negative = text_encoding_negative[0][0]
        all_ = (siglip_encoding_1, synchformer_encoding_2, text_encoding_positive, text_encoding_negative)
        biggest = max([t.size(1) for t in all_])
        siglip_encoding_1, synchformer_encoding_2, text_encoding_positive, text_encoding_negative = [
            F.pad(t, (0, 0, 0, biggest - t.size(1), 0, 0)) for t in all_
        ]
        positive_tensor = torch.cat([siglip_encoding_1, synchformer_encoding_2, text_encoding_positive])
        negative_tensor = torch.cat([torch.zeros_like(siglip_encoding_1), torch.zeros_like(synchformer_encoding_2), text_encoding_negative])
        negative = [[positive_tensor.view(1, -1, siglip_encoding_1.size(-1)), {}]]
        positive = [[negative_tensor.view(1, -1, siglip_encoding_1.size(-1)), {}]]

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

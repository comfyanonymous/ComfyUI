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

        max_l = max([t.size(1) for t in all_])
        max_d = max([t.size(2) for t in all_])

        def repeat_shapes(max_value, input, dim = 1):
            # temporary repeat values on the cpu
            factor_pos, remainder = divmod(max_value, input.shape[dim])

            positions = [1] * input.ndim 
            positions[dim] = factor_pos
            input = input.cpu().repeat(*positions)

            if remainder > 0:
                pad = input[:, :remainder, :]
                input = torch.cat([input, pad], dim =1)

            return input
        
        siglip_encoding_1, synchformer_encoding_2, text_encoding_positive, text_encoding_negative = [repeat_shapes(max_l, t) for t in all_]
        siglip_encoding_1, synchformer_encoding_2, text_encoding_positive, text_encoding_negative = [repeat_shapes(max_d, t, dim = 2) for t in all_]

        embeds = torch.cat([siglip_encoding_1.cpu(), synchformer_encoding_2.cpu()], dim = 0)

        x = siglip_encoding_1
        negative = [[torch.cat([torch.zeros_like(embeds), text_encoding_negative]).contiguous().view(1, -1, x.size(-1)).pin_memory(), {}]]
        positive = [[torch.cat([embeds, text_encoding_positive]).contiguous().view(1, -1, x.size(-1)).pin_memory(), {}]]

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

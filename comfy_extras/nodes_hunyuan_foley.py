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
                io.Float.Input("length", min = 1.0, max = 15.0, default = 12.0),
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

class CpuLockedTensor(torch.Tensor):
    def __new__(cls, data):
        base = torch.as_tensor(data, device='cpu')
        return torch.Tensor._make_subclass(cls, base, require_grad=False)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):

        if kwargs is None:
            kwargs = {}

        # if any of the args/kwargs were CpuLockedTensor, it will cause infinite recursion
        def unwrap(x):
            return x.as_subclass(torch.Tensor) if isinstance(x, CpuLockedTensor) else x

        unwrapped_args = torch.utils._pytree.tree_map(unwrap, args)
        unwrapped_kwargs = torch.utils._pytree.tree_map(unwrap, kwargs)

        result = func(*unwrapped_args, **unwrapped_kwargs)

        # rewrap the resulted tensors
        if isinstance(result, torch.Tensor):
            return CpuLockedTensor(result.detach().cpu())
        elif isinstance(result, (list, tuple)):
            return type(result)(
                CpuLockedTensor(x.detach().cpu()) if isinstance(x, torch.Tensor) else x
                for x in result
            )
        return result

    def to(self, *args, allow_gpu=False, **kwargs):
        if allow_gpu:
            return super().to(*args, **kwargs)
        return self.detach().clone().cpu()

    def cuda(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def pin_memory(self):
        return self

    def detach(self):
        out = super().detach()
        return CpuLockedTensor(out)

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

            if input.shape[dim] == max_value:
                return input

            # temporary repeat values on the cpu
            factor_pos, remainder = divmod(max_value, input.shape[dim])

            positions = [1] * input.ndim
            positions[dim] = factor_pos
            input = input.cpu().repeat(*positions)

            if remainder > 0:
                if dim == 1:
                    pad = input[:, :remainder, :]
                else:
                    pad = input[:, :, :remainder]
                input = torch.cat([input, pad], dim = dim)

            return input

        siglip_encoding_1, synchformer_encoding_2, text_encoding_positive, text_encoding_negative = [repeat_shapes(max_l, t) for t in all_]
        siglip_encoding_1, synchformer_encoding_2, text_encoding_positive, text_encoding_negative = [repeat_shapes(max_d, t, dim = 2) for t in
                                                                                                    (siglip_encoding_1, synchformer_encoding_2, text_encoding_positive, text_encoding_negative)]

        embeds = torch.cat([siglip_encoding_1.cpu(), synchformer_encoding_2.cpu()], dim = 0)

        x = siglip_encoding_1
        positive_tensor = CpuLockedTensor(torch.cat([torch.zeros_like(embeds), text_encoding_negative])
                                          .contiguous().view(1, -1, x.size(-1)))
        negative_tensor = CpuLockedTensor(torch.cat([embeds, text_encoding_positive])
                                          .contiguous().view(1, -1, x.size(-1)))

        negative = [[positive_tensor, {}]]
        positive = [[negative_tensor, {}]]

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

import torch
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


class CFGRenormLongCatImage(io.ComfyNode):
    """Per-patch CFG renormalization matching HuggingFace's LongCat-Image pipeline.

    After standard CFG combination, rescales the noise prediction at each 2x2 patch
    so its norm doesn't exceed the conditional prediction's norm.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGRenormLongCatImage",
            display_name="CFG Renorm (LongCat-Image)",
            category="advanced/model/longcat",
            description="Applies per-patch CFG renormalization used by the LongCat-Image pipeline. Connect between the model loader and the sampler.",
            inputs=[
                io.Model.Input("model"),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        def cfg_renorm_post(args):
            denoised = args["denoised"]
            cond_denoised = args["cond_denoised"]
            x = args["input"]

            B, C, H, W = denoised.shape
            ps = 2

            noise = x - denoised
            noise_cond = x - cond_denoised

            noise_packed = noise.reshape(B, C, H // ps, ps, W // ps, ps) \
                               .permute(0, 2, 4, 1, 3, 5) \
                               .reshape(B, -1, C * ps * ps)
            cond_packed = noise_cond.reshape(B, C, H // ps, ps, W // ps, ps) \
                                    .permute(0, 2, 4, 1, 3, 5) \
                                    .reshape(B, -1, C * ps * ps)

            noise_norm = torch.norm(noise_packed, dim=-1, keepdim=True)
            cond_norm = torch.norm(cond_packed, dim=-1, keepdim=True)

            scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=0.0, max=1.0)

            renormed = (noise_packed * scale) \
                .reshape(B, H // ps, W // ps, C, ps, ps) \
                .permute(0, 3, 1, 4, 2, 5) \
                .reshape(B, C, H, W)

            return x - renormed

        m = model.clone()
        m.set_model_sampler_post_cfg_function(cfg_renorm_post, disable_cfg1_optimization=True)
        return io.NodeOutput(m)


class LongCatImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeLongCatImage,
            CFGRenormLongCatImage,
        ]


async def comfy_entrypoint() -> LongCatImageExtension:
    return LongCatImageExtension()

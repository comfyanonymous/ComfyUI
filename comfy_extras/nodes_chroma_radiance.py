from typing_extensions import override

import torch

import comfy.model_management
from comfy_api.latest import ComfyExtension, io

import nodes

class EmptyChromaRadianceLatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmptyChromaRadianceLatentImage",
            category="latent/chroma_radiance",
            inputs=[
                io.Int.Input(id="width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input(id="height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input(id="batch_size", default=1, min=1, max=4096),
            ],
            outputs=[io.Latent().Output()],
        )

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int=1) -> io.NodeOutput:
        latent = torch.zeros((batch_size, 3, height, width), device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples":latent})


class ChromaRadianceStubVAE:
    @classmethod
    def encode(cls, pixels: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        device = comfy.model_management.intermediate_device()
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
        elif pixels.ndim != 4:
            raise ValueError("Unexpected input image shape")
        dims = pixels.shape[1:-1]
        for d in range(len(dims)):
            d_adj = (dims[d] // 16) * 16
            if d_adj == d:
                continue
            d_offset = (dims[d] % 16) // 2
            pixels = pixels.narrow(d + 1, d_offset, d_adj)
        h, w, c = pixels.shape[1:]
        if h < 16 or w < 16:
            raise ValueError("Chroma Radiance image inputs must have height/width of at least 16 pixels.")
        pixels= pixels[..., :3]
        if c == 1:
            pixels = pixels.expand(-1, -1, -1, 3)
        elif c != 3:
            raise ValueError("Unexpected number of channels in input image")
        latent = pixels.to(device=device, dtype=torch.float32, copy=True)
        latent = latent.clamp_(0, 1).movedim(-1, 1).contiguous()
        latent -= 0.5
        latent *= 2
        return latent.clamp_(-1, 1)

    @classmethod
    def decode(cls, samples: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        device = comfy.model_management.intermediate_device()
        img = samples.to(device=device, dtype=torch.float32, copy=True)
        img = img.clamp_(-1, 1).movedim(1, -1).contiguous()
        img += 1.0
        img *= 0.5
        return img.clamp_(0, 1)

    encode_tiled = encode
    decode_tiled = decode

    @classmethod
    def spacial_compression_decode(cls) -> int:
        return 1

    spacial_compression_encode = spacial_compression_decode
    temporal_compression_decode = spacial_compression_decode


class ChromaRadianceLatentToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ChromaRadianceLatentToImage",
            category="latent/chroma_radiance",
            description="For use with Chroma Radiance. Converts an input LATENT to IMAGE.",
            inputs=[io.Latent.Input(id="latent")],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, *, latent: dict) -> io.NodeOutput:
        return io.NodeOutput(ChromaRadianceStubVAE.decode(latent["samples"]))


class ChromaRadianceImageToLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ChromaRadianceImageToLatent",
            category="latent/chroma_radiance",
            description="For use with Chroma Radiance. Converts an input IMAGE to LATENT. Note: Radiance requires inputs with width/height that are multiples of 16 so your image will be cropped if necessary.",
            inputs=[io.Image.Input(id="image")],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, *, image: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput({"samples": ChromaRadianceStubVAE.encode(image)})


class ChromaRadianceStubVAENode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ChromaRadianceStubVAE",
            category="vae/chroma_radiance",
            description="For use with Chroma Radiance. Allows converting between latent and image types with nodes that require a VAE input. Note: Radiance requires inputs with width/height that are multiples of 16 so your image will be cropped if necessary.",
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(ChromaRadianceStubVAE())


class ChromaRadianceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EmptyChromaRadianceLatentImage,
            ChromaRadianceLatentToImage,
            ChromaRadianceImageToLatent,
            ChromaRadianceStubVAENode,
        ]


async def comfy_entrypoint() -> ChromaRadianceExtension:
    return ChromaRadianceExtension()

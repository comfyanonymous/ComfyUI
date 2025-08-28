import torch

import comfy.model_management

import nodes

class EmptyChromaRadianceLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"

    CATEGORY = "latent/chroma_radiance"

    def go(self, *, width, height, batch_size=1):
        latent = torch.zeros((batch_size, 3, height, width), device=self.device)
        return ({"samples":latent}, )


class ChromaRadianceLatentToImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {"required": {"latent": ("LATENT",)}}

    DESCRIPTION = "For use with Chroma Radiance. Converts an input LATENT to IMAGE."
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "go"

    CATEGORY = "latent/chroma_radiance"

    def go(self, *, latent: dict) -> tuple[torch.Tensor]:
        img = latent["samples"].to(device=self.device, dtype=torch.float32, copy=True)
        img = img.clamp_(-1, 1).movedim(1, -1).contiguous()
        img += 1.0
        img *= 0.5
        return (img.clamp_(0, 1),)

class ChromaRadianceImageToLatent:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {"required": {"image": ("IMAGE",)}}

    DESCRIPTION = "For use with Chroma Radiance. Converts an input IMAGE to LATENT. Note: Radiance requires inputs with width/height that are multiples of 16 so your image will be cropped if necessary."
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"

    CATEGORY = "latent/chroma_radiance"

    def go(self, *, image: torch.Tensor) -> tuple[dict]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim != 4:
            raise ValueError("Unexpected input image shape")
        dims = image.shape[1:-1]
        for d in range(len(dims)):
            d_adj = (dims[d] // 16) * 16
            if d_adj == d:
                continue
            d_offset = (dims[d] % 16) // 2
            image = image.narrow(d + 1, d_offset, d_adj)
        h, w, c = image.shape[1:]
        if h < 16 or w < 16:
            raise ValueError("Chroma Radiance image inputs must have height/width of at least 16 pixels.")
        image = image[..., :3]
        if c == 1:
            image = image.expand(-1, -1, -1, 3)
        elif c != 3:
            raise ValueError("Unexpected number of channels in input image")
        latent = image.to(device=self.device, dtype=torch.float32, copy=True)
        latent = latent.clamp_(0, 1).movedim(-1, 1).contiguous()
        latent -= 0.5
        latent *= 2
        return ({"samples": latent.clamp_(-1, 1)},)

class ChromaRadianceStubVAE:
    def __init__(self):
        self.image_to_latent = ChromaRadianceImageToLatent()
        self.latent_to_image = ChromaRadianceLatentToImage()

    DESCRIPTION = "For use with Chroma Radiance. Allows converting between latent and image types with nodes that require a VAE input. Note: Radiance requires inputs with width/height that are multiples of 16 so your image will be cropped if necessary."
    RETURN_TYPES = ("VAE",)
    FUNCTION = "go"

    CATEGORY = "vae/chroma_radiance"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {}

    def go(self) -> tuple["ChromaRadianceStubVAE"]:
        return (self,)

    def encode(self, pixels: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        return self.image_to_latent.go(image=pixels)[0]["samples"]

    encode_tiled = encode

    def decode(self, samples: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        return self.latent_to_image.go(latent={"samples": samples})[0]

    decode_tiled = decode

    def spacial_compression_decode(self) -> int:
        return 1

    spacial_compression_encode = spacial_compression_decode
    temporal_compression_decode = spacial_compression_decode


NODE_CLASS_MAPPINGS = {
    "EmptyChromaRadianceLatentImage": EmptyChromaRadianceLatentImage,
    "ChromaRadianceLatentToImage": ChromaRadianceLatentToImage,
    "ChromaRadianceImageToLatent": ChromaRadianceImageToLatent,
    "ChromaRadianceStubVAE": ChromaRadianceStubVAE,
}

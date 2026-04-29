import torch
from comfy_api.latest import IO
from typing_extensions import override
from comfy_api.latest import ComfyExtension

# Rec.709 to Rec.2020 Gamut Conversion Matrix
M_709_to_2020 = torch.tensor([[0.6274, 0.3293, 0.0433],[0.0691, 0.9195, 0.0114],[0.0164, 0.0880, 0.8956]
])

# Rec.2020 to Rec.709 Gamut Conversion Matrix
M_2020_to_709 = torch.tensor([[ 1.6605, -0.5876, -0.0728],[-0.1246,  1.1329, -0.0083],[-0.0182, -0.1006,  1.1187]
])

def srgb_to_linear(tensor):
    mask = tensor <= 0.04045
    return torch.where(mask, tensor / 12.92, torch.pow((tensor + 0.055) / 1.055, 2.4))

def linear_to_srgb(tensor):
    mask = tensor <= 0.0031308
    return torch.where(mask, tensor * 12.92, 1.055 * torch.pow(tensor.clamp(min=1e-8), 1.0 / 2.4) - 0.055)

def linear_to_pq(linear_tensor):
    """SMPTE ST 2084 (PQ) encoding"""
    m1, m2 = (2610 / 4096 / 4), (2523 / 4096 * 128)
    c1, c2, c3 = (3424 / 4096), (2413 / 4096 * 32), (2392 / 4096 * 32)
    l_norm = torch.clamp(linear_tensor, 0.0, 1.0)
    l_m1 = torch.pow(l_norm, m1)
    return torch.pow((c1 + c2 * l_m1) / (1 + c3 * l_m1), m2)

def pq_to_linear(pq_tensor):
    """Inverse SMPTE ST 2084 (PQ) decoding"""
    m1, m2 = (2610 / 4096 / 4), (2523 / 4096 * 128)
    c1, c2, c3 = (3424 / 4096), (2413 / 4096 * 32), (2392 / 4096 * 32)
    n = torch.pow(torch.clamp(pq_tensor, 0.0, 1.0), 1/m2)
    return torch.pow(torch.clamp((n - c1) / (c2 - c3 * n), min=0.0), 1/m1)

class ConvertColorSpace(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Convert Color Space",
            category="image/color",
            inputs=[
                IO.Image.Input("images"),
                IO.Combo.Input("source_color_space", options=["sRGB", "Linear", "HDR (Rec.2020)", "Grayscale"], default="sRGB"),
                IO.Combo.Input("target_color_space", options=["sRGB", "Linear", "HDR (Rec.2020)", "Grayscale"], default="Linear"),
            ],
            outputs=[
                IO.Image.Output("images"),
            ]
        )

    @classmethod
    def execute(cls, images, source_color_space, target_color_space) -> IO.NodeOutput:
        img_tensor = images.clone()
        device = img_tensor.device

        has_alpha = img_tensor.shape[-1] == 4
        alpha = img_tensor[..., 3:4] if has_alpha else None
        rgb = img_tensor[..., :3]

        # turn source into linear
        if source_color_space == "sRGB":
            rgb = srgb_to_linear(rgb)

        elif source_color_space == "Grayscale":
            # assume Grayscale has sRGB gamma
            luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            rgb = luma.unsqueeze(-1).repeat(1, 1, 1, 3)
            rgb = srgb_to_linear(rgb)

        elif source_color_space == "HDR (Rec.2020)":
            # assuming Linear Rec.2020 input. Convert to Linear Rec.709
            matrix = M_2020_to_709.to(device)
            rgb = pq_to_linear(rgb)
            rgb = torch.matmul(rgb, matrix.T)


        # turn source into target space
        if target_color_space == "sRGB":
            rgb = linear_to_srgb(rgb)

        elif target_color_space == "Grayscale":
            luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            rgb = luma.unsqueeze(-1).repeat(1, 1, 1, 3)
            rgb = linear_to_srgb(rgb) # reapply srgb gamma

        elif target_color_space == "HDR (Rec.2020)":
            # convert Gamut from Linear Rec.709 to Linear Rec.2020
            rgb = torch.matmul(rgb, M_709_to_2020.to(device).T).clamp(min=0)
            rgb = linear_to_pq(rgb)

        img_tensor = torch.cat([rgb, alpha], dim=-1) if has_alpha else rgb

        return IO.NodeOutput(images=img_tensor)


class ConvertColorSpaceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ConvertColorSpace
        ]


async def comfy_entrypoint() -> ConvertColorSpaceExtension:
    return ConvertColorSpaceExtension()

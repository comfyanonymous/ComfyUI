
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io, ui
import torch
import math
from einops import rearrange

from torchvision.transforms import functional as TVF
from torchvision.transforms import Lambda, Normalize
from torchvision.transforms.functional import InterpolationMode


def area_resize(image, max_area):

    height, width = image.shape[-2:]
    scale = math.sqrt(max_area / (height * width))

    resized_height, resized_width = round(height * scale), round(width * scale)

    return TVF.resize(
        image,
        size=(resized_height, resized_width),
        interpolation=InterpolationMode.BICUBIC,
    )

def crop(image, factor):
    height_factor, width_factor = factor
    height, width = image.shape[-2:]

    cropped_height = height - (height % height_factor)
    cropped_width = width - (width % width_factor)

    image = TVF.center_crop(img=image, output_size=(cropped_height, cropped_width))
    return image

def cut_videos(videos):
    t = videos.size(1)
    if t == 1:
        return videos
    if t <= 4 :
        padding = [videos[:, -1].unsqueeze(1)] * (4 - t + 1)
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        return videos
    if (t - 1) % (4) == 0:
        return videos
    else:
        padding = [videos[:, -1].unsqueeze(1)] * (
            4 - ((t - 1) % (4))
        )
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        assert (videos.size(1) - 1) % (4) == 0
        return videos
    
class SeedVR2InputProcessing(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id = "SeedVR2InputProcessing",
            category="image/video",
            inputs = [
                io.Image.Input("images"),
                io.Int.Input("resolution_height"),
                io.Int.Input("resolution_width")
            ],
            outputs = [
                io.Image.Output("images")
            ]
        )
    
    @classmethod
    def execute(cls, images, resolution_height, resolution_width):
        max_area = ((resolution_height * resolution_width)** 0.5) ** 2
        clip = Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        normalize = Normalize(0.5, 0.5)
        images = area_resize(images, max_area)
        images = clip(images)
        images = crop(images, (16, 16))
        images = normalize(images)
        images = rearrange(images, "t c h w -> c t h w")
        images = cut_videos(images)
        return

class SeedVR2Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2Conditioning",
            category="image/video",
            inputs=[
                io.Conditioning.Input("text_positive_conditioning"),
                io.Conditioning.Input("text_negative_conditioning"),
                io.Conditioning.Input("vae_conditioning")
            ],
            outputs=[io.Conditioning.Output("positive"), io.Conditioning.Output("negative")],
        )

    @classmethod
    def execute(cls, text_positive_conditioning, text_negative_conditioning, vae_conditioning) -> io.NodeOutput:
        # TODO
        pos_cond = text_positive_conditioning[0][0]
        neg_cond = text_negative_conditioning[0][0]

        return io.NodeOutput()

class SeedVRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SeedVR2Conditioning,
            SeedVR2InputProcessing
        ]

async def comfy_entrypoint() -> SeedVRExtension:
    return SeedVRExtension()
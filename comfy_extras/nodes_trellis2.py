from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO
import torch
from comfy.ldm.trellis2.model import SparseTensor
import comfy.model_management
from PIL import Image
import PIL
import numpy as np

shape_slat_normalization = {
    "mean": torch.tensor([
        0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
        -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
        0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
        -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665
    ])[None],
    "std": torch.tensor([
        5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
        5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
        4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
        5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691
    ])[None]
}

tex_slat_normalization = {
    "mean": torch.tensor([
        3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364, 0.439898, -0.928075,
        0.029398, -0.339596, -0.869527, 1.038479, -0.972385, 0.126042, -1.129303, 0.455149,
        -1.209521, 2.069067, 0.544735, 2.569128, -0.323407, 2.293000, -1.925608, -1.217717,
        1.213905, 0.971588, -0.023631, 0.106750, 2.021786, 0.250524, -0.662387, -0.768862
    ])[None],
    "std": torch.tensor([
        2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316, 2.144656, 2.911822,
        2.969419, 2.501689, 2.154811, 3.163343, 2.621215, 2.381943, 3.186697, 3.021588,
        2.295916, 3.234985, 3.233086, 2.260140, 2.874801, 2.810596, 3.292720, 2.674999,
        2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849, 2.786195, 2.775190
    ])[None]
}

def smart_crop_square(
    image: torch.Tensor,
    background_color=(128, 128, 128),
):
    C, H, W = image.shape
    size = max(H, W)
    canvas = torch.empty(
        (C, size, size),
        dtype=image.dtype,
        device=image.device
    )
    for c in range(C):
        canvas[c].fill_(background_color[c])
    top = (size - H) // 2
    left = (size - W) // 2
    canvas[:, top:top + H, left:left + W] = image

    return canvas

def run_conditioning(
    model,
    image: torch.Tensor,
    include_1024: bool = True,
    background_color: str = "black",
):
    # TODO: should check if normalization was applied in these steps
    model = model.model
    device = comfy.model_management.intermediate_device() # replaces .cpu()
    torch_device =  comfy.model_management.get_torch_device() # replaces .cuda()
    bg_colors = {
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "white": (255, 255, 255),
    }
    bg_color = bg_colors.get(background_color, (128, 128, 128))

    # Convert image to PIL
    if image.dim() == 4:
        pil_image = (image[0] * 255).clip(0, 255).astype(torch.uint8)
    else:
        pil_image = (image * 255).clip(0, 255).astype(torch.uint8)

    pil_image = smart_crop_square(pil_image, background_color=bg_color)

    model.image_size = 512
    def set_image_size(image, image_size=512):
        image = PIL.from_array(image)
        image = [i.resize((image_size, image_size), Image.LANCZOS) for i in image]
        image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
        image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
        image = torch.stack(image).to(torch_device)
        return image

    pil_image = set_image_size(image, 512)
    cond_512 = model([pil_image])

    cond_1024 = None
    if include_1024:
        model.image_size = 1024
        pil_image = set_image_size(pil_image, 1024)
        cond_1024 = model([pil_image])

    neg_cond = torch.zeros_like(cond_512)

    conditioning = {
        'cond_512': cond_512.to(device),
        'neg_cond': neg_cond.to(device),
    }
    if cond_1024 is not None:
        conditioning['cond_1024'] = cond_1024.to(device)

    preprocessed_tensor = pil_image.to(torch.float32) / 255.0
    preprocessed_tensor = torch.from_numpy(preprocessed_tensor).unsqueeze(0)

    return conditioning, preprocessed_tensor

class VaeDecodeShapeTrellis(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VaeDecodeShapeTrellis",
            category="latent/3d",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
                IO.Int.Input("resolution", tooltip="Shape Generation Resolution"),
            ],
            outputs=[
                IO.Mesh.Output("mesh"),
                IO.AnyType.Output("shape_subs"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae, resolution):
        std = shape_slat_normalization["std"]
        mean = shape_slat_normalization["mean"]
        samples = samples * std + mean

        mesh, subs = vae.decode_shape_slat(resolution, samples)
        return mesh, subs

class VaeDecodeTextureTrellis(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VaeDecodeTextureTrellis",
            category="latent/3d",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
                IO.AnyType.Input("shape_subs"),
            ],
            outputs=[
                IO.Mesh.Output("mesh"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae, shape_subs):
        if shape_subs is None:
            raise ValueError("Shape subs must be provided for texture generation")

        std = tex_slat_normalization["std"]
        mean = tex_slat_normalization["mean"]
        samples = samples * std + mean

        mesh = vae.decode_tex_slat(samples, shape_subs)
        return mesh

class Trellis2Conditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2Conditioning",
            category="conditioning/video_models",
            inputs=[
                IO.ClipVision.Input("clip_vision_model"),
                IO.Image.Input("image"),
                IO.MultiCombo.Input("background_color", options=["black", "gray", "white"], default="black")
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, clip_vision_model, image, background_color) -> IO.NodeOutput:
        # could make 1024 an option
        conditioning, _ = run_conditioning(clip_vision_model, image, include_1024=True, background_color=background_color)
        embeds = conditioning["cond_1024"] # should add that
        positive = [[conditioning["cond_512"], {embeds}]]
        negative = [[conditioning["cond_neg"], {embeds}]]
        return IO.NodeOutput(positive, negative)

class EmptyShapeLatentTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyLatentTrellis2",
            category="latent/3d",
            inputs=[
                IO.Latent.Input("structure_output"),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )
    
    def execute(cls, structure_output):
        # i will see what i have to do here
        coords = structure_output or structure_output.coords
        in_channels = 32
        latent = SparseTensor(feats=torch.randn(coords.shape[0], in_channels), coords=coords)
        return IO.NodeOutput({"samples": latent, "type": "trellis2"})

class EmptyTextureLatentTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyLatentTrellis2",
            category="latent/3d",
            inputs=[
                IO.Latent.Input("structure_output"),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )
    
    def execute(cls, structure_output):
        # TODO
        in_channels = 32
        latent = structure_output.replace(feats=torch.randn(structure_output.coords.shape[0], in_channels - structure_output.feats.shape[1]))
        return IO.NodeOutput({"samples": latent, "type": "trellis2"})

class EmptyStructureLatentTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyLatentTrellis2",
            category="latent/3d",
            inputs=[
                IO.Int.Input("resolution", default=3072, min=1, max=8192),
                IO.Int.Input("batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )
    
    def execute(cls, res, batch_size):
        in_channels = 32
        latent = torch.randn(batch_size, in_channels, res, res, res)
        return IO.NodeOutput({"samples": latent, "type": "trellis2"})


class Trellis2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Trellis2Conditioning,
            EmptyShapeLatentTrellis2,
            EmptyStructureLatentTrellis2,
            EmptyTextureLatentTrellis2,
            VaeDecodeTextureTrellis,
            VaeDecodeShapeTrellis
        ]


async def comfy_entrypoint() -> Trellis2Extension:
    return Trellis2Extension()

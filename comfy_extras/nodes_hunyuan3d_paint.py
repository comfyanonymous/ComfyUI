"""Native ComfyUI nodes for Hunyuan3D 2.1 PBR paint (multiview texture generation).

Given a mesh's rendered world-space normal + position multiview maps and a reference
image, the multiview node runs the hunyuan3d-paintpbr-v2-1 UNet to produce per-view
albedo and metallic-roughness IMAGE batches, which can then be baked to UV textures.
"""

import torch
from typing_extensions import override

import comfy.model_management
import comfy.utils
import folder_paths
from comfy.ldm.hunyuan3d.paint.loader import load_paint_unet
from comfy.ldm.hunyuan3d.paint.sampler import generate_multiview, SD_SCALING_FACTOR
from comfy_api.latest import ComfyExtension, IO

PAINT_MODEL = IO.Custom("HUNYUAN3D_PAINT_MODEL")
DINO_FEATURES = IO.Custom("HY3D_DINO_FEATURES")


def _parse_floats(text, count):
    vals = [float(x) for x in str(text).replace(" ", "").split(",") if x != ""]
    if not vals:
        vals = [0.0]
    while len(vals) < count:
        vals.append(vals[-1])
    return vals[:count]


class Hunyuan3DPaintModelLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3DPaintModelLoader",
            display_name="Hunyuan3D Paint Model Loader",
            category="loaders/hunyuan 3d",
            description="Load a Hunyuan3D 2.1 PBR paint UNet (hunyuan3d-paintpbr-v2-1).",
            inputs=[
                IO.Combo.Input("model_name", options=folder_paths.get_filename_list("diffusion_models")),
            ],
            outputs=[
                PAINT_MODEL.Output(display_name="paint_model"),
            ],
        )

    @classmethod
    def execute(cls, model_name) -> IO.NodeOutput:
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        sd = comfy.utils.load_torch_file(path)
        patcher, config = load_paint_unet(sd)
        return IO.NodeOutput((patcher, config))


class Hunyuan3DPaintMultiView(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3DPaintMultiView",
            display_name="Hunyuan3D Paint MultiView",
            category="model/hunyuan 3d",
            description=(
                "Generate multiview albedo and metallic-roughness images for a mesh from its "
                "rendered normal + position maps and a reference image."
            ),
            inputs=[
                PAINT_MODEL.Input("paint_model"),
                IO.Vae.Input("vae", tooltip="SD-2.x image VAE from the paint model (hunyuan3d-paintpbr-v2-1/vae)."),
                IO.Image.Input("reference_image", tooltip="White-background reference image."),
                IO.Image.Input("normal_maps", tooltip="World-space normal maps, one per view."),
                IO.Image.Input("position_maps", tooltip="Position maps, one per view (same order as normals)."),
                IO.Int.Input("steps", default=15, min=1, max=100),
                IO.Float.Input("guidance_scale", default=3.0, min=0.0, max=30.0, step=0.1),
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                IO.String.Input("camera_azimuths", default="0,90,180,270,0,180",
                                tooltip="Per-view azimuths (degrees) used for view-dependent guidance scaling."),
                DINO_FEATURES.Input("dino_features", optional=True,
                                    tooltip="Optional precomputed DINOv2 reference tokens (improves fidelity)."),
            ],
            outputs=[
                IO.Image.Output(display_name="albedo"),
                IO.Image.Output(display_name="mr"),
            ],
        )

    @classmethod
    def execute(cls, paint_model, vae, reference_image, normal_maps, position_maps, steps,
                guidance_scale, seed, camera_azimuths, dino_features=None) -> IO.NodeOutput:
        patcher, config = paint_model
        comfy.model_management.load_models_gpu([patcher])
        model = patcher.model
        device = patcher.load_device
        dtype = getattr(model, "manual_cast_dtype", None) or next(model.parameters()).dtype

        num_views = normal_maps.shape[0]
        azims = _parse_floats(camera_azimuths, num_views)

        def enc(image):
            latent = vae.encode(image[:, :, :, :3])
            return latent.to(device=device, dtype=dtype) * SD_SCALING_FACTOR

        ref_latent = enc(reference_image[:1])
        normal_latents = enc(normal_maps)
        position_latents = enc(position_maps)
        pos_maps = position_maps[:, :, :, :3].permute(0, 3, 1, 2).contiguous().to(device=device, dtype=dtype)

        dino = None
        if dino_features is not None:
            dino = dino_features.to(device=device, dtype=dtype)

        out = generate_multiview(
            model, config, ref_latent, normal_latents, position_latents, pos_maps,
            dino_features=dino, camera_azims=azims, num_inference_steps=steps,
            guidance_scale=guidance_scale, seed=seed, device=device, dtype=dtype)

        pbr = list(config["pbr_setting"])
        albedo = vae.decode(out[pbr[0]] / SD_SCALING_FACTOR)
        if "mr" in out:
            mr = vae.decode(out["mr"] / SD_SCALING_FACTOR)
        else:
            mr = torch.zeros_like(albedo)
        return IO.NodeOutput(albedo, mr)


class Hunyuan3DPaintExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Hunyuan3DPaintModelLoader,
            Hunyuan3DPaintMultiView,
        ]


async def comfy_entrypoint() -> Hunyuan3DPaintExtension:
    return Hunyuan3DPaintExtension()

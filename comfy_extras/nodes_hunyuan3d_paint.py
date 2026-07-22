"""Native ComfyUI nodes for Hunyuan3D 2.1 PBR paint (multiview texture generation).

Given a mesh and a reference image, the multiview node renders the geometry conditioning
(world-space normal + position maps) with a torch-native rasterizer, runs the
hunyuan3d-paintpbr-v2-1 UNet to produce per-view albedo and metallic-roughness IMAGE
batches, and the bake node back-projects those views onto a UV atlas to produce the
albedo + metallic-roughness textures and a UV-unwrapped, textured mesh.
"""

import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.model_management
import comfy.utils
import folder_paths
from comfy.ldm.hunyuan3d.paint import render as paint_render
from comfy.ldm.hunyuan3d.paint.loader import load_paint_unet
from comfy.ldm.hunyuan3d.paint.sampler import generate_multiview, SD_SCALING_FACTOR
from comfy_extras.nodes_save_3d import get_mesh_batch_item
from comfy_api.latest import ComfyExtension, IO, Types

PAINT_MODEL = IO.Custom("HUNYUAN3D_PAINT_MODEL")
DINO_FEATURES = IO.Custom("HY3D_DINO_FEATURES")
CAMERAS = IO.Custom("HY3D_CAMERAS")


def _first_mesh(mesh):
    """Return (vertices (N,3), faces (M,3) long) for batch item 0 of a MESH."""
    vertices, faces, _colors, uvs = get_mesh_batch_item(mesh, 0)
    return vertices.float(), faces.long(), uvs


def _prep_reference(image, size):
    """White-composite (if RGBA) and resize a reference IMAGE to (size, size, 3)."""
    img = image[:1]
    if img.shape[-1] == 4:
        rgb, alpha = img[..., :3], img[..., 3:4]
        img = rgb * alpha + (1.0 - alpha)
    else:
        img = img[..., :3]
    img = img.permute(0, 3, 1, 2)
    img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
    return img.permute(0, 2, 3, 1).contiguous()


def _pack_mr_gltf(mr_native):
    """Repack the model's native MR image (R=metallic, G=roughness) into the glTF
    metallicRoughness layout (R=1, G=roughness, B=metallic)."""
    out = torch.ones_like(mr_native)
    out[..., 1] = mr_native[..., 1]  # roughness -> G
    out[..., 2] = mr_native[..., 0]  # metallic  -> B
    return out


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
                "Render a mesh's geometry conditioning (normal + position maps) and run the "
                "2.1 paint UNet to generate multiview albedo and metallic-roughness images."
            ),
            inputs=[
                PAINT_MODEL.Input("paint_model"),
                IO.Vae.Input("vae", tooltip="SD-2.x image VAE from the paint model (hunyuan3d-paintpbr-v2-1/vae)."),
                IO.Mesh.Input("mesh", tooltip="Untextured mesh to paint (e.g. from Hunyuan3D 2.1 shape)."),
                IO.Image.Input("reference_image", tooltip="Reference image; alpha is composited over white."),
                IO.Int.Input("num_views", default=6, min=1, max=6,
                             tooltip="Number of standard views (front, right, back, left, top, bottom)."),
                IO.Int.Input("resolution", default=512, min=256, max=1024, step=64,
                             tooltip="Per-view render/diffusion resolution."),
                IO.Int.Input("steps", default=15, min=1, max=100),
                IO.Float.Input("guidance_scale", default=3.0, min=0.0, max=30.0, step=0.1),
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                DINO_FEATURES.Input("dino_features", optional=True,
                                    tooltip="Optional precomputed DINOv2 reference tokens (improves fidelity)."),
            ],
            outputs=[
                IO.Image.Output(display_name="albedo"),
                IO.Image.Output(display_name="mr"),
                CAMERAS.Output(display_name="cameras"),
                IO.Image.Output(display_name="normal_maps"),
                IO.Image.Output(display_name="position_maps"),
            ],
        )

    @classmethod
    def execute(cls, paint_model, vae, mesh, reference_image, num_views, resolution, steps,
                guidance_scale, seed, dino_features=None) -> IO.NodeOutput:
        patcher, config = paint_model
        comfy.model_management.load_models_gpu([patcher])
        model = patcher.model
        device = patcher.load_device
        dtype = getattr(model, "manual_cast_dtype", None) or next(model.parameters()).dtype

        vertices, faces, _uvs = _first_mesh(mesh)
        cameras = paint_render.standard_cameras(num_views)

        normal_maps, position_maps, _masks = paint_render.render_geometry_maps(
            vertices, faces, cameras, resolution=resolution)
        # move to CPU float for the VAE, which handles its own device placement
        normal_maps = normal_maps.cpu().float()
        position_maps = position_maps.cpu().float()

        reference = _prep_reference(reference_image, resolution)

        def enc(image):
            latent = vae.encode(image[:, :, :, :3])
            return latent.to(device=device, dtype=dtype) * SD_SCALING_FACTOR

        ref_latent = enc(reference)
        normal_latents = enc(normal_maps)
        position_latents = enc(position_maps)
        pos_maps = position_maps[:, :, :, :3].permute(0, 3, 1, 2).contiguous().to(device=device, dtype=dtype)

        dino = None
        if dino_features is not None:
            dino = dino_features.to(device=device, dtype=dtype)

        out = generate_multiview(
            model, config, ref_latent, normal_latents, position_latents, pos_maps,
            dino_features=dino, camera_azims=cameras.azims, num_inference_steps=steps,
            guidance_scale=guidance_scale, seed=seed, device=device, dtype=dtype)

        pbr = list(config["pbr_setting"])
        albedo = vae.decode(out[pbr[0]] / SD_SCALING_FACTOR)
        if "mr" in out:
            mr = vae.decode(out["mr"] / SD_SCALING_FACTOR)
        else:
            mr = torch.zeros_like(albedo)
        return IO.NodeOutput(albedo, mr, cameras, normal_maps, position_maps)


class Hunyuan3DBakeMultiView(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3DBakeMultiView",
            display_name="Hunyuan3D Bake MultiView",
            category="model/hunyuan 3d",
            description=(
                "Back-project multiview albedo + metallic-roughness images onto the mesh's UV "
                "atlas and attach the baked textures. A mesh that already has UVs (artist-made "
                "or from an unwrap node) is baked onto those UVs; a mesh without UVs falls back "
                "to a built-in per-triangle atlas. Outputs the textured mesh and the baked "
                "albedo/MR textures as images."
            ),
            inputs=[
                IO.Mesh.Input("mesh", tooltip="The mesh that was painted (same one fed to the multiview node). "
                                              "Existing per-vertex UVs are used as the bake target; a mesh "
                                              "without UVs gets a built-in per-triangle atlas."),
                IO.Image.Input("albedo", tooltip="Per-view albedo images from the multiview node."),
                IO.Image.Input("mr", tooltip="Per-view metallic-roughness images from the multiview node."),
                CAMERAS.Input("cameras", tooltip="Cameras output by the multiview node."),
                IO.Int.Input("texture_size", default=1024, min=256, max=4096, step=256),
                IO.Float.Input("bake_exponent", default=4.0, min=1.0, max=16.0, step=0.5, advanced=True,
                               tooltip="Cosine weighting exponent; higher favours front-facing views."),
                IO.Boolean.Input("fill_holes", default=True, advanced=True,
                                 tooltip="Fill unseen texels by dilating nearest valid colours."),
            ],
            outputs=[
                IO.Mesh.Output(display_name="mesh"),
                IO.Image.Output(display_name="albedo_texture"),
                IO.Image.Output(display_name="mr_texture"),
                IO.Image.Output(display_name="albedo_views"),
                IO.Image.Output(display_name="mr_views"),
            ],
        )

    @classmethod
    def execute(cls, mesh, albedo, mr, cameras, texture_size, bake_exponent, fill_holes) -> IO.NodeOutput:
        vertices, faces, uvs = _first_mesh(mesh)

        # A mesh that already carries per-vertex UVs (artist-authored or produced by an
        # unwrap node) keeps them as the bake target; only UV-less meshes fall back to
        # the built-in per-triangle atlas.
        if uvs is not None and uvs.shape[0] == vertices.shape[0]:
            v_uv, f_uv, uv = vertices, faces, uvs.float()
        else:
            v_uv, f_uv, uv = paint_render.pack_per_triangle_uv(vertices, faces)

        albedo_views = albedo[..., :3].float()
        mr_views = mr[..., :3].float()

        albedo_tex, mask_a = paint_render.bake_multiview(
            v_uv, f_uv, uv, albedo_views, cameras, texture_size=texture_size, bake_exp=bake_exponent)
        mr_tex, mask_mr = paint_render.bake_multiview(
            v_uv, f_uv, uv, mr_views, cameras, texture_size=texture_size, bake_exp=bake_exponent)

        if fill_holes:
            albedo_tex = paint_render.fill_holes(albedo_tex, mask_a)
            mr_tex = paint_render.fill_holes(mr_tex, mask_mr)

        albedo_tex = albedo_tex.clamp(0.0, 1.0)
        mr_tex = mr_tex.clamp(0.0, 1.0)

        out_mesh = Types.MESH(
            v_uv.unsqueeze(0), f_uv.unsqueeze(0), uvs=uv.unsqueeze(0),
            texture=albedo_tex.unsqueeze(0),
            texture_mr=_pack_mr_gltf(mr_tex).unsqueeze(0))

        return IO.NodeOutput(out_mesh, albedo_tex.unsqueeze(0), mr_tex.unsqueeze(0), albedo, mr)


class Hunyuan3DPaintExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Hunyuan3DPaintModelLoader,
            Hunyuan3DPaintMultiView,
            Hunyuan3DBakeMultiView,
        ]


async def comfy_entrypoint() -> Hunyuan3DPaintExtension:
    return Hunyuan3DPaintExtension()

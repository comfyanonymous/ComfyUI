"""Unit tests for the torch-native Hunyuan3D 2.1 paint renderer, UV unwrap and baker.

All CPU, no weights. Covers the barycentric z-buffer rasterizer (known triangle ->
known pixels, occlusion), the geometry conditioning maps, the per-triangle UV atlas
(non-overlapping charts, all triangles packed), the multiview baker (flat-colour
round-trip + MR channel preservation), hole filling, and node schema/wiring including
the textured-GLB write-back with a metallic-roughness texture.
"""

from __future__ import annotations

import inspect
import json
import struct

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.hunyuan3d.paint import render as R  # noqa: E402


# --------------------------------------------------------------------------- #
# Rasterizer
# --------------------------------------------------------------------------- #
def test_rasterize_known_triangle_to_known_pixels():
    # NDC triangle -> screen triangle (0,0),(4,0),(0,4): row grows with NDC +y (no
    # top/bottom flip, matching Tencent's custom_rasterizer pixel convention), so this
    # covers pixels where col + row <= ~4 (dense near row 0, tapering by row 3).
    verts = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]])
    face_id, bary = R.rasterize(verts, faces, 4, 4)
    expected = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, -1],
        [0, 0, -1, -1],
        [0, -1, -1, -1],
    ])
    assert torch.equal(face_id, expected)
    # barycentric weights of covered pixels sum to 1
    covered = face_id >= 0
    assert torch.allclose(bary[covered].sum(-1), torch.ones(int(covered.sum())), atol=1e-5)


def test_rasterize_barycentric_center():
    # A triangle whose centroid lands on a pixel center yields ~(1/3, 1/3, 1/3).
    verts = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]])
    face_id, bary = R.rasterize(verts, faces, 32, 32)
    covered = face_id >= 0
    assert covered.any()
    assert torch.allclose(bary[covered].sum(-1), torch.ones(int(covered.sum())), atol=1e-5)


def test_rasterize_z_buffer_occlusion():
    # Two full-covering triangles; the nearer (smaller z) one wins everywhere.
    near = torch.tensor([[-1.0, -1.0, -0.5], [3.0, -1.0, -0.5], [-1.0, 3.0, -0.5]])
    far = torch.tensor([[-1.0, -1.0, 0.5], [3.0, -1.0, 0.5], [-1.0, 3.0, 0.5]])
    verts = torch.cat([far, near], dim=0)  # face 0 = far, face 1 = near
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    face_id, _ = R.rasterize(verts, faces, 8, 8)
    covered = face_id >= 0
    assert covered.any()
    assert torch.all(face_id[covered] == 1)  # the near triangle


def test_rasterize_empty_faces():
    verts = torch.zeros((0, 3))
    faces = torch.zeros((0, 3), dtype=torch.long)
    face_id, bary = R.rasterize(verts, faces, 4, 4)
    assert torch.all(face_id == -1)
    assert bary.shape == (4, 4, 3)


# --------------------------------------------------------------------------- #
# Geometry conditioning maps
# --------------------------------------------------------------------------- #
def _quad():
    v = torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
    f = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return v, f


def test_render_geometry_maps_shapes_and_ranges():
    v, f = _quad()
    cams = R.standard_cameras(6)
    normals, positions, masks = R.render_geometry_maps(v, f, cams, resolution=32)
    assert normals.shape == (6, 32, 32, 3)
    assert positions.shape == (6, 32, 32, 3)
    assert masks.shape == (6, 32, 32)
    assert float(normals.min()) >= 0.0 and float(normals.max()) <= 1.0
    assert float(positions.min()) >= 0.0 and float(positions.max()) <= 1.0
    # background is white for both control maps
    bg = masks[0] < 0.5
    assert torch.allclose(normals[0][bg], torch.ones_like(normals[0][bg]))
    assert torch.allclose(positions[0][bg], torch.ones_like(positions[0][bg]))
    # a front-facing quad is visible from the front view
    assert float(masks[0].mean()) > 0.1


# --------------------------------------------------------------------------- #
# Per-triangle UV atlas
# --------------------------------------------------------------------------- #
def test_uv_unwrap_packs_all_triangles_non_overlapping():
    torch.manual_seed(0)
    v = torch.rand(10, 3)
    f = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6], [1, 3, 5], [0, 2, 7],
                      [6, 7, 8], [8, 9, 0], [1, 4, 7]])
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    Fn = f.shape[0]
    # mesh is unwelded: 3 unique verts/uvs per face, faces index them 0..3F-1
    assert nv.shape == (3 * Fn, 3)
    assert uv.shape == (3 * Fn, 2)
    assert torch.equal(nf, torch.arange(3 * Fn).reshape(Fn, 3))
    assert float(uv.min()) >= 0.0 and float(uv.max()) <= 1.0
    # each triangle's UVs live in its own grid cell -> charts cannot overlap
    import math
    g = int(math.ceil(math.sqrt(Fn)))
    cell = 1.0 / g
    tri_uv = uv.reshape(Fn, 3, 2)
    for i in range(Fn):
        c, rr = i % g, i // g
        assert torch.all(tri_uv[i, :, 0] >= c * cell - 1e-6)
        assert torch.all(tri_uv[i, :, 0] <= (c + 1) * cell + 1e-6)
        assert torch.all(tri_uv[i, :, 1] >= rr * cell - 1e-6)
        assert torch.all(tri_uv[i, :, 1] <= (rr + 1) * cell + 1e-6)


# --------------------------------------------------------------------------- #
# Multiview baker
# --------------------------------------------------------------------------- #
def _cube():
    v = torch.tensor([
        [-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
        [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5],
    ])
    f = torch.tensor([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ])
    return v, f


def test_bake_flat_color_round_trip():
    v, f = _cube()
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    cams = R.standard_cameras(6)
    color = torch.tensor([0.1, 0.7, 0.3])
    views = color.view(1, 1, 1, 3).expand(6, 48, 48, 3).contiguous()
    tex, mask = R.bake_multiview(nv, nf, uv, views, cams, texture_size=256)
    assert tex.shape == (256, 256, 3)
    assert int(mask.sum()) > 0
    covered = tex[mask]
    assert torch.allclose(covered, color.expand_as(covered), atol=1e-3)


def test_bake_channels_are_independent():
    # metallic in R, roughness in G survive baking independently.
    v, f = _cube()
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    cams = R.standard_cameras(6)
    mr = torch.zeros(6, 48, 48, 3)
    mr[..., 0] = 0.2
    mr[..., 1] = 0.9
    tex, mask = R.bake_multiview(nv, nf, uv, mr, cams, texture_size=256)
    covered = tex[mask]
    assert torch.allclose(covered[:, 0], torch.full((covered.shape[0],), 0.2), atol=1e-3)
    assert torch.allclose(covered[:, 1], torch.full((covered.shape[0],), 0.9), atol=1e-3)
    assert torch.allclose(covered[:, 2], torch.zeros(covered.shape[0]), atol=1e-3)


def test_fill_holes_fills_from_neighbors():
    tex = torch.zeros(16, 16, 3)
    mask = torch.zeros(16, 16, dtype=torch.bool)
    tex[8, 8] = torch.tensor([1.0, 0.0, 0.0])
    mask[8, 8] = True
    filled = R.fill_holes(tex, mask, max_iters=32)
    # the single valid red texel propagates outward to fill the whole map
    assert float((filled.abs().sum(-1) > 0).float().mean()) > 0.9


# --------------------------------------------------------------------------- #
# Node schema / wiring + textured-GLB write-back
# --------------------------------------------------------------------------- #
def _load_nodes():
    import comfy_extras.nodes_hunyuan3d_paint as N
    return N


def test_node_schema_matches_execute_signature():
    N = _load_nodes()
    for cls in (N.Hunyuan3DPaintModelLoader, N.Hunyuan3DPaintMultiView, N.Hunyuan3DBakeMultiView):
        ids = [i.id for i in cls.define_schema().inputs]
        params = [p for p in inspect.signature(cls.execute).parameters if p != "cls"]
        assert ids == params, (cls.__name__, ids, params)


def test_multiview_node_outputs_declared():
    N = _load_nodes()
    outs = [o.display_name for o in N.Hunyuan3DPaintMultiView.define_schema().outputs]
    assert outs == ["albedo", "mr", "cameras", "normal_maps", "position_maps"]
    outs = [o.display_name for o in N.Hunyuan3DBakeMultiView.define_schema().outputs]
    assert outs == ["mesh", "albedo_texture", "mr_texture", "albedo_views", "mr_views"]


def test_bake_node_execute_attaches_textures_and_uvs():
    from comfy_api.latest import Types
    N = _load_nodes()
    v, f = _cube()
    mesh = Types.MESH(v.unsqueeze(0), f.unsqueeze(0))
    cams = R.standard_cameras(6)
    albedo = torch.zeros(6, 48, 48, 3)
    albedo[..., 1] = 1.0  # green
    mr = torch.zeros(6, 48, 48, 3)
    mr[..., 0] = 0.2  # metallic
    mr[..., 1] = 0.8  # roughness
    out_mesh, alb_tex, mr_tex, alb_views, mr_views = N.Hunyuan3DBakeMultiView.execute(
        mesh, albedo, mr, cams, 256, 4.0, False)

    assert out_mesh.uvs is not None
    assert out_mesh.uvs.shape[1] == out_mesh.vertices.shape[1]  # UVs 1:1 with vertices
    assert out_mesh.texture is not None and out_mesh.texture.shape == (1, 256, 256, 3)
    assert out_mesh.texture_mr is not None and out_mesh.texture_mr.shape == (1, 256, 256, 3)
    # native MR image output keeps R=metallic, G=roughness
    covered = alb_tex[0].abs().sum(-1) > 0
    assert torch.allclose(mr_tex[0][covered][:, 0], torch.full((int(covered.sum()),), 0.2), atol=1e-3)
    # glTF-packed MR on the mesh: G=roughness, B=metallic
    packed = out_mesh.texture_mr[0]
    assert torch.allclose(packed[covered][:, 1], torch.full((int(covered.sum()),), 0.8), atol=1e-3)
    assert torch.allclose(packed[covered][:, 2], torch.full((int(covered.sum()),), 0.2), atol=1e-3)


def test_save_glb_writes_metallic_roughness_texture(tmp_path):
    from PIL import Image
    from comfy_extras.nodes_save_3d import save_glb
    v, f = _cube()
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    base = Image.new("RGB", (32, 32), (0, 255, 0))
    mrimg = Image.new("RGB", (32, 32), (255, 200, 40))
    path = str(tmp_path / "out.glb")
    save_glb(nv, nf, path, uvs=uv, texture_image=base, mr_texture_image=mrimg)

    with open(path, "rb") as fh:
        data = fh.read()
    assert data[:4] == b"glTF"
    jlen = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + jlen])
    assert len(gltf["images"]) == 2
    assert len(gltf["textures"]) == 2
    pbr = gltf["materials"][0]["pbrMetallicRoughness"]
    assert "baseColorTexture" in pbr
    assert "metallicRoughnessTexture" in pbr
    assert pbr["metallicRoughnessTexture"]["index"] == 1


def _fake_vae():
    class FakeVAE:
        def encode(self, img):
            x = img.permute(0, 3, 1, 2)
            lat = torch.nn.functional.interpolate(x, scale_factor=0.125, mode="bilinear", align_corners=False)
            return torch.cat([lat, lat[:, :1]], dim=1)  # 4-channel latent at H/8

        def decode(self, lat):
            x = torch.nn.functional.interpolate(lat[:, :3], scale_factor=8, mode="bilinear", align_corners=False)
            return x.permute(0, 2, 3, 1).clamp(0, 1)

    return FakeVAE()


def test_multiview_then_bake_end_to_end_with_random_model():
    """Exercise the full node glue: MESH + reference -> render -> multiview diffusion
    -> VAE decode -> bake -> textured mesh, on a small random-init UNet + stub VAE."""
    import comfy.ops as comfy_ops
    from comfy.ldm.hunyuan3d.paint.unet import UNet2p5DConditionModel
    from comfy.ldm.hunyuan3d.paint.loader import load_paint_unet
    from comfy_api.latest import Types
    N = _load_nodes()

    cfg = dict(
        in_channels=12, ref_in_channels=4, out_channels=4,
        block_out_channels=(64, 64, 128, 128), layers_per_block=2, cross_attention_dim=64,
        num_attention_heads=(1, 1, 2, 2), transformer_layers_per_block=1, norm_num_groups=32,
        pbr_setting=("albedo", "mr"), pbr_token_channels=7, dino_embeddings_dim=32, use_dino=True)
    model = UNet2p5DConditionModel(dtype=torch.float32, device="cpu",
                                   operations=comfy_ops.disable_weight_init, **cfg)
    model.eval()
    patcher, config = load_paint_unet(model.state_dict(), model_options={"dtype": torch.float32})

    v, f = _cube()
    mesh = Types.MESH(v.unsqueeze(0), f.unsqueeze(0))
    ref = torch.rand(1, 128, 128, 3)

    albedo, mr, cams, normals, positions = N.Hunyuan3DPaintMultiView.execute(
        (patcher, config), _fake_vae(), mesh, ref, 6, 128, 2, 3.0, 0)
    assert albedo.shape == (6, 128, 128, 3)
    assert mr.shape == (6, 128, 128, 3)
    assert normals.shape == (6, 128, 128, 3)
    assert len(cams) == 6

    out_mesh, alb_tex, mr_tex, _av, _mv = N.Hunyuan3DBakeMultiView.execute(
        mesh, albedo, mr, cams, 256, 4.0, True)
    assert out_mesh.uvs is not None and out_mesh.texture is not None
    assert out_mesh.texture_mr is not None
    assert alb_tex.shape == (1, 256, 256, 3)


def test_save_glb_without_mr_still_single_texture(tmp_path):
    from PIL import Image
    from comfy_extras.nodes_save_3d import save_glb
    v, f = _cube()
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    base = Image.new("RGB", (16, 16), (128, 128, 128))
    path = str(tmp_path / "out.glb")
    save_glb(nv, nf, path, uvs=uv, texture_image=base)
    with open(path, "rb") as fh:
        data = fh.read()
    jlen = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + jlen])
    assert len(gltf["images"]) == 1
    pbr = gltf["materials"][0]["pbrMetallicRoughness"]
    assert "metallicRoughnessTexture" not in pbr

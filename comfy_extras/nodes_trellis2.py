from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, Types
import torch
from comfy.ldm.trellis2.model import SparseTensor
import comfy.model_management
import comfy.model_patcher

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

dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
dino_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def smart_crop_square(image, mask, margin_ratio=0.1, bg_color=(128, 128, 128)):
    nz = torch.nonzero(mask[0] > 0.5)
    if nz.shape[0] == 0:
        C, H, W = image.shape
        side = max(H, W)
        canvas = torch.full((C, side, side), 0.5, device=image.device) # Gray
        canvas[:, (side-H)//2:(side-H)//2+H, (side-W)//2:(side-W)//2+W] = image
        return canvas

    y_min, x_min = nz.min(dim=0)[0]
    y_max, x_max = nz.max(dim=0)[0]

    obj_w, obj_h = x_max - x_min, y_max - y_min
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2

    side = int(max(obj_w, obj_h) * (1 + margin_ratio * 2))
    half_side = side / 2

    x1, y1 = int(center_x - half_side), int(center_y - half_side)
    x2, y2 = x1 + side, y1 + side

    C, H, W = image.shape
    canvas = torch.ones((C, side, side), device=image.device)
    for c in range(C):
        canvas[c] *= (bg_color[c] / 255.0)

    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(W, x2), min(H, y2)

    dst_x1, dst_y1 = max(0, -x1), max(0, -y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    canvas[:, dst_y1:dst_y2, dst_x1:dst_x2] = image[:, src_y1:src_y2, src_x1:src_x2]

    return canvas

def run_conditioning(model, image, mask, include_1024 = True, background_color = "black"):
    model_internal = model.model
    device = comfy.model_management.intermediate_device()
    torch_device = comfy.model_management.get_torch_device()

    bg_colors = {"black": (0, 0, 0), "gray": (128, 128, 128), "white": (255, 255, 255)}
    bg_rgb = bg_colors.get(background_color, (128, 128, 128))

    img_t = image[0].movedim(-1, 0).to(torch_device).float()
    mask_t = mask[0].to(torch_device).float()
    if mask_t.ndim == 2:
        mask_t = mask_t.unsqueeze(0)

    cropped_img = smart_crop_square(img_t, mask_t, bg_color=bg_rgb)

    def prepare_tensor(img, size):
        resized = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(size, size), mode='bicubic', align_corners=False
        )
        return (resized - dino_mean.to(torch_device)) / dino_std.to(torch_device)

    model_internal.image_size = 512
    input_512 = prepare_tensor(cropped_img, 512)
    cond_512 = model_internal(input_512)[0]

    cond_1024 = None
    if include_1024:
        model_internal.image_size = 1024
        input_1024 = prepare_tensor(cropped_img, 1024)
        cond_1024 = model_internal(input_1024)[0]

    conditioning = {
        'cond_512': cond_512.to(device),
        'neg_cond': torch.zeros_like(cond_512).to(device),
    }
    if cond_1024 is not None:
        conditioning['cond_1024'] = cond_1024.to(device)

    preprocessed_tensor = cropped_img.movedim(0, -1).unsqueeze(0).cpu()

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
        vae = vae.first_stage_model
        samples = samples["samples"]
        std = shape_slat_normalization["std"]
        mean = shape_slat_normalization["mean"]
        samples = samples * std + mean

        mesh, subs = vae.decode_shape_slat(resolution, samples)
        return IO.NodeOutput(mesh, subs)

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
        vae = vae.first_stage_model
        samples = samples["samples"]
        std = tex_slat_normalization["std"]
        mean = tex_slat_normalization["mean"]
        samples = samples * std + mean

        mesh = vae.decode_tex_slat(samples, shape_subs)
        return IO.NodeOutput(mesh)

class VaeDecodeStructureTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VaeDecodeStructureTrellis2",
            category="latent/3d",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
            ],
            outputs=[
                IO.Voxel.Output("structure_output"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae):
        vae = vae.first_stage_model
        decoder = vae.struct_dec
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.vae_offload_device()
        decoder = decoder.to(load_device)
        samples = samples["samples"]
        samples = samples.to(load_device)
        decoded = decoder(samples)>0
        decoder.to(offload_device)
        comfy.model_management.get_offload_stream
        out = Types.VOXEL(decoded.squeeze(1).float())
        return IO.NodeOutput(out)

class Trellis2Conditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2Conditioning",
            category="conditioning/video_models",
            inputs=[
                IO.ClipVision.Input("clip_vision_model"),
                IO.Image.Input("image"),
                IO.Mask.Input("mask"),
                IO.Combo.Input("background_color", options=["black", "gray", "white"], default="black")
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, clip_vision_model, image, mask, background_color) -> IO.NodeOutput:
        # could make 1024 an option
        conditioning, _ = run_conditioning(clip_vision_model, image, mask, include_1024=True, background_color=background_color)
        embeds = conditioning["cond_1024"] # should add that
        positive = [[conditioning["cond_512"], {"embeds": embeds}]]
        negative = [[conditioning["neg_cond"], {"embeds": embeds}]]
        return IO.NodeOutput(positive, negative)

class EmptyShapeLatentTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyShapeLatentTrellis2",
            category="latent/3d",
            inputs=[
                IO.Voxel.Input("structure_output"),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, structure_output):
        decoded = structure_output.data
        coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int().unsqueeze(1)
        in_channels = 32
        latent = SparseTensor(feats=torch.randn(coords.shape[0], in_channels), coords=coords)
        return IO.NodeOutput({"samples": latent, "type": "trellis2", "generation_mode": "shape_generation"})

class EmptyTextureLatentTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyTextureLatentTrellis2",
            category="latent/3d",
            inputs=[
                IO.Voxel.Input("structure_output"),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, structure_output):
        # TODO
        in_channels = 32
        latent = structure_output.replace(feats=torch.randn(structure_output.data.shape[0], in_channels - structure_output.feats.shape[1]))
        return IO.NodeOutput({"samples": latent, "type": "trellis2", "generation_mode": "texture_generation"})

class EmptyStructureLatentTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyStructureLatentTrellis2",
            category="latent/3d",
            inputs=[
                IO.Int.Input("batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )
    @classmethod
    def execute(cls, batch_size):
        in_channels = 8
        resolution = 16
        latent = torch.randn(batch_size, in_channels, resolution, resolution, resolution)
        return IO.NodeOutput({"samples": latent, "type": "trellis2", "generation_mode": "structure_generation"})

def simplify_fn(vertices, faces, target=100000):

    if vertices.shape[0] <= target:
        return vertices, faces

    min_feat = vertices.min(dim=0)[0]
    max_feat = vertices.max(dim=0)[0]
    extent = (max_feat - min_feat).max()

    grid_resolution = int(torch.sqrt(torch.tensor(target)).item() * 1.5)
    voxel_size = extent / grid_resolution

    quantized_coords = ((vertices - min_feat) / voxel_size).long()

    unique_coords, inverse_indices = torch.unique(quantized_coords, dim=0, return_inverse=True)

    num_new_verts = unique_coords.shape[0]
    new_vertices = torch.zeros((num_new_verts, 3), dtype=vertices.dtype, device=vertices.device)

    counts = torch.zeros((num_new_verts, 1), dtype=vertices.dtype, device=vertices.device)

    new_vertices.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), vertices)
    counts.scatter_add_(0, inverse_indices.unsqueeze(1), torch.ones_like(vertices[:, :1]))

    new_vertices = new_vertices / counts.clamp(min=1)

    new_faces = inverse_indices[faces]

    v0 = new_faces[:, 0]
    v1 = new_faces[:, 1]
    v2 = new_faces[:, 2]

    valid_mask = (v0 != v1) & (v1 != v2) & (v2 != v0)
    new_faces = new_faces[valid_mask]

    unique_face_indices, inv_face = torch.unique(new_faces.reshape(-1), return_inverse=True)
    final_vertices = new_vertices[unique_face_indices]
    final_faces = inv_face.reshape(-1, 3)

    return final_vertices, final_faces

def fill_holes_fn(vertices, faces, max_hole_perimeter=3e-2):
    is_batched = vertices.ndim == 3
    if is_batched:
        batch_size = vertices.shape[0]
        if batch_size > 1:
            v_out, f_out = [], []
            for i in range(batch_size):
                v, f = fill_holes_fn(vertices[i], faces[i], max_hole_perimeter)
                v_out.append(v)
                f_out.append(f)
            return torch.stack(v_out), torch.stack(f_out)

        vertices = vertices.squeeze(0)
        faces = faces.squeeze(0)

    device = vertices.device
    orig_vertices = vertices
    orig_faces = faces

    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)

    edges_sorted, _ = torch.sort(edges, dim=1)
    unique_edges, counts = torch.unique(edges_sorted, dim=0, return_counts=True)
    boundary_mask = counts == 1
    boundary_edges_sorted = unique_edges[boundary_mask]

    if boundary_edges_sorted.shape[0] == 0:
        if is_batched:
            return orig_vertices.unsqueeze(0), orig_faces.unsqueeze(0)
        return orig_vertices, orig_faces

    max_idx = vertices.shape[0]

    packed_edges_all = torch.sort(edges, dim=1).values
    packed_edges_all = packed_edges_all[:, 0] * max_idx + packed_edges_all[:, 1]

    packed_boundary = boundary_edges_sorted[:, 0] * max_idx + boundary_edges_sorted[:, 1]

    is_boundary_edge = torch.isin(packed_edges_all, packed_boundary)
    active_boundary_edges = edges[is_boundary_edge]

    adj = {}
    edges_np = active_boundary_edges.cpu().numpy()
    for u, v in edges_np:
        adj[u] = v

    loops = []
    visited_edges = set()
    processed_nodes = set()
    for start_node in list(adj.keys()):
        if start_node in processed_nodes: continue
        current_loop, curr = [], start_node
        while curr in adj:
            next_node = adj[curr]
            if (curr, next_node) in visited_edges: break
            visited_edges.add((curr, next_node))
            processed_nodes.add(curr)
            current_loop.append(curr)
            curr = next_node
            if curr == start_node:
                loops.append(current_loop)
                break
            if len(current_loop) > len(edges_np): break

    if not loops:
        if is_batched: return orig_vertices.unsqueeze(0), orig_faces.unsqueeze(0)
        return orig_vertices, orig_faces

    new_faces = []
    v_offset = vertices.shape[0]
    valid_new_verts = []

    for loop_indices in loops:
        if len(loop_indices) < 3: continue
        loop_tensor = torch.tensor(loop_indices, dtype=torch.long, device=device)
        loop_verts = vertices[loop_tensor]
        diffs = loop_verts - torch.roll(loop_verts, shifts=-1, dims=0)
        perimeter = torch.norm(diffs, dim=1).sum()

        if perimeter > max_hole_perimeter: continue

        center = loop_verts.mean(dim=0)
        valid_new_verts.append(center)
        c_idx = v_offset
        v_offset += 1

        num_v = len(loop_indices)
        for i in range(num_v):
            v_curr, v_next = loop_indices[i], loop_indices[(i + 1) % num_v]
            new_faces.append([v_curr, v_next, c_idx])

    if len(valid_new_verts) > 0:
        added_vertices = torch.stack(valid_new_verts, dim=0)
        added_faces = torch.tensor(new_faces, dtype=torch.long, device=device)
        vertices = torch.cat([orig_vertices, added_vertices], dim=0)
        faces = torch.cat([orig_faces, added_faces], dim=0)
    else:
        vertices, faces = orig_vertices, orig_faces

    if is_batched:
        return vertices.unsqueeze(0), faces.unsqueeze(0)

    return vertices, faces

class PostProcessMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PostProcessMesh",
            category="latent/3d",
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Int.Input("simplify", default=100_000, min=0, max=50_000_000), # max?
                IO.Float.Input("fill_holes_perimeter", default=0.003, min=0.0, step=0.0001)
            ],
            outputs=[
                IO.Mesh.Output("output_mesh"),
            ]
        )
    @classmethod
    def execute(cls, mesh, simplify, fill_holes_perimeter):
        verts, faces = mesh.vertices, mesh.faces

        if fill_holes_perimeter != 0.0:
            verts, faces = fill_holes_fn(verts, faces, max_hole_perimeter=fill_holes_perimeter)

        if simplify != 0:
            verts, faces = simplify_fn(verts, faces, simplify)


        mesh.vertices = verts
        mesh.faces = faces

        return IO.NodeOutput(mesh)

class Trellis2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Trellis2Conditioning,
            EmptyShapeLatentTrellis2,
            EmptyStructureLatentTrellis2,
            EmptyTextureLatentTrellis2,
            VaeDecodeTextureTrellis,
            VaeDecodeShapeTrellis,
            VaeDecodeStructureTrellis2,
            PostProcessMesh
        ]


async def comfy_entrypoint() -> Trellis2Extension:
    return Trellis2Extension()

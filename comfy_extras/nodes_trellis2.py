from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, Types, UI, io
from comfy.ldm.trellis2.vae import SparseTensor
from comfy.ldm.trellis2.model import build_proj_transform_matrix, compute_stage_proj_feats

from comfy_extras.nodes_mesh_postprocess import pack_variable_mesh_batch
from server import PromptServer
import comfy.latent_formats
import comfy.model_management
import comfy.utils
from PIL import Image
import logging
import numpy as np
import math
import torch

ShapeSubdivides = io.Custom("SHAPE_SUBDIVIDES")


shape_slat_format = comfy.latent_formats.Trellis2ShapeSLAT()
tex_slat_format = comfy.latent_formats.Trellis2TexSLAT()

def shape_norm(shape_latent, coords):
    feats = shape_slat_format.process_out(shape_latent)
    return SparseTensor(feats=feats, coords=coords)


def infer_batched_coord_layout(coords):
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected Trellis2 coords with shape [N, 4], got {tuple(coords.shape)}")

    if coords.shape[0] == 0:
        raise ValueError("Trellis2 coords can't be empty")

    batch_ids = coords[:, 0].to(torch.int64)
    if (batch_ids < 0).any():
        raise ValueError(f"Trellis2 batch ids must be non-negative, got {batch_ids.unique(sorted=True).tolist()}")
    batch_size = int(batch_ids.max().item()) + 1
    counts = torch.bincount(batch_ids, minlength=batch_size)

    if (counts == 0).any():
        raise ValueError(f"Non-contiguous Trellis2 batch ids in coords: {batch_ids.unique(sorted=True).tolist()}")

    max_tokens = int(counts.max().item())
    return batch_size, counts, max_tokens


def split_batched_coords(coords, coord_counts):
    if coord_counts.ndim != 1:
        raise ValueError(f"Trellis2 coord_counts must be 1D, got shape {tuple(coord_counts.shape)}")
    if (coord_counts < 0).any():
        raise ValueError(f"Trellis2 coord_counts must be non-negative, got {coord_counts.tolist()}")
    if int(coord_counts.sum().item()) != coords.shape[0]:
        raise ValueError(
            f"Trellis2 coord_counts total {int(coord_counts.sum().item())} does not match coords rows {coords.shape[0]}"
        )

    batch_ids = coords[:, 0].to(torch.int64)
    order = torch.argsort(batch_ids, stable=True)
    sorted_coords = coords.index_select(0, order)
    sorted_batch_ids = batch_ids.index_select(0, order)

    offsets = coord_counts.cumsum(0) - coord_counts
    items = []
    for i in range(coord_counts.shape[0]):
        count = int(coord_counts[i].item())
        start = int(offsets[i].item())
        coords_i = sorted_coords[start:start + count]
        ids_i = sorted_batch_ids[start:start + count]
        if coords_i.shape[0] != count or not torch.all(ids_i == i):
            raise ValueError(f"Trellis2 coords rows for batch {i} expected {count}, got {coords_i.shape[0]}")
        items.append(coords_i)
    return items

def flatten_batched_sparse_latent(samples, coords, coord_counts):
    samples = samples.squeeze(-1).transpose(1, 2)
    if coord_counts is None:
        return samples.reshape(-1, samples.shape[-1]), coords

    coords_items = split_batched_coords(coords, coord_counts)
    feat_list = []
    coord_list = []
    for i, coords_i in enumerate(coords_items):
        count = int(coord_counts[i].item())
        feat_list.append(samples[i, :count])
        coord_list.append(coords_i)

    return torch.cat(feat_list, dim=0), torch.cat(coord_list, dim=0)


def split_batched_sparse_latent(samples, coords, coord_counts):
    samples = samples.squeeze(-1).transpose(1, 2)
    if coord_counts is None:
        return [(samples.reshape(-1, samples.shape[-1]), coords)]

    coords_items = split_batched_coords(coords, coord_counts)
    items = []
    for i, coords_i in enumerate(coords_items):
        count = int(coord_counts[i].item())
        items.append((samples[i, :count], coords_i))
    return items

class VaeDecodeShapeTrellis(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VaeDecodeShapeTrellis",
            category="latent/3d",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
            ],
            outputs=[
                IO.Mesh.Output("mesh"),
                ShapeSubdivides.Output(display_name = "shape_subdivides"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae):
        # Mesh grid_size must match the actual coord resolution the upstream
        # stage was run at (1024 cascade -> 64, 1536 cascade -> 96). The VAE's
        # built-in `.resolution` buffer defaults to 1024 and is otherwise stale;
        # take coord_resolution from the latent dict if the stage node set it.
        coord_resolution = samples.get("coord_resolution")
        if coord_resolution is not None:
            resolution = int(coord_resolution) * 16
        else:
            resolution = int(vae.first_stage_model.resolution.item())
        model_frame = samples.get("model_frame", "y_up")
        sample_tensor = samples["samples"]
        device = comfy.model_management.get_torch_device()
        coords = samples["coords"]
        vae.prepare_decode(sample_tensor.shape)
        trellis_vae = vae.first_stage_model
        coord_counts = samples.get("coord_counts")

        samples = samples["samples"]
        if coord_counts is None:
            samples, coords = flatten_batched_sparse_latent(samples, coords, coord_counts)
            samples = shape_norm(samples.to(device), coords.to(device))
            mesh, subs = trellis_vae.decode_shape_slat(samples.to(vae.vae_dtype), resolution)
        else:
            split_items = split_batched_sparse_latent(samples, coords, coord_counts)
            mesh = []
            subs_per_sample = []
            for feats_i, coords_i in split_items:
                coords_i = coords_i.to(device).clone()
                coords_i[:, 0] = 0
                sample_i = shape_norm(feats_i.to(device), coords_i)
                mesh_i, subs_i = trellis_vae.decode_shape_slat(sample_i.to(vae.vae_dtype), resolution)
                mesh.append(mesh_i[0])
                subs_per_sample.append(subs_i)

            subs = []
            for stage_index in range(len(subs_per_sample[0])):
                stage_tensors = [sample_subs[stage_index] for sample_subs in subs_per_sample]
                feats_list = [stage_tensor.feats for stage_tensor in stage_tensors]
                coords_list = [stage_tensor.coords for stage_tensor in stage_tensors]
                subs.append(SparseTensor.from_tensor_list(feats_list, coords_list))

        # Rotate Z-up (Trellis2 training frame) vertices to glTF Y-up. Pixal3D outputs are already Y-up.
        if model_frame == "z_up":
            vert_list = [torch.stack([v[..., 0], v[..., 2], -v[..., 1]], dim=-1).float().cpu()
                         for v, _ in mesh]
        else:
            vert_list = [v.float().cpu() for v, _ in mesh]
        face_list = [f.int().cpu() for _, f in mesh]
        if all(v.shape == vert_list[0].shape for v in vert_list) and all(f.shape == face_list[0].shape for f in face_list):
            mesh = Types.MESH(vertices=torch.stack(vert_list), faces=torch.stack(face_list))
        else:
            mesh = pack_variable_mesh_batch(vert_list, face_list)
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
                ShapeSubdivides.Input("shape_subdivides",
                                 tooltip=(
                                    "Shape information used to guide higher-detail reconstruction during decoding. "
                                    "Helps preserve structure consistency at higher resolutions."
                                 )),
            ],
            outputs=[
                IO.Voxel.Output("voxel_colors"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae, shape_subdivides):
        sample_tensor = samples["samples"]
        device = comfy.model_management.get_torch_device()
        coords = samples["coords"]
        vae.prepare_decode(sample_tensor.shape)
        trellis_vae = vae.first_stage_model
        coord_counts = samples.get("coord_counts")
        model_frame = samples.get("model_frame", "y_up")
        coord_resolution = samples.get("coord_resolution")

        samples = samples["samples"]
        samples, coords = flatten_batched_sparse_latent(samples, coords, coord_counts)
        samples = samples.to(device)
        feats = tex_slat_format.process_out(samples)
        samples = SparseTensor(feats=feats, coords=coords.to(device))

        voxel = trellis_vae.decode_tex_slat(samples.to(vae.vae_dtype), shape_subdivides)
        # Keep all decoded channels. The texture VAE emits 6: base_color (0:3),
        # metallic (3), roughness (4), alpha (5) — all in [0, 1]. Vertex-color
        # consumers (PaintMesh) slice [:3]
        color_feats = voxel.feats
        voxel_coords = voxel.coords

        if coord_resolution is not None:
            tex_resolution = int(coord_resolution) * 16
        elif voxel_coords.numel() > 0 and voxel_coords.shape[-1] >= 3:
            spatial = voxel_coords[:, -3:] if voxel_coords.shape[-1] == 4 else voxel_coords
            max_idx = int(spatial.max().item()) + 1
            tex_resolution = next((r for r in (256, 512, 1024, 1536, 2048) if r >= max_idx), max_idx)
        else:
            tex_resolution = 1024

        # Remap Z-up voxel coords to Y-up: (x, y, z) -> (x, z, R-1-y), matching the
        # R_x(-90°) applied to mesh vertices in VaeDecodeShapeTrellis. Keeps PaintMesh's
        # NN lookup correctly aligned without it needing to know the source frame.
        if model_frame == "z_up" and voxel_coords.numel() > 0 and voxel_coords.shape[-1] >= 3:
            R = tex_resolution
            if voxel_coords.shape[-1] == 4:
                batch_col = voxel_coords[:, :1]
                spatial = voxel_coords[:, 1:]
                spatial_yup = torch.stack(
                    [spatial[:, 0], spatial[:, 2], (R - 1) - spatial[:, 1]], dim=-1
                )
                voxel_coords = torch.cat([batch_col, spatial_yup], dim=-1)
            else:
                voxel_coords = torch.stack(
                    [voxel_coords[:, 0], voxel_coords[:, 2], (R - 1) - voxel_coords[:, 1]],
                    dim=-1,
                )

        voxel = Types.VOXEL(voxel_coords, color_feats, tex_resolution)
        return IO.NodeOutput(voxel)

class VaeDecodeStructureTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VaeDecodeStructureTrellis2",
            category="latent/3d",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
                IO.Combo.Input("resolution", options=["32", "64"], default="32"),
            ],
            outputs=[
                IO.Voxel.Output("voxel"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae, resolution):
        resolution = int(resolution)
        sample_tensor = samples["samples"]
        sample_tensor = sample_tensor[:, :8]
        batch_number = vae.prepare_decode(sample_tensor.shape)
        shape_vae = vae.first_stage_model
        load_device = comfy.model_management.get_torch_device()
        decoded_batches = []
        for start in range(0, sample_tensor.shape[0], batch_number):
            sample_chunk = sample_tensor[start:start + batch_number].to(load_device)
            decoded_batches.append(shape_vae.decode_structure(sample_chunk.to(vae.vae_dtype)) > 0)
        decoded = torch.cat(decoded_batches, dim=0)
        current_res = decoded.shape[2]

        if current_res != resolution:
            ratio = current_res // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        voxel_data = decoded.squeeze(1).float()
        return IO.NodeOutput(Types.VOXEL(voxel_data))

class Trellis2UpsampleStage(IO.ComfyNode):
    """Cascade-upsamples a 512-resolution shape latent into high-resolution
    sparse coords and sets up the second shape-stage sampling pass at the
    target resolution, attaching per-stage metadata to the conditioning for
    the model to consume via extra_conds. target_resolution is reduced in
    128-step decrements until the unique upsampled coord count fits under
    max_tokens (floor 1024)."""
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2UpsampleStage",
            category="latent/3d",
            display_name="Trellis2 Upsample Stage",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Latent.Input("shape_latent", tooltip="The 512-resolution shape latent output from the first shape-stage KSampler."),
                IO.Vae.Input("vae"),
                IO.Combo.Input("target_resolution", options=["1024", "1536"], default="1024", tooltip="Controls output detail level for upsampling."),
                IO.Int.Input("max_tokens", default=49152, min=1024, max=100000,
                             tooltip=(
                                "Maximum number of output elements (coordinates) allowed after upsampling. "
                                "Used to limit memory usage and control mesh density."
                            )),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                IO.Latent.Output(),
            ]
        )

    @staticmethod
    def _quantize_unique(hr_coords: torch.Tensor, lr_resolution: int, hr_resolution: int, pixal3d_mode: bool = False) -> torch.Tensor:
        # Trellis2 uses `floor((c+0.5) * grid_res / lr_res)
        # Pixal3D uses `round((c+0.5) * (grid_res-1) / lr_res)`
        # this is a half-cell spatial shift. Branch so each upstream is matched bit-for-bit.
        grid_res = hr_resolution // 16
        spatial = hr_coords[:, 1:].float()
        if pixal3d_mode:
            spatial.add_(0.5).mul_((grid_res - 1) / lr_resolution).round_()
        else:
            spatial.add_(0.5).mul_(grid_res / lr_resolution)
        quant = torch.cat([hr_coords[:, :1], spatial.int()], dim=1)
        return quant.unique(dim=0)

    @classmethod
    def execute(cls, positive, negative, shape_latent, vae, target_resolution, max_tokens):
        device = comfy.model_management.get_torch_device()
        vae.prepare_decode(shape_latent["samples"].shape)

        coord_counts = shape_latent.get("coord_counts")
        shape_vae = vae.first_stage_model
        lr_resolution = 512
        target_resolution = int(target_resolution)
        proj_pack = _proj_pack_from_conditioning(positive)
        pixal3d_mode = proj_pack is not None

        # Decode each sample's HR coords, then search for the largest hr_resolution
        # that fits under max_tokens across all samples.
        if coord_counts is None:
            feats, coords_512 = flatten_batched_sparse_latent(
                shape_latent["samples"], shape_latent["coords"], coord_counts,
            )
            slat = shape_norm(feats.to(device), coords_512.to(device))
            sample_hr_coords = [shape_vae.upsample_shape(slat.to(vae.vae_dtype), upsample_times=4)]
        else:
            items = split_batched_sparse_latent(
                shape_latent["samples"], shape_latent["coords"], coord_counts,
            )
            sample_hr_coords = []
            for feats_i, coords_i in items:
                coords_i = coords_i.to(device).clone()
                coords_i[:, 0] = 0
                slat_i = shape_norm(feats_i.to(device), coords_i)
                sample_hr_coords.append(shape_vae.upsample_shape(slat_i.to(vae.vae_dtype), upsample_times=4))

        # Resolution search — cache the final iteration's quantized unique tensors
        # so we don't recompute .unique() per sample after picking hr_resolution.
        hr_resolution = target_resolution
        quant_unique_list = []
        while True:
            quant_unique_list = []
            exceeds_limit = False
            for hr_coords_i in sample_hr_coords:
                qu = cls._quantize_unique(hr_coords_i, lr_resolution, hr_resolution, pixal3d_mode)
                quant_unique_list.append(qu)
                if qu.shape[0] >= max_tokens:
                    exceeds_limit = True
                    break
            if not exceeds_limit:
                break
            if hr_resolution <= 1024:
                for k in range(len(quant_unique_list), len(sample_hr_coords)):
                    quant_unique_list.append(
                        cls._quantize_unique(sample_hr_coords[k], lr_resolution, hr_resolution, pixal3d_mode)
                    )
                break
            hr_resolution -= 128

        # Rewrite batch column to match per-sample offset and concat.
        per_sample_counts = []
        for sample_offset, qu in enumerate(quant_unique_list):
            qu[:, 0] = sample_offset
            per_sample_counts.append(int(qu.shape[0]))
        coords = torch.cat(quant_unique_list, dim=0)
        counts = torch.tensor(per_sample_counts, dtype=torch.int64)
        coord_resolution = hr_resolution // 16

        batch_size, _, max_tokens_out = infer_batched_coord_layout(coords)
        latent = torch.zeros(batch_size, 32, max_tokens_out, 1)

        extras = {
            "trellis2_generation_mode": "shape_generation",
            "trellis2_coords": coords,
            "trellis2_coord_counts": counts,
        }
        if proj_pack is not None:
            extras["trellis2_proj_feats"] = compute_stage_proj_feats(
                proj_pack, "shape_1024", coords=coords, coord_resolution=coord_resolution,
            )
        positive_out = _conditioning_set_extras(positive, extras)
        negative_out = _conditioning_set_extras(negative, extras)
        out_latent = {"samples": latent, "coords": coords, "coord_counts": counts,
                      "coord_resolution": coord_resolution, "type": "trellis2",
                      "model_frame": shape_latent.get("model_frame",
                                                       "y_up" if proj_pack is not None else "z_up")}
        return IO.NodeOutput(positive_out, negative_out, out_latent)

def _dinov3_encode(model, image_bchw, image_size, want_patches=False):
    """Run DINOv3 once at the requested resolution.

    image_bchw: [B, 3, H, W] float in [0, 1] (any source resolution; resized here).
    Returns the full sequence tensor (Trellis2 path) or a dict with the global
    tokens split out + a 2D patch grid (Pixal3D path) when `want_patches=True`.
    """
    model_internal = model.model
    device = comfy.model_management.get_torch_device()
    img_t = comfy.utils.common_upscale(image_bchw, image_size, image_size, "lanczos", "disabled").to(device)
    mean = torch.tensor(model.image_mean or [0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor(model.image_std or [0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_t = (img_t - mean) / std
    model_internal.image_size = image_size
    tokens = model_internal(img_t, skip_norm_elementwise=True)[0]
    if not want_patches:
        return tokens
    h_p = w_p = image_size // 16
    n_reg = tokens.shape[1] - 1 - h_p * w_p
    return {"tokens": tokens[:, :1 + n_reg], "patches_2d": _dinov3_patches_to_2d(tokens, image_size)}


def run_conditioning(model, cropped_pil_img, include_1024=True):
    device = comfy.model_management.intermediate_device()

    img_np = np.array(cropped_pil_img).astype(np.float32) / 255.0
    image_bchw = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).contiguous()

    cond_512 = _dinov3_encode(model, image_bchw, 512)
    conditioning = {
        "cond_512": cond_512.to(device),
        "neg_cond": torch.zeros_like(cond_512).to(device),
    }
    if include_1024:
        cond_1024 = _dinov3_encode(model, image_bchw, 1024)
        conditioning["cond_1024"] = cond_1024.to(device)
    return conditioning

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
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, clip_vision_model, image, mask) -> IO.NodeOutput:
        # Normalize to batched form so per-image conditioning loop below is uniform.
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim == 4:
            if image.shape[1] in [1, 3, 4] and image.shape[-1] not in [1, 3, 4]:
                image = image.permute(0, 2, 3, 1)

        # normalize mask to standard [B, H, W] (handling 2D, 3D, and 4D variants)
        if mask.ndim == 4:
            if mask.shape[1] == 1:
                mask = mask.squeeze(1)
            elif mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            else:
                mask = mask[:, :, :, 0] # take first channel as fallback

        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask.squeeze(-1).unsqueeze(0)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)

        batch_size = image.shape[0]
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        elif mask.shape[0] != batch_size:
            raise ValueError(f"Trellis2Conditioning mask batch {mask.shape[0]} does not match image batch {batch_size}")

        cond_512_list = []
        cond_1024_list = []

        for b in range(batch_size):
            item_image = image[b]
            item_mask = mask[b] if mask.size(0) > 1 else mask[0]

            img_np = (item_image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            mask_np = (item_mask.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            # Ensure img_np is either 2D (grayscale) or 3D (RGB/RGBA)
            if img_np.ndim == 3 and img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)

            mask_np = mask_np.squeeze()

            # detect inverted mask
            border_pixels = np.concatenate([
                mask_np[0, :], mask_np[-1, :], mask_np[:, 0], mask_np[:, -1]
            ])
            if np.mean(border_pixels) > 127:
                mask_np = 255 - mask_np

            mask_np[mask_np < 35] = 0

            border_shave = 4
            mask_np[:border_shave, :] = 0
            mask_np[-border_shave:, :] = 0
            mask_np[:, :border_shave] = 0
            mask_np[:, -border_shave:] = 0

            pil_img = Image.fromarray(img_np)
            pil_mask = Image.fromarray(mask_np)

            max_size = max(pil_img.size)
            scale = min(1.0, 1024 / max_size)
            if scale < 1.0:
                new_w, new_h = int(pil_img.width * scale), int(pil_img.height * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                pil_mask = pil_mask.resize((new_w, new_h), Image.Resampling.NEAREST)

            rgba_np = np.zeros((pil_img.height, pil_img.width, 4), dtype=np.uint8)
            rgba_np[:, :, :3] = np.array(pil_img.convert("RGB"))
            rgba_np[:, :, 3] = np.array(pil_mask)

            alpha = rgba_np[:, :, 3]
            bbox_coords = np.argwhere(alpha > 0.8 * 255)

            if len(bbox_coords) > 0:
                y_min, x_min = np.min(bbox_coords[:, 0]), np.min(bbox_coords[:, 1])
                y_max, x_max = np.max(bbox_coords[:, 0]), np.max(bbox_coords[:, 1])

                center_y, center_x = (y_min + y_max) / 2.0, (x_min + x_max) / 2.0
                size = max(y_max - y_min, x_max - x_min)

                crop_x1 = int(center_x - size // 2)
                crop_y1 = int(center_y - size // 2)
                crop_x2 = int(center_x + size // 2)
                crop_y2 = int(center_y + size // 2)

                rgba_pil = Image.fromarray(rgba_np)
                cropped_rgba = rgba_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                cropped_np = np.array(cropped_rgba).astype(np.float32) / 255.0
            else:
                logging.warning("Mask for the image is empty. Trellis2 requires an image with a mask for the best mesh quality.")
                cropped_np = rgba_np.astype(np.float32) / 255.0

            bg_rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            fg = cropped_np[:, :, :3]
            alpha_float = cropped_np[:, :, 3:4]
            composite_np = fg * alpha_float + bg_rgb * (1.0 - alpha_float)

            # Keep the image as 4-channel RGBA to force TRELLIS to bypass its internal background remover
            rgb_uint8 = (composite_np * 255.0).round().clip(0, 255).astype(np.uint8)
            alpha_uint8 = (alpha_float.squeeze(-1) * 255.0).round().clip(0, 255).astype(np.uint8)

            rgba_composite = np.zeros((cropped_np.shape[0], cropped_np.shape[1], 4), dtype=np.uint8)
            rgba_composite[:, :, :3] = rgb_uint8
            rgba_composite[:, :, 3] = alpha_uint8

            cropped_pil = Image.fromarray(rgba_composite, mode="RGBA")

            # Convert to RGB to ensure the CLIP/DINO model receives a 3-channel image
            item_conditioning = run_conditioning(clip_vision_model, cropped_pil.convert("RGB"), include_1024=True)
            cond_512_list.append(item_conditioning["cond_512"])
            cond_1024_list.append(item_conditioning["cond_1024"])

        cond_512_batched = torch.cat(cond_512_list, dim=0)
        cond_1024_batched = torch.cat(cond_1024_list, dim=0)
        neg_cond_batched = torch.zeros_like(cond_512_batched)
        neg_embeds_batched = torch.zeros_like(cond_1024_batched)

        positive = [[cond_512_batched, {"embeds": cond_1024_batched}]]
        negative = [[neg_cond_batched, {"embeds": neg_embeds_batched}]]
        return IO.NodeOutput(positive, negative)

def _proj_pack_from_conditioning(conditioning):
    """Return the proj_feat_pack dict embedded in a Pixal3D conditioning (or None
    for vanilla Trellis2 / no conditioning connected). Pixal3DConditioning ships
    the pack in cond[0][1]["proj_feat_pack"]; Trellis2Conditioning doesn't set it."""
    if not conditioning:
        return None
    entry = conditioning[0]
    if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
        return None
    return entry[1].get("proj_feat_pack")


def _conditioning_set_extras(conditioning, extras: dict):
    """Return a copy of `conditioning` with `extras` merged into each entry's
    dict — same shallow-copy pattern ControlNetApplyAdvanced uses. The dicts
    are copied so we don't mutate upstream conditioning."""
    out = []
    for entry in conditioning:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            new_dict = entry[1].copy()
            new_dict.update(extras)
            out.append([entry[0], new_dict])
        else:
            out.append(entry)
    return out


class Trellis2ShapeStage(IO.ComfyNode):
    """Sets up the first shape-stage sampling pass: extracts sparse coords from
    the dense structure voxel produced by VaeDecodeStructureTrellis2, builds an
    empty sparse latent, and attaches per-stage metadata to the conditioning so
    the model reads it via extra_conds at sample time. For the second shape pass
    (post-upsample), use Trellis2UpsampleStage instead — it combines the cascade
    and the second-pass stage setup."""
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2ShapeStage",
            category="latent/3d",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Voxel.Input(
                    "voxel",
                    tooltip="Dense structure voxel from VaeDecodeStructureTrellis2.",
                ),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, positive, negative, voxel):
        decoded = voxel.data.unsqueeze(1)
        coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int()
        coord_resolution = int(decoded.shape[-1])

        # Dispatch based on the upstream voxel resolution, mirroring upstream's
        # pipeline_type → ss_res table:
        #   coord_res == 32 → first cascade shape pass OR pure-512 pipeline
        #                     (img2shape_512 + shape_512 proj stage, 512 DINO).
        #   coord_res >  32 → pure-1024 non-cascade pipeline
        #                     (img2shape + shape_1024 proj stage, 1024 DINO).
        if coord_resolution <= 32:
            mode = "shape_generation_512"
            stage = "shape_512"
        else:
            mode = "shape_generation"
            stage = "shape_1024"

        batch_size, counts, max_tokens = infer_batched_coord_layout(coords)
        latent = torch.zeros(batch_size, 32, max_tokens, 1)

        extras = {
            "trellis2_generation_mode": mode,
            "trellis2_coords": coords,
            "trellis2_coord_counts": counts,
        }
        proj_pack = _proj_pack_from_conditioning(positive)
        if proj_pack is not None:
            extras["trellis2_proj_feats"] = compute_stage_proj_feats(
                proj_pack, stage, coords=coords, coord_resolution=coord_resolution,
            )
        positive_out = _conditioning_set_extras(positive, extras)
        negative_out = _conditioning_set_extras(negative, extras)
        out_latent = {"samples": latent, "coords": coords, "coord_counts": counts,
                      "coord_resolution": coord_resolution, "type": "trellis2",
                      "model_frame": "y_up" if proj_pack is not None else "z_up"}
        return IO.NodeOutput(positive_out, negative_out, out_latent)

class Trellis2TextureStage(IO.ComfyNode):
    """Sets up the texture-stage sampling pass. Reads coords / coord_counts /
    coord_resolution and the shape_slat (the per-voxel shape latent) from the
    incoming shape_latent dict — set there by Trellis2ShapeStage or
    Trellis2UpsampleStage. Builds an empty sparse latent at the same coord
    layout and attaches per-stage metadata to the conditioning."""
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2TextureStage",
            category="latent/3d",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Latent.Input("shape_latent"),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, positive, negative, shape_latent):
        channels = 32
        coords = shape_latent["coords"]
        coord_resolution = shape_latent.get("coord_resolution")

        batch_size, counts, max_tokens = infer_batched_coord_layout(coords)

        shape_slat = shape_latent["samples"]
        if shape_slat.ndim == 4:
            shape_slat = shape_slat.squeeze(-1).transpose(1, 2).reshape(-1, channels)

        latent = torch.zeros(batch_size, channels, max_tokens, 1)
        proj_pack = _proj_pack_from_conditioning(positive)
        model_frame = shape_latent.get("model_frame",
                                       "y_up" if proj_pack is not None else "z_up")
        extras = {
            "trellis2_generation_mode": "texture_generation",
            "trellis2_coords": coords,
            "trellis2_coord_counts": counts,
            "trellis2_shape_slat": shape_slat,
            "trellis2_model_frame": model_frame,
        }
        if proj_pack is not None and coord_resolution is not None:
            extras["trellis2_proj_feats"] = compute_stage_proj_feats(
                proj_pack, "tex_1024", coords=coords, coord_resolution=coord_resolution,
            )

        positive_out = _conditioning_set_extras(positive, extras)
        negative_out = _conditioning_set_extras(negative, extras)
        out_latent = {"samples": latent, "type": "trellis2", "coords": coords, "coord_counts": counts,
                      "model_frame": shape_latent.get("model_frame",
                                                       "y_up" if proj_pack is not None else "z_up")}
        if coord_resolution is not None:
            out_latent["coord_resolution"] = coord_resolution
        return IO.NodeOutput(positive_out, negative_out, out_latent)


class EmptyTrellis2LatentStructure(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyTrellis2LatentStructure",
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
        in_channels = 32
        resolution = 16
        latent = torch.zeros(batch_size, in_channels, resolution, resolution, resolution)
        return IO.NodeOutput({"samples": latent, "type": "trellis2"})

def _dinov3_patches_to_2d(tokens, image_size, patch_size=16):
    h_p = w_p = image_size // patch_size
    n_patches = h_p * w_p
    n_reg = tokens.shape[1] - 1 - n_patches
    if n_reg < 0 or tokens.shape[1] != 1 + n_reg + n_patches:
        raise ValueError(
            f"_dinov3_patches_to_2d: got {tokens.shape[1]} tokens, expected "
            f"1 (CLS) + N_reg + {h_p}*{w_p}={n_patches} patches at image_size={image_size}, "
            f"patch_size={patch_size}. Inferred N_reg={n_reg} which is invalid."
        )
    start = 1 + n_reg
    patches = tokens[:, start:start + n_patches]
    return patches.transpose(1, 2).reshape(tokens.shape[0], -1, h_p, w_p).contiguous()


def _crop_image_with_mask(item_image, item_mask, max_image_size=1024):
    img = item_image.permute(2, 0, 1).unsqueeze(0).cpu().float()
    mask = item_mask.unsqueeze(0).unsqueeze(0).cpu().float()
    # Upstream went float→PIL uint8 implicitly; match that to keep composite bit-exact.
    img = (img.clamp(0, 1) * 255.0).to(torch.uint8).float() / 255.0
    mask = (mask.clamp(0, 1) * 255.0).to(torch.uint8).float() / 255.0

    # Detect & correct an inverted mask
    m2d = mask[0, 0]
    border = torch.cat([m2d[0, :], m2d[-1, :], m2d[:, 0], m2d[:, -1]])
    if float(border.mean()) > 0.5:
        mask = 1.0 - mask

    H, W = img.shape[-2:]
    if max(H, W) > max_image_size:
        scale = max_image_size / max(H, W)
        new_w, new_h = int(W * scale), int(H * scale)
        img  = comfy.utils.common_upscale(img,  new_w, new_h, "lanczos",       "disabled")
        mask = comfy.utils.common_upscale(mask, new_w, new_h, "nearest-exact", "disabled")
        H, W = new_h, new_w
    scene_size = (W, H)

    alpha_u8 = (mask[0, 0].clamp(0, 1) * 255.0).to(torch.uint8)
    fg_pixels = (alpha_u8 > 204).nonzero()
    if fg_pixels.numel() > 0:
        y_min, x_min = fg_pixels.min(dim=0).values.tolist()
        y_max, x_max = fg_pixels.max(dim=0).values.tolist()
        center_y, center_x = (y_min + y_max) / 2.0, (x_min + x_max) / 2.0
        size = int(max(y_max - y_min, x_max - x_min) * 1.1)
        half = size // 2
        crop_x1 = int(center_x - half)
        crop_y1 = int(center_y - half)
        crop_x2 = crop_x1 + 2 * half
        crop_y2 = crop_y1 + 2 * half
    else:
        logging.warning("Mask for the image is empty. Pixal3D requires a clean foreground mask.")
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, W, H
    crop_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)

    # Zero-pad out-of-bounds slice (PIL.crop semantics).
    pad_l = max(0, -crop_x1)
    pad_t = max(0, -crop_y1)
    pad_r = max(0, crop_x2 - W)
    pad_b = max(0, crop_y2 - H)
    if pad_l or pad_t or pad_r or pad_b:
        img  = torch.nn.functional.pad(img,  (pad_l, pad_r, pad_t, pad_b), value=0.0)
        mask = torch.nn.functional.pad(mask, (pad_l, pad_r, pad_t, pad_b), value=0.0)
        crop_x1 += pad_l
        crop_x2 += pad_l
        crop_y1 += pad_t
        crop_y2 += pad_t
    cropped_img  = img [..., crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_mask = mask[..., crop_y1:crop_y2, crop_x1:crop_x2]

    composite = (cropped_img * cropped_mask).clamp(0, 1)
    composite = (composite * 255.0).round().clamp(0, 255).to(torch.uint8).float() / 255.0
    return composite, crop_bbox, scene_size

class Pixal3DConditioning(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Pixal3DConditioning",
            category="conditioning/video_models",
            inputs=[
                IO.ClipVision.Input("clip_vision_model", tooltip="DINOv3 ViT-L/16 ClipVision."),
                IO.Image.Input("image"),
                IO.Mask.Input("mask"),
                IO.Float.Input(
                    "camera_angle_x", display_name="fov",
                    default=11.46, min=1.0, max=170.0, step=0.01, advanced=True,
                    tooltip="Horizontal FOV in degrees (original default ~11.46° = 0.2 rad). "
                            "Wire a MoGeGeometryToFOV (axis='horizontal', unit='degrees') "
                            "output here for a MoGe-derived FOV.",
                ),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, clip_vision_model, image, mask, camera_angle_x) -> IO.NodeOutput:
        naf_model = getattr(clip_vision_model, "naf", None)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        batch_size = image.shape[0]
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        elif mask.shape[0] != batch_size:
            raise ValueError(f"Pixal3DConditioning mask batch {mask.shape[0]} != image batch {batch_size}")

        device = comfy.model_management.intermediate_device()

        cond_512_list, cond_1024_list = [], []
        patches_512_list, patches_1024_list = [], []
        composite_list = []
        crop_bbox_list, scene_size_list = [], []

        torch_device = comfy.model_management.get_torch_device()
        for b in range(batch_size):
            item_image = image[b]
            item_mask = mask[b] if mask.size(0) > 1 else mask[0]
            composite, crop_bbox, scene_size = _crop_image_with_mask(
                item_image, item_mask, max_image_size=1024)
            crop_bbox_list.append(crop_bbox)
            scene_size_list.append(scene_size)
            composite_list.append(composite)

            cond_512 = _dinov3_encode(clip_vision_model, composite, 512, want_patches=True)
            cond_1024 = _dinov3_encode(clip_vision_model, composite, 1024, want_patches=True)
            cond_512_list.append(cond_512["tokens"].to(device))
            cond_1024_list.append(cond_1024["tokens"].to(device))
            patches_512_list.append(cond_512["patches_2d"].to(device))
            patches_1024_list.append(cond_1024["patches_2d"].to(device))

        global_512 = torch.cat(cond_512_list, dim=0)
        global_1024 = torch.cat(cond_1024_list, dim=0)

        fm_512_dino = torch.cat(patches_512_list, dim=0)
        fm_1024_dino = torch.cat(patches_1024_list, dim=0)

        # The LR DINO grid AND the NAF HR grid are sampled separately
        # NAF targets per stage: shape_512=512, shape_1024=512, tex_1024=1024.
        def _naf_hr(lr_feat, composites, image_size, naf_target):
            if naf_model is None or naf_target is None:
                return None
            comfy.model_management.load_model_gpu(naf_model)
            inner = naf_model.model
            target_dtype = comfy.model_management.text_encoder_dtype(torch_device)
            if next(inner.parameters()).dtype != target_dtype:
                inner.to(dtype=target_dtype)
            hrs = []
            for i, c in enumerate(composites):
                img_i = comfy.utils.common_upscale(c, image_size, image_size, "lanczos", "disabled")\
                    .to(torch_device).to(target_dtype)
                lr_i = lr_feat[i:i + 1].to(torch_device).to(target_dtype)
                hr_i = inner(img_i, lr_i, naf_target, output_device=device)
                hrs.append(hr_i)
            return torch.cat(hrs, dim=0)

        hr_shape_512  = _naf_hr(fm_512_dino,  composite_list, 512,  (512, 512))
        hr_shape_1024 = _naf_hr(fm_1024_dino, composite_list, 1024, (512, 512))
        hr_tex_1024   = _naf_hr(fm_1024_dino, composite_list, 1024, (1024, 1024))

        # distance_from_fov: grid_point (-1, 0, 0) projects to pixel (0, image_resolution-1).
        # FOV widget is in degrees for UX; trig + downstream projection expect radians.
        camera_angle_x = math.radians(float(camera_angle_x))
        distance = 0.5 / math.tan(camera_angle_x / 2.0)
        cam_angle_t = torch.tensor([camera_angle_x] * batch_size, device=device, dtype=torch.float32)
        dist_t = torch.tensor([distance] * batch_size, device=device, dtype=torch.float32)
        scale_t = torch.ones(batch_size, device=device, dtype=torch.float32)
        T = build_proj_transform_matrix(dist_t, batch_size, device=device, dtype=torch.float32)

        proj_pack = {
            "stages": {
                "ss":         {"feature_map": fm_512_dino,    "feature_map_hr": None,         "image_resolution": 512},
                "shape_512":  {"feature_map": fm_512_dino,    "feature_map_hr": hr_shape_512, "image_resolution": 512},
                "shape_1024": {"feature_map": fm_1024_dino,   "feature_map_hr": hr_shape_1024,"image_resolution": 1024},
                "tex_1024":   {"feature_map": fm_1024_dino,   "feature_map_hr": hr_tex_1024,  "image_resolution": 1024},
            },
            "transform_matrix": T,
            "camera_angle_x": cam_angle_t,
            "mesh_scale": scale_t,
            "distance": dist_t,
            "patch_size": 16,
            "crop_bboxes": crop_bbox_list,
            "scene_sizes": scene_size_list,
        }

        # global_512 → SS/shape_512 cross-attn; global_1024 → shape_1024/tex_1024.
        ss_proj_feats = compute_stage_proj_feats(
            proj_pack, "ss", dense_grid_resolution=16, batch_size=batch_size,
            device=torch_device,
        )
        neg_global = torch.zeros_like(global_512)
        neg_embeds = torch.zeros_like(global_1024)
        base_extras = {
            "embeds": global_1024, "proj_feat_pack": proj_pack,
            "trellis2_proj_feats": ss_proj_feats,
        }
        neg_extras = {
            "embeds": neg_embeds, "proj_feat_pack": proj_pack,
            "trellis2_proj_feats": ss_proj_feats,
        }
        positive = [[global_512, base_extras]]
        negative = [[neg_global, neg_extras]]
        return IO.NodeOutput(positive, negative)


class GetMeshInfo(IO.ComfyNode):
    """Report vertex / face counts and attributes for a MESH, displayed on the
    node (and as a string output). Counts are comma-formatted since meshes can
    run into the millions of faces. Passes the mesh through unchanged."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GetMeshInfo",
            display_name="Get Mesh Info",
            category="latent/3d",
            inputs=[IO.Mesh.Input("mesh")],
            outputs=[
                IO.Mesh.Output(display_name="mesh"),
                IO.String.Output(display_name="info"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @staticmethod
    def _fmt(n: int) -> str:
        # e.g. 1234567 -> "1,234,567 (1.23M)"; small numbers stay plain.
        s = f"{n:,}"
        if n >= 1_000_000:
            s += f" ({n / 1_000_000:.2f}M)"
        elif n >= 10_000:
            s += f" ({n / 1_000:.1f}K)"
        return s

    @classmethod
    def execute(cls, mesh):
        B = mesh.vertices.shape[0]
        # Honour per-item counts when the batch is zero-padded; else use the row sizes.
        if mesh.vertex_counts is not None:
            v_counts = [int(x) for x in mesh.vertex_counts.tolist()]
            f_counts = [int(x) for x in mesh.face_counts.tolist()]
        else:
            v_counts = [int(mesh.vertices.shape[1])] * B
            f_counts = [int(mesh.faces.shape[1])] * B

        attrs = []
        for name in ("uvs", "vertex_colors", "normals", "tangents", "texture", "metallic_roughness", "normal_map"):
            t = getattr(mesh, name, None)
            if t is not None:
                if name in ("texture", "metallic_roughness", "normal_map"):
                    attrs.append(f"{name} {int(t.shape[-3])}×{int(t.shape[-2])}")  # H×W
                else:
                    attrs.append(name)

        lines = []
        if B > 1:
            lines.append(f"Batch:      {B} meshes")
            lines.append(f"Vertices:   {cls._fmt(sum(v_counts))} total")
            lines.append(f"Faces:      {cls._fmt(sum(f_counts))} total")
            for i in range(B):
                lines.append(f"  [{i}]  {v_counts[i]:>10,} verts  ·  {f_counts[i]:>10,} faces")
        else:
            lines.append(f"Vertices:   {cls._fmt(v_counts[0])}")
            lines.append(f"Faces:      {cls._fmt(f_counts[0])}")
        lines.append(f"Attributes: {', '.join(attrs) if attrs else 'none'}")

        info = "\n".join(lines)
        logging.info("[GetMeshInfo]\n%s", info)

        if cls.hidden.unique_id:
            PromptServer.instance.send_progress_text(info, cls.hidden.unique_id)
        return IO.NodeOutput(mesh, info, ui=UI.PreviewText(info))


class Trellis2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Trellis2Conditioning,
            Pixal3DConditioning,
            Trellis2ShapeStage,
            EmptyTrellis2LatentStructure,
            Trellis2TextureStage,
            VaeDecodeTextureTrellis,
            VaeDecodeShapeTrellis,
            VaeDecodeStructureTrellis2,
            Trellis2UpsampleStage,
            GetMeshInfo,
        ]


async def comfy_entrypoint() -> Trellis2Extension:
    return Trellis2Extension()

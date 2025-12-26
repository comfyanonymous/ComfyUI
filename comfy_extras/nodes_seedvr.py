from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import torch
import math
from einops import rearrange

import gc
import comfy.model_management
from comfy.utils import ProgressBar

import torch.nn.functional as F
from torchvision.transforms import functional as TVF
from torchvision.transforms import Lambda, Normalize
from torchvision.transforms.functional import InterpolationMode

@torch.inference_mode()
def tiled_vae(x, vae_model, tile_size=(512, 512), tile_overlap=(64, 64), temporal_size=16, encode=True):

    gc.collect()
    torch.cuda.empty_cache()

    x = x.to(next(vae_model.parameters()).dtype)
    if x.ndim != 5:
        x = x.unsqueeze(2)

    b, c, d, h, w = x.shape

    sf_s = getattr(vae_model, "spatial_downsample_factor", 8)
    sf_t = getattr(vae_model, "temporal_downsample_factor", 4)

    if encode:
        ti_h, ti_w = tile_size
        ov_h, ov_w = tile_overlap
        target_d = (d + sf_t - 1) // sf_t
        target_h = (h + sf_s - 1) // sf_s
        target_w = (w + sf_s - 1) // sf_s
    else:
        ti_h = max(1, tile_size[0] // sf_s)
        ti_w = max(1, tile_size[1] // sf_s)
        ov_h = max(0, tile_overlap[0] // sf_s)
        ov_w = max(0, tile_overlap[1] // sf_s)

        target_d = d * sf_t
        target_h = h * sf_s
        target_w = w * sf_s

    stride_h = max(1, ti_h - ov_h)
    stride_w = max(1, ti_w - ov_w)

    storage_device = vae_model.device
    result = None
    count = None

    def run_temporal_chunks(spatial_tile):
        chunk_results = []
        t_dim_size = spatial_tile.shape[2]

        if encode:
            input_chunk = temporal_size
        else:
            input_chunk = max(1, temporal_size // sf_t)

        for i in range(0, t_dim_size, input_chunk):
            t_chunk = spatial_tile[:, :, i : i + input_chunk, :, :]

            if encode:
                out = vae_model.encode(t_chunk)
            else:
                out = vae_model.decode_(t_chunk)

            if isinstance(out, (tuple, list)): out = out[0]

            if out.ndim == 4: out = out.unsqueeze(2)

            chunk_results.append(out.to(storage_device))

        return torch.cat(chunk_results, dim=2)

    ramp_cache = {}
    def get_ramp(steps):
        if steps not in ramp_cache:
            t = torch.linspace(0, 1, steps=steps, device=storage_device, dtype=torch.float32)
            ramp_cache[steps] = 0.5 - 0.5 * torch.cos(t * torch.pi)
        return ramp_cache[steps]

    total_tiles = len(range(0, h, stride_h)) * len(range(0, w, stride_w))
    bar = ProgressBar(total_tiles)

    for y_idx in range(0, h, stride_h):
        y_end = min(y_idx + ti_h, h)

        for x_idx in range(0, w, stride_w):
            x_end = min(x_idx + ti_w, w)

            tile_x = x[:, :, :, y_idx:y_end, x_idx:x_end]

            # Run VAE
            tile_out = run_temporal_chunks(tile_x)

            if result is None:
                b_out, c_out = tile_out.shape[0], tile_out.shape[1]
                result = torch.zeros((b_out, c_out, target_d, target_h, target_w), device=storage_device, dtype=torch.float32)
                count = torch.zeros((1, 1, 1, target_h, target_w), device=storage_device, dtype=torch.float32)

            if encode:
                ys, ye = y_idx // sf_s, (y_idx // sf_s) + tile_out.shape[3]
                xs, xe = x_idx // sf_s, (x_idx // sf_s) + tile_out.shape[4]
                cur_ov_h = max(0, min(ov_h // sf_s, tile_out.shape[3] // 2))
                cur_ov_w = max(0, min(ov_w // sf_s, tile_out.shape[4] // 2))
            else:
                ys, ye = y_idx * sf_s, (y_idx * sf_s) + tile_out.shape[3]
                xs, xe = x_idx * sf_s, (x_idx * sf_s) + tile_out.shape[4]
                cur_ov_h = max(0, min(ov_h, tile_out.shape[3] // 2))
                cur_ov_w = max(0, min(ov_w, tile_out.shape[4] // 2))

            w_h = torch.ones((tile_out.shape[3],), device=storage_device)
            w_w = torch.ones((tile_out.shape[4],), device=storage_device)

            if cur_ov_h > 0:
                r = get_ramp(cur_ov_h)
                if y_idx > 0: w_h[:cur_ov_h] = r
                if y_end < h: w_h[-cur_ov_h:] = 1.0 - r

            if cur_ov_w > 0:
                r = get_ramp(cur_ov_w)
                if x_idx > 0: w_w[:cur_ov_w] = r
                if x_end < w: w_w[-cur_ov_w:] = 1.0 - r

            final_weight = w_h.view(1,1,1,-1,1) * w_w.view(1,1,1,1,-1)

            valid_d = min(tile_out.shape[2], result.shape[2])
            tile_out = tile_out[:, :, :valid_d, :, :]

            tile_out.mul_(final_weight)

            result[:, :, :valid_d, ys:ye, xs:xe] += tile_out
            count[:, :, :, ys:ye, xs:xe] += final_weight

            del tile_out, final_weight, tile_x, w_h, w_w
            bar.update(1)

    result.div_(count.clamp(min=1e-6))

    if result.device != x.device:
        result = result.to(x.device).to(x.dtype)

    if x.shape[2] == 1 and sf_t == 1:
        result = result.squeeze(2)

    return result

def clear_vae_memory(vae_model):
    for module in vae_model.modules():
        if hasattr(module, "memory"):
            module.memory = None
    if hasattr(vae_model, "original_image_video"):
        del vae_model.original_image_video

    if hasattr(vae_model, "tiled_args"):
        del vae_model.tiled_args
    gc.collect()
    torch.cuda.empty_cache()

def expand_dims(tensor, ndim):
    shape = tensor.shape + (1,) * (ndim - tensor.ndim)
    return tensor.reshape(shape)

def get_conditions(latent, latent_blur):
    t, h, w, c = latent.shape
    cond = torch.ones([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
    cond[:, ..., :-1] = latent_blur[:]
    cond[:, ..., -1:] = 1.0
    return cond

def timestep_transform(timesteps, latents_shapes):
    vt = 4
    vs = 8
    frames = (latents_shapes[:, 0] - 1) * vt + 1
    heights = latents_shapes[:, 1] * vs
    widths = latents_shapes[:, 2] * vs

    # Compute shift factor.
    def get_lin_function(x1, y1, x2, y2):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    img_shift_fn = get_lin_function(x1=256 * 256, y1=1.0, x2=1024 * 1024, y2=3.2)
    vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
    shift = torch.where(
        frames > 1,
        vid_shift_fn(heights * widths * frames),
        img_shift_fn(heights * widths),
    ).to(timesteps.device)

    # Shift timesteps.
    T = 1000.0
    timesteps = timesteps / T
    timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
    timesteps = timesteps * T
    return timesteps

def inter(x_0, x_T, t):
    t = expand_dims(t, x_0.ndim)
    T = 1000.0
    B = lambda t: t / T
    A = lambda t: 1 - (t / T)
    return A(t) * x_0 + B(t) * x_T
def area_resize(image, max_area):

    height, width = image.shape[-2:]
    scale = math.sqrt(max_area / (height * width))

    resized_height, resized_width = round(height * scale), round(width * scale)

    return TVF.resize(
        image,
        size=(resized_height, resized_width),
        interpolation=InterpolationMode.BICUBIC,
    )

def div_pad(image, factor):

    height_factor, width_factor = factor
    height, width = image.shape[-2:]

    pad_height = (height_factor - (height % height_factor)) % height_factor
    pad_width = (width_factor - (width % width_factor)) % width_factor

    if pad_height == 0 and pad_width == 0:
        return image

    if isinstance(image, torch.Tensor):
        padding = (0, pad_width, 0, pad_height)
        image = torch.nn.functional.pad(image, padding, mode='constant', value=0.0)

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

def side_resize(image, size):
    antialias = not (isinstance(image, torch.Tensor) and image.device.type == 'mps')
    resized = TVF.resize(image, size, InterpolationMode.BICUBIC, antialias=antialias)
    return resized

class SeedVR2InputProcessing(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id = "SeedVR2InputProcessing",
            category="image/video",
            inputs = [
                io.Image.Input("images"),
                io.Vae.Input("vae"),
                io.Int.Input("resolution", default = 1280, min = 120), # just non-zero value
                io.Int.Input("spatial_tile_size", default = 512, min = 1),
                io.Int.Input("spatial_overlap", default = 64, min = 1),
                io.Int.Input("temporal_tile_size", default = 8, min = 1),
                io.Boolean.Input("enable_tiling", default=False),
            ],
            outputs = [
                io.Latent.Output("vae_conditioning")
            ]
        )

    @classmethod
    def execute(cls, images, vae, resolution, spatial_tile_size, temporal_tile_size, spatial_overlap, enable_tiling):
        device = vae.patcher.load_device

        offload_device = comfy.model_management.intermediate_device()
        main_device = comfy.model_management.get_torch_device()
        images = images.to(main_device)
        vae_model = vae.first_stage_model
        scale = 0.9152; shift = 0
        if images.dim() != 5: # add the t dim
            images = images.unsqueeze(0)
        images = images.permute(0, 1, 4, 2, 3)

        b, t, c, h, w = images.shape
        images = images.reshape(b * t, c, h, w)

        clip = Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        normalize = Normalize(0.5, 0.5)
        images = side_resize(images, resolution)

        images = clip(images)
        o_h, o_w = images.shape[-2:]
        images = div_pad(images, (16, 16))
        images = normalize(images)
        _, _, new_h, new_w = images.shape

        images = images.reshape(b, t, c, new_h, new_w)
        images = cut_videos(images)

        images = rearrange(images, "b t c h w -> b c t h w")
        images = images.to(device)
        vae_model = vae_model.to(device)

        # in case users a non-compatiable number for tiling
        def make_divisible(val, divisor):
            return max(divisor, round(val / divisor) * divisor)

        temporal_tile_size = make_divisible(temporal_tile_size, 4)
        spatial_tile_size = make_divisible(spatial_tile_size, 32)
        spatial_overlap = make_divisible(spatial_overlap, 32)

        if spatial_overlap >= spatial_tile_size:
            spatial_overlap = max(0, spatial_tile_size - 8)

        args = {"tile_size": (spatial_tile_size, spatial_tile_size), "tile_overlap": (spatial_overlap, spatial_overlap),
                "temporal_size":temporal_tile_size}
        if enable_tiling:
            latent = tiled_vae(images, vae_model, encode=True, **args)
        else:
            latent = vae_model.encode(images, orig_dims = [o_h, o_w])[0]

        clear_vae_memory(vae_model)
        #images = images.to(offload_device)
        #vae_model = vae_model.to(offload_device)

        vae_model.img_dims = [o_h, o_w]
        args["enable_tiling"] = enable_tiling
        vae_model.tiled_args = args
        vae_model.original_image_video = images

        latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
        latent = rearrange(latent, "b c ... -> b ... c")

        latent = (latent - shift) * scale
        latent = latent.to(offload_device)

        return io.NodeOutput({"samples": latent})

class SeedVR2Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2Conditioning",
            category="image/video",
            inputs=[
                io.Latent.Input("vae_conditioning"),
                io.Model.Input("model"),
            ],
            outputs=[io.Conditioning.Output(display_name = "positive"),
                     io.Conditioning.Output(display_name = "negative"),
                     io.Latent.Output(display_name = "latent")],
        )

    @classmethod
    def execute(cls, vae_conditioning, model) -> io.NodeOutput:

        vae_conditioning = vae_conditioning["samples"]
        device = vae_conditioning.device
        model = model.model.diffusion_model
        pos_cond = model.positive_conditioning
        neg_cond = model.negative_conditioning

        noises = torch.randn_like(vae_conditioning).to(device)
        aug_noises =  torch.randn_like(vae_conditioning).to(device)
        aug_noises = noises * 0.1 + aug_noises * 0.05
        cond_noise_scale = 0.0
        t = (
            torch.tensor([1000.0])
            * cond_noise_scale
        ).to(device)
        shape = torch.tensor(vae_conditioning.shape[1:]).to(device)[None] # avoid batch dim
        t = timestep_transform(t, shape)
        cond = inter(vae_conditioning, aug_noises, t)
        condition = torch.stack([get_conditions(noise, c) for noise, c in zip(noises, cond)])
        condition = condition.movedim(-1, 1)
        noises = noises.movedim(-1, 1)

        pos_shape = pos_cond.shape[0]
        neg_shape = neg_cond.shape[0]
        diff = abs(pos_shape - neg_shape)
        if pos_shape > neg_shape:
            neg_cond = F.pad(neg_cond, (0, 0, 0, diff))
        else:
            pos_cond = F.pad(pos_cond, (0, 0, 0, diff))

        noises = rearrange(noises, "b c t h w -> b (c t) h w")
        condition = rearrange(condition, "b c t h w -> b (c t) h w")

        negative = [[neg_cond.unsqueeze(0), {"condition": condition}]]
        positive = [[pos_cond.unsqueeze(0), {"condition": condition}]]

        return io.NodeOutput(positive, negative, {"samples": noises})

class SeedVRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SeedVR2Conditioning,
            SeedVR2InputProcessing
        ]

async def comfy_entrypoint() -> SeedVRExtension:
    return SeedVRExtension()

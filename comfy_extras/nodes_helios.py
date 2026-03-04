import math
import torch

import nodes
import comfy.model_management
import comfy.model_patcher
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import node_helpers

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


def _parse_int_list(values, default):
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        out = []
        for v in values:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out if len(out) > 0 else default

    parts = [x.strip() for x in str(values).replace(";", ",").split(",")]
    out = []
    for p in parts:
        if len(p) == 0:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return out if len(out) > 0 else default


def _parse_float_list(values, default):
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        out = []
        for v in values:
            try:
                out.append(float(v))
            except Exception:
                pass
        return out if len(out) > 0 else default

    parts = [x.strip() for x in str(values).replace(";", ",").split(",")]
    out = []
    for p in parts:
        if len(p) == 0:
            continue
        try:
            out.append(float(p))
        except Exception:
            continue
    return out if len(out) > 0 else default


def _extract_condition_value(conditioning, key):
    for c in conditioning:
        if len(c) < 2:
            continue
        value = c[1].get(key, None)
        if value is not None:
            return value
    return None


def _upsample_latent_5d(latent, scale=2):
    b, c, t, h, w = latent.shape
    x = latent.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = comfy.utils.common_upscale(x, w * scale, h * scale, "nearest-exact", "disabled")
    x = x.reshape(b, t, c, h * scale, w * scale).permute(0, 2, 1, 3, 4)
    return x


def _sample_block_noise_like(latent, gamma, patch_size=(1, 2, 2)):
    b, c, t, h, w = latent.shape
    _, ph, pw = patch_size
    block_size = ph * pw

    cov = torch.eye(block_size, device=latent.device) * (1.0 + gamma) - torch.ones(block_size, block_size, device=latent.device) * gamma
    cov += torch.eye(block_size, device=latent.device) * 1e-6

    dist = torch.distributions.MultivariateNormal(torch.zeros(block_size, device=latent.device), covariance_matrix=cov)
    block_number = b * c * t * max(1, h // ph) * max(1, w // pw)

    noise = dist.sample((block_number,))
    noise = noise.view(b, c, t, max(1, h // ph), max(1, w // pw), ph, pw)
    noise = noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(b, c, t, max(1, h // ph) * ph, max(1, w // pw) * pw)
    noise = noise[:, :, :, :h, :w]
    return noise


def _helios_global_sigmas(num_train_timesteps=1000, shift=1.0):
    alphas = torch.linspace(1.0, 1.0 / float(num_train_timesteps), num_train_timesteps + 1)
    sigmas = 1.0 - alphas
    if abs(shift - 1.0) > 1e-8:
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    return torch.flip(sigmas, dims=[0])[:-1]


def _helios_stage_tables(stage_count, stage_range, gamma, num_train_timesteps=1000, shift=1.0):
    sigmas = _helios_global_sigmas(num_train_timesteps=num_train_timesteps, shift=shift)

    ori_start_sigmas = {}
    start_sigmas = {}
    end_sigmas = {}
    timestep_ratios = {}
    timesteps_per_stage = {}
    sigmas_per_stage = {}

    stage_distance = []
    for i in range(stage_count):
        start_indice = int(max(0.0, min(1.0, stage_range[i])) * num_train_timesteps)
        end_indice = int(max(0.0, min(1.0, stage_range[i + 1])) * num_train_timesteps)
        start_indice = max(0, min(num_train_timesteps - 1, start_indice))
        end_indice = max(0, min(num_train_timesteps, end_indice))

        start_sigma = float(sigmas[start_indice].item())
        end_sigma = float(sigmas[end_indice].item()) if end_indice < num_train_timesteps else 0.0
        ori_start_sigmas[i] = start_sigma

        if i != 0:
            ori_sigma = 1.0 - start_sigma
            corrected_sigma = (1.0 / (math.sqrt(1.0 + (1.0 / gamma)) * (1.0 - ori_sigma) + ori_sigma)) * ori_sigma
            start_sigma = 1.0 - corrected_sigma

        stage_distance.append(start_sigma - end_sigma)
        start_sigmas[i] = start_sigma
        end_sigmas[i] = end_sigma

    tot_distance = sum(stage_distance) if sum(stage_distance) > 1e-12 else 1.0
    for i in range(stage_count):
        start_ratio = 0.0 if i == 0 else sum(stage_distance[:i]) / tot_distance
        end_ratio = 0.9999999999999999 if i == stage_count - 1 else sum(stage_distance[: i + 1]) / tot_distance
        timestep_ratios[i] = (start_ratio, end_ratio)

        tmax = min(float(sigmas[int(start_ratio * num_train_timesteps)].item() * num_train_timesteps), 999.0)
        tmin = float(sigmas[min(int(end_ratio * num_train_timesteps), num_train_timesteps - 1)].item() * num_train_timesteps)
        timesteps_per_stage[i] = torch.linspace(tmax, tmin, num_train_timesteps)
        sigmas_per_stage[i] = torch.linspace(0.999, 0.0, num_train_timesteps)

    return {
        "ori_start_sigmas": ori_start_sigmas,
        "start_sigmas": start_sigmas,
        "end_sigmas": end_sigmas,
        "timestep_ratios": timestep_ratios,
        "timesteps_per_stage": timesteps_per_stage,
        "sigmas_per_stage": sigmas_per_stage,
    }


def _helios_stage_sigmas(stage_idx, stage_steps, stage_tables, is_distilled=False, is_amplify_first_stage=False):
    stage_steps = max(1, int(stage_steps))
    if is_distilled:
        stage_steps = stage_steps * 2 if (is_amplify_first_stage and stage_idx == 0) else stage_steps

    stage_sigma_src = stage_tables["sigmas_per_stage"][stage_idx]
    sigmas = torch.linspace(float(stage_sigma_src[0].item()), float(stage_sigma_src[-1].item()), stage_steps + 1)
    return sigmas


def _helios_stage_timesteps(stage_idx, stage_steps, stage_tables, is_distilled=False, is_amplify_first_stage=False):
    stage_steps = max(1, int(stage_steps))
    if is_distilled:
        stage_steps = stage_steps * 2 if (is_amplify_first_stage and stage_idx == 0) else stage_steps

    stage_timestep_src = stage_tables["timesteps_per_stage"][stage_idx]
    timesteps = torch.linspace(float(stage_timestep_src[0].item()), float(stage_timestep_src[-1].item()), stage_steps)
    return timesteps


def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / float(max_seq_len - base_seq_len)
    b = base_shift - m * float(base_seq_len)
    return float(image_seq_len) * m + b


def _time_shift_linear(mu, sigma, t):
    return mu / (mu + (1.0 / t - 1.0) ** sigma)


def _time_shift_exponential(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def _time_shift(t, mu, sigma=1.0, mode="exponential"):
    t = torch.clamp(t, min=1e-6, max=0.999999)
    if mode == "linear":
        return _time_shift_linear(mu, sigma, t)
    return _time_shift_exponential(mu, sigma, t)


def _optimized_scale(positive_flat, negative_flat):
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat * negative_flat, dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm


def _build_cfg_zero_star_pre_cfg(stage_idx, zero_steps, use_zero_init):
    state = {"i": 0}

    def pre_cfg_fn(args):
        conds_out = args["conds_out"]
        if len(conds_out) < 2 or conds_out[1] is None:
            state["i"] += 1
            return conds_out

        noise_pred_text = conds_out[0]
        noise_uncond = conds_out[1]
        cfg = float(args.get("cond_scale", 1.0))

        positive_flat = noise_pred_text.view(noise_pred_text.shape[0], -1)
        negative_flat = noise_uncond.view(noise_uncond.shape[0], -1)
        alpha = _optimized_scale(positive_flat, negative_flat)
        alpha = alpha.view(noise_pred_text.shape[0], *([1] * (noise_pred_text.ndim - 1))).to(noise_pred_text.dtype)

        if stage_idx == 0 and state["i"] <= int(zero_steps) and bool(use_zero_init):
            final = noise_pred_text * 0.0
        else:
            final = noise_uncond * alpha + cfg * (noise_pred_text - noise_uncond * alpha)

        state["i"] += 1
        # Return identical cond/uncond so downstream cfg_function keeps `final` unchanged.
        return [final, final]

    return pre_cfg_fn


def _helios_euler_sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = model(x, sigma * s_in, **extra_args)

        sigma_safe = sigma if float(sigma) > 1e-8 else sigma.new_tensor(1e-8)
        flow_pred = (x - denoised) / sigma_safe

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})

        x = x + (sigma_next - sigma) * flow_pred

    return x


def _helios_dmd_sample(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    dmd_noisy_tensor=None,
    dmd_sigmas=None,
    dmd_timesteps=None,
    all_timesteps=None,
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    if dmd_noisy_tensor is None:
        dmd_noisy_tensor = x
    dmd_noisy_tensor = dmd_noisy_tensor.to(device=x.device, dtype=x.dtype)
    if dmd_sigmas is None:
        dmd_sigmas = sigmas
    if dmd_timesteps is None:
        dmd_timesteps = torch.arange(len(sigmas) - 1, device=sigmas.device, dtype=sigmas.dtype)
    if all_timesteps is None:
        all_timesteps = dmd_timesteps

    def timestep_to_sigma(t):
        dt = dmd_timesteps.to(device=x.device, dtype=x.dtype)
        ds = dmd_sigmas.to(device=x.device, dtype=x.dtype)
        tid = torch.argmin(torch.abs(dt - t))
        tid = torch.clamp(tid, min=0, max=ds.shape[0] - 1)
        return ds[tid]

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        timestep = all_timesteps[i] if i < len(all_timesteps) else i
        denoised = model(x, sigma * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})

        if i < (len(sigmas) - 2):
            timestep_next = all_timesteps[i + 1] if i + 1 < len(all_timesteps) else (i + 1)
            sigma_t = timestep_to_sigma(torch.as_tensor(timestep, device=x.device, dtype=x.dtype))
            sigma_next_t = timestep_to_sigma(torch.as_tensor(timestep_next, device=x.device, dtype=x.dtype))
            x0_pred = x - sigma_t * ((x - denoised) / torch.clamp(sigma_t, min=1e-8))
            x = (1.0 - sigma_next_t) * x0_pred + sigma_next_t * dmd_noisy_tensor
        else:
            x = denoised

    return x


def _set_helios_history_values(positive, negative, history_latent, history_sizes, keep_first_frame, prefix_latent=None):
    latent = history_latent
    if latent is None or len(latent.shape) != 5:
        return positive, negative

    sizes = list(history_sizes)
    if len(sizes) != 3:
        sizes = [16, 2, 1]
    sizes = [max(0, int(v)) for v in sizes]
    total = sum(sizes)
    if total <= 0:
        return positive, negative

    if latent.shape[2] < total:
        pad = torch.zeros(
            latent.shape[0],
            latent.shape[1],
            total - latent.shape[2],
            latent.shape[3],
            latent.shape[4],
            device=latent.device,
            dtype=latent.dtype,
        )
        hist = torch.cat([pad, latent], dim=2)
    else:
        hist = latent[:, :, -total:]

    latents_history_long, latents_history_mid, latents_history_short_base = hist.split(sizes, dim=2)

    if keep_first_frame:
        if prefix_latent is not None:
            prefix = prefix_latent
        elif latent.shape[2] > 0:
            prefix = latent[:, :, :1]
        else:
            prefix = torch.zeros(latent.shape[0], latent.shape[1], 1, latent.shape[3], latent.shape[4], device=latent.device, dtype=latent.dtype)
        latents_history_short = torch.cat([prefix, latents_history_short_base], dim=2)
    else:
        latents_history_short = latents_history_short_base

    idx_short = torch.arange(latents_history_short.shape[2], device=latent.device, dtype=latent.dtype).unsqueeze(0).expand(latent.shape[0], -1)
    idx_mid = torch.arange(latents_history_mid.shape[2], device=latent.device, dtype=latent.dtype).unsqueeze(0).expand(latent.shape[0], -1)
    idx_long = torch.arange(latents_history_long.shape[2], device=latent.device, dtype=latent.dtype).unsqueeze(0).expand(latent.shape[0], -1)

    values = {
        "latents_history_short": latents_history_short,
        "latents_history_mid": latents_history_mid,
        "latents_history_long": latents_history_long,
        "indices_latents_history_short": idx_short,
        "indices_latents_history_mid": idx_mid,
        "indices_latents_history_long": idx_long,
    }

    positive = node_helpers.conditioning_set_values(positive, values)
    negative = node_helpers.conditioning_set_values(negative, values)
    return positive, negative


def _build_helios_indices(batch, history_sizes, keep_first_frame, hidden_frames, device, dtype):
    sizes = list(history_sizes)
    if len(sizes) != 3:
        sizes = [16, 2, 1]
    sizes = [max(0, int(v)) for v in sizes]
    long_size, mid_size, short_base_size = sizes

    if keep_first_frame:
        total = 1 + long_size + mid_size + short_base_size + hidden_frames
        indices = torch.arange(total, device=device, dtype=dtype)
        splits = [1, long_size, mid_size, short_base_size, hidden_frames]
        indices_prefix, idx_long, idx_mid, idx_1x, idx_hidden = torch.split(indices, splits, dim=0)
        idx_short = torch.cat([indices_prefix, idx_1x], dim=0)
    else:
        total = long_size + mid_size + short_base_size + hidden_frames
        indices = torch.arange(total, device=device, dtype=dtype)
        splits = [long_size, mid_size, short_base_size, hidden_frames]
        idx_long, idx_mid, idx_short, idx_hidden = torch.split(indices, splits, dim=0)

    idx_hidden = idx_hidden.unsqueeze(0).expand(batch, -1)
    idx_short = idx_short.unsqueeze(0).expand(batch, -1)
    idx_mid = idx_mid.unsqueeze(0).expand(batch, -1)
    idx_long = idx_long.unsqueeze(0).expand(batch, -1)
    return idx_hidden, idx_short, idx_mid, idx_long


class HeliosImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HeliosImageToVideo",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=640, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=384, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=132, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image", optional=True),
                io.String.Input("history_sizes", default="16,2,1", advanced=True),
                io.Boolean.Input("keep_first_frame", default=True, advanced=True),
                io.Int.Input("num_latent_frames_per_chunk", default=9, min=1, max=256, advanced=True),
                io.Boolean.Input("add_noise_to_image_latents", default=True, advanced=True),
                io.Float.Input("image_noise_sigma_min", default=0.111, min=0.0, max=1.0, step=0.0001, round=False, advanced=True),
                io.Float.Input("image_noise_sigma_max", default=0.135, min=0.0, max=1.0, step=0.0001, round=False, advanced=True),
                io.Int.Input("noise_seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, advanced=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        start_image=None,
        history_sizes="16,2,1",
        keep_first_frame=True,
        num_latent_frames_per_chunk=9,
        add_noise_to_image_latents=True,
        image_noise_sigma_min=0.111,
        image_noise_sigma_max=0.135,
        noise_seed=0,
    ) -> io.NodeOutput:
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        latent = torch.zeros([batch_size, latent_channels, latent_t, height // spacial_scale, width // spacial_scale], device=comfy.model_management.intermediate_device())

        sizes = _parse_int_list(history_sizes, [16, 2, 1])
        if len(sizes) != 3:
            sizes = [16, 2, 1]
        sizes = sorted([max(0, int(v)) for v in sizes], reverse=True)
        hist_len = max(1, sum(sizes))
        history_latent = torch.zeros([batch_size, latent_channels, hist_len, latent.shape[-2], latent.shape[-1]], device=latent.device, dtype=latent.dtype)
        image_latent_prefix = None

        if start_image is not None:
            image = comfy.utils.common_upscale(start_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            img_latent = vae.encode(image[:, :, :, :3])
            img_latent = comfy.utils.repeat_to_batch_size(img_latent, batch_size)
            image_latent_prefix = img_latent[:, :, :1]

            if add_noise_to_image_latents:
                g = torch.Generator(device=img_latent.device)
                g.manual_seed(int(noise_seed))
                sigma = (
                    torch.rand((img_latent.shape[0], 1, 1, 1, 1), device=img_latent.device, generator=g, dtype=img_latent.dtype)
                    * (float(image_noise_sigma_max) - float(image_noise_sigma_min))
                    + float(image_noise_sigma_min)
                )
                image_latent_prefix = sigma * torch.randn_like(image_latent_prefix, generator=g) + (1.0 - sigma) * image_latent_prefix

            min_frames = max(1, (int(num_latent_frames_per_chunk) - 1) * 4 + 1)
            fake_video = image.repeat(min_frames, 1, 1, 1)
            fake_latents_full = vae.encode(fake_video)
            fake_latent = comfy.utils.repeat_to_batch_size(fake_latents_full[:, :, -1:], batch_size)
            history_latent[:, :, -1:] = fake_latent

        positive, negative = _set_helios_history_values(positive, negative, history_latent, sizes, keep_first_frame, prefix_latent=image_latent_prefix)
        return io.NodeOutput(
            positive,
            negative,
            {
                "samples": latent,
                "helios_history_latent": history_latent,
                "helios_image_latent_prefix": image_latent_prefix,
            },
        )


class HeliosVideoToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HeliosVideoToVideo",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=640, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=384, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=132, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("video", optional=True),
                io.String.Input("history_sizes", default="16,2,1", advanced=True),
                io.Boolean.Input("keep_first_frame", default=True, advanced=True),
                io.Boolean.Input("add_noise_to_video_latents", default=True, advanced=True),
                io.Float.Input("video_noise_sigma_min", default=0.111, min=0.0, max=1.0, step=0.0001, round=False, advanced=True),
                io.Float.Input("video_noise_sigma_max", default=0.135, min=0.0, max=1.0, step=0.0001, round=False, advanced=True),
                io.Int.Input("noise_seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, advanced=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        video=None,
        history_sizes="16,2,1",
        keep_first_frame=True,
        add_noise_to_video_latents=True,
        video_noise_sigma_min=0.111,
        video_noise_sigma_max=0.135,
        noise_seed=0,
    ) -> io.NodeOutput:
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        latent = torch.zeros([batch_size, latent_channels, latent_t, height // spacial_scale, width // spacial_scale], device=comfy.model_management.intermediate_device())

        sizes = _parse_int_list(history_sizes, [16, 2, 1])
        if len(sizes) != 3:
            sizes = [16, 2, 1]
        sizes = sorted([max(0, int(v)) for v in sizes], reverse=True)
        hist_len = max(1, sum(sizes))
        history_latent = torch.zeros([batch_size, latent_channels, hist_len, latent.shape[-2], latent.shape[-1]], device=latent.device, dtype=latent.dtype)
        image_latent_prefix = None

        if video is not None:
            video = comfy.utils.common_upscale(video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            vid_latent = vae.encode(video[:, :, :, :3])
            if add_noise_to_video_latents:
                g = torch.Generator(device=vid_latent.device)
                g.manual_seed(int(noise_seed))
                frame_sigmas = (
                    torch.rand((1, 1, vid_latent.shape[2], 1, 1), device=vid_latent.device, generator=g, dtype=vid_latent.dtype)
                    * (float(video_noise_sigma_max) - float(video_noise_sigma_min))
                    + float(video_noise_sigma_min)
                )
                vid_latent = frame_sigmas * torch.randn_like(vid_latent, generator=g) + (1.0 - frame_sigmas) * vid_latent
            vid_latent = vid_latent[:, :, :hist_len]
            if vid_latent.shape[2] < hist_len:
                pad = vid_latent[:, :, -1:].repeat(1, 1, hist_len - vid_latent.shape[2], 1, 1)
                vid_latent = torch.cat([vid_latent, pad], dim=2)
            vid_latent = comfy.utils.repeat_to_batch_size(vid_latent, batch_size)
            history_latent = vid_latent
            image_latent_prefix = history_latent[:, :, :1]

        positive, negative = _set_helios_history_values(positive, negative, history_latent, sizes, keep_first_frame, prefix_latent=image_latent_prefix)
        return io.NodeOutput(
            positive,
            negative,
            {
                "samples": latent,
                "helios_history_latent": history_latent,
                "helios_image_latent_prefix": image_latent_prefix,
            },
        )


class HeliosHistoryConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HeliosHistoryConditioning",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("history_latent"),
                io.String.Input("history_sizes", default="16,2,1"),
                io.Boolean.Input("keep_first_frame", default=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, history_latent, history_sizes, keep_first_frame) -> io.NodeOutput:
        latent = history_latent["samples"]
        if latent is None or len(latent.shape) != 5:
            return io.NodeOutput(positive, negative)
        sizes = _parse_int_list(history_sizes, [16, 2, 1])
        sizes = sorted([max(0, int(v)) for v in sizes], reverse=True)
        prefix = history_latent.get("helios_image_latent_prefix", None)
        positive, negative = _set_helios_history_values(positive, negative, latent, sizes, keep_first_frame, prefix_latent=prefix)
        return io.NodeOutput(positive, negative)


class HeliosPyramidSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HeliosPyramidSampler",
            category="sampling/video_models",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("add_noise", default=True, advanced=True),
                io.Int.Input("noise_seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Float.Input("cfg", default=5.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.String.Input("pyramid_steps", default="10,10,10"),
                io.String.Input("stage_range", default="0,0.333333,0.666667,1"),
                io.Boolean.Input("is_distilled", default=False),
                io.Boolean.Input("is_amplify_first_stage", default=False),
                io.Combo.Input("scheduler_mode", options=["euler", "unipc_bh2"]),
                io.Float.Input("gamma", default=1.0 / 3.0, min=0.0001, max=10.0, step=0.0001, round=False),
                io.Float.Input("shift", default=1.0, min=0.001, max=100.0, step=0.001, round=False, advanced=True),
                io.Boolean.Input("use_dynamic_shifting", default=False, advanced=True),
                io.Combo.Input("time_shift_type", options=["exponential", "linear"], advanced=True),
                io.Int.Input("base_image_seq_len", default=256, min=1, max=65536, advanced=True),
                io.Int.Input("max_image_seq_len", default=4096, min=1, max=65536, advanced=True),
                io.Float.Input("base_shift", default=0.5, min=0.0, max=10.0, step=0.0001, round=False, advanced=True),
                io.Float.Input("max_shift", default=1.15, min=0.0, max=10.0, step=0.0001, round=False, advanced=True),
                io.Int.Input("num_train_timesteps", default=1000, min=10, max=100000, advanced=True),
                io.String.Input("history_sizes", default="16,2,1", advanced=True),
                io.Boolean.Input("keep_first_frame", default=True, advanced=True),
                io.Int.Input("num_latent_frames_per_chunk", default=9, min=1, max=256, advanced=True),
                io.Boolean.Input("is_cfg_zero_star", default=False, advanced=True),
                io.Boolean.Input("use_zero_init", default=True, advanced=True),
                io.Int.Input("zero_steps", default=1, min=0, max=10000, advanced=True),
                io.Boolean.Input("is_skip_first_chunk", default=False, advanced=True),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        add_noise,
        noise_seed,
        cfg,
        positive,
        negative,
        latent_image,
        pyramid_steps,
        stage_range,
        is_distilled,
        is_amplify_first_stage,
        scheduler_mode,
        gamma,
        shift,
        use_dynamic_shifting,
        time_shift_type,
        base_image_seq_len,
        max_image_seq_len,
        base_shift,
        max_shift,
        num_train_timesteps,
        history_sizes,
        keep_first_frame,
        num_latent_frames_per_chunk,
        is_cfg_zero_star,
        use_zero_init,
        zero_steps,
        is_skip_first_chunk,
    ) -> io.NodeOutput:
        latent = latent_image.copy()
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent["samples"], latent.get("downscale_ratio_spacial", None))

        stage_steps = _parse_int_list(pyramid_steps, [10, 10, 10])
        stage_steps = [max(1, int(s)) for s in stage_steps]
        stage_count = len(stage_steps)
        history_sizes_list = sorted([max(0, int(v)) for v in _parse_int_list(history_sizes, [16, 2, 1])], reverse=True)

        stage_range_values = _parse_float_list(stage_range, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        if len(stage_range_values) != stage_count + 1:
            stage_range_values = [float(i) / float(stage_count) for i in range(stage_count + 1)]

        stage_tables = _helios_stage_tables(
            stage_count=stage_count,
            stage_range=stage_range_values,
            gamma=float(gamma),
            num_train_timesteps=int(num_train_timesteps),
            shift=float(shift),
        )

        b, c, t, h, w = latent_samples.shape
        chunk_t = max(1, int(num_latent_frames_per_chunk))
        chunk_count = max(1, (t + chunk_t - 1) // chunk_t)
        low_scale = 2 ** max(0, stage_count - 1)
        low_h = max(1, h // low_scale)
        low_w = max(1, w // low_scale)

        base_latent = torch.zeros((b, c, chunk_t, low_h, low_w), dtype=latent_samples.dtype, layout=latent_samples.layout, device=latent_samples.device)

        if add_noise:
            stage_latent = comfy.sample.prepare_noise(base_latent, noise_seed)
        else:
            stage_latent = torch.zeros_like(base_latent, device="cpu")

        stage_latent = stage_latent.to(base_latent.dtype).to(comfy.model_management.intermediate_device())
        euler_sampler = comfy.samplers.KSAMPLER(_helios_euler_sample)

        latents_history_short = _extract_condition_value(positive, "latents_history_short")
        latents_history_mid = _extract_condition_value(positive, "latents_history_mid")
        latents_history_long = _extract_condition_value(positive, "latents_history_long")
        image_latent_prefix = latent.get("helios_image_latent_prefix", None)
        if latents_history_short is None and "helios_history_latent" in latent:
            positive, negative = _set_helios_history_values(
                positive,
                negative,
                latent["helios_history_latent"],
                history_sizes_list,
                keep_first_frame,
                prefix_latent=image_latent_prefix,
            )
            latents_history_short = _extract_condition_value(positive, "latents_history_short")
            latents_history_mid = _extract_condition_value(positive, "latents_history_mid")
            latents_history_long = _extract_condition_value(positive, "latents_history_long")

        x0_output = {}
        generated_chunks = []
        if latents_history_short is not None and latents_history_mid is not None and latents_history_long is not None:
            rolling_history = torch.cat([latents_history_long, latents_history_mid, latents_history_short], dim=2)
        elif "helios_history_latent" in latent:
            rolling_history = latent["helios_history_latent"]
        else:
            hist_len = max(1, sum(history_sizes_list))
            rolling_history = torch.zeros((b, c, hist_len, h, w), device=latent_samples.device, dtype=latent_samples.dtype)

        for chunk_idx in range(chunk_count):
            if add_noise:
                stage_latent = comfy.sample.prepare_noise(base_latent, noise_seed + chunk_idx).to(base_latent.dtype).to(comfy.model_management.intermediate_device())
            else:
                stage_latent = torch.zeros_like(base_latent, device=comfy.model_management.intermediate_device())

            positive_chunk, negative_chunk = _set_helios_history_values(
                positive,
                negative,
                rolling_history,
                history_sizes_list,
                keep_first_frame,
                prefix_latent=image_latent_prefix,
            )
            latents_history_short = _extract_condition_value(positive_chunk, "latents_history_short")
            latents_history_mid = _extract_condition_value(positive_chunk, "latents_history_mid")
            latents_history_long = _extract_condition_value(positive_chunk, "latents_history_long")

            for stage_idx in range(stage_count):
                if stage_idx > 0:
                    stage_latent = _upsample_latent_5d(stage_latent, scale=2)

                    ori_sigma = 1.0 - float(stage_tables["ori_start_sigmas"][stage_idx])
                    alpha = 1.0 / (math.sqrt(1.0 + (1.0 / gamma)) * (1.0 - ori_sigma) + ori_sigma)
                    beta = alpha * (1.0 - ori_sigma) / math.sqrt(gamma)

                    noise = _sample_block_noise_like(stage_latent, gamma, patch_size=(1, 2, 2)).to(stage_latent)
                    stage_latent = alpha * stage_latent + beta * noise

                sigmas = _helios_stage_sigmas(
                    stage_idx=stage_idx,
                    stage_steps=stage_steps[stage_idx],
                    stage_tables=stage_tables,
                    is_distilled=is_distilled,
                    is_amplify_first_stage=is_amplify_first_stage and chunk_idx == 0,
                ).to(stage_latent.dtype)
                timesteps = _helios_stage_timesteps(
                    stage_idx=stage_idx,
                    stage_steps=stage_steps[stage_idx],
                    stage_tables=stage_tables,
                    is_distilled=is_distilled,
                    is_amplify_first_stage=is_amplify_first_stage and chunk_idx == 0,
                ).to(stage_latent.dtype)
                if use_dynamic_shifting:
                    patch_size = (1, 2, 2)
                    image_seq_len = (stage_latent.shape[-1] * stage_latent.shape[-2] * stage_latent.shape[-3]) // (patch_size[0] * patch_size[1] * patch_size[2])
                    mu = _calculate_shift(
                        image_seq_len=image_seq_len,
                        base_seq_len=base_image_seq_len,
                        max_seq_len=max_image_seq_len,
                        base_shift=base_shift,
                        max_shift=max_shift,
                    )
                    sigmas = _time_shift(sigmas, mu=mu, sigma=1.0, mode=time_shift_type).to(stage_latent.dtype)
                    tmin = torch.min(timesteps)
                    tmax = torch.max(timesteps)
                    timesteps = tmin + sigmas[:-1] * (tmax - tmin)

                indices_hidden_states, idx_short, idx_mid, idx_long = _build_helios_indices(
                    batch=stage_latent.shape[0],
                    history_sizes=history_sizes_list,
                    keep_first_frame=keep_first_frame,
                    hidden_frames=stage_latent.shape[2],
                    device=stage_latent.device,
                    dtype=stage_latent.dtype,
                )
                positive_stage = node_helpers.conditioning_set_values(positive_chunk, {"indices_hidden_states": indices_hidden_states})
                negative_stage = node_helpers.conditioning_set_values(negative_chunk, {"indices_hidden_states": indices_hidden_states})

                if latents_history_short is not None:
                    values = {"latents_history_short": latents_history_short, "indices_latents_history_short": idx_short}
                    positive_stage = node_helpers.conditioning_set_values(positive_stage, values)
                    negative_stage = node_helpers.conditioning_set_values(negative_stage, values)

                if latents_history_mid is not None:
                    values = {"latents_history_mid": latents_history_mid, "indices_latents_history_mid": idx_mid}
                    positive_stage = node_helpers.conditioning_set_values(positive_stage, values)
                    negative_stage = node_helpers.conditioning_set_values(negative_stage, values)

                if latents_history_long is not None:
                    values = {"latents_history_long": latents_history_long, "indices_latents_history_long": idx_long}
                    positive_stage = node_helpers.conditioning_set_values(positive_stage, values)
                    negative_stage = node_helpers.conditioning_set_values(negative_stage, values)

                cfg_use = 1.0 if is_distilled else cfg

                if stage_idx == 0 and add_noise:
                    noise = comfy.sample.prepare_noise(stage_latent, noise_seed + chunk_idx * 100 + stage_idx)
                    latent_start = torch.zeros_like(stage_latent)
                else:
                    sigma0 = max(float(sigmas[0].item()), 1e-6)
                    noise = (stage_latent / sigma0).to("cpu")
                    latent_start = torch.zeros_like(stage_latent)

                stage_start_for_dmd = stage_latent.clone()

                if is_distilled:
                    sampler = comfy.samplers.KSAMPLER(
                        _helios_dmd_sample,
                        extra_options={
                            "dmd_noisy_tensor": stage_start_for_dmd,
                            "dmd_sigmas": sigmas,
                            "dmd_timesteps": timesteps,
                            "all_timesteps": timesteps,
                        },
                    )
                else:
                    if scheduler_mode == "unipc_bh2":
                        sampler = comfy.samplers.ksampler("uni_pc_bh2")
                    else:
                        sampler = euler_sampler

                callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
                stage_model = model
                if is_cfg_zero_star and not is_distilled:
                    stage_model = model.clone()
                    stage_model.model_options = comfy.model_patcher.set_model_options_pre_cfg_function(
                        stage_model.model_options,
                        _build_cfg_zero_star_pre_cfg(stage_idx=stage_idx, zero_steps=zero_steps, use_zero_init=use_zero_init),
                        disable_cfg1_optimization=True,
                    )
                stage_latent = comfy.sample.sample_custom(
                    stage_model,
                    noise,
                    cfg_use,
                    sampler,
                    sigmas,
                    positive_stage,
                    negative_stage,
                    latent_start,
                    noise_mask=None,
                    callback=callback,
                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                    seed=noise_seed + chunk_idx * 100 + stage_idx,
                )

            if stage_latent.shape[-2] != h or stage_latent.shape[-1] != w:
                b2, c2, t2, h2, w2 = stage_latent.shape
                x = stage_latent.permute(0, 2, 1, 3, 4).reshape(b2 * t2, c2, h2, w2)
                x = comfy.utils.common_upscale(x, w, h, "nearest-exact", "disabled")
                stage_latent = x.reshape(b2, t2, c2, h, w).permute(0, 2, 1, 3, 4)
                stage_latent = stage_latent[:, :, :, :h, :w]

            generated_chunks.append(stage_latent)
            if keep_first_frame and ((chunk_idx == 0 and image_latent_prefix is None) or (is_skip_first_chunk and chunk_idx == 1)):
                image_latent_prefix = stage_latent[:, :, :1]
            rolling_history = torch.cat([rolling_history, stage_latent.to(rolling_history.device, rolling_history.dtype)], dim=2)
            keep_hist = max(1, sum(history_sizes_list))
            rolling_history = rolling_history[:, :, -keep_hist:]

        stage_latent = torch.cat(generated_chunks, dim=2)[:, :, :t]

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = stage_latent

        if "x0" in x0_output:
            x0_out = model.model.process_latent_out(x0_output["x0"].cpu())
            out_denoised = latent.copy()
            out_denoised["samples"] = x0_out
        else:
            out_denoised = out

        return io.NodeOutput(out, out_denoised)


class HeliosExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            HeliosImageToVideo,
            HeliosVideoToVideo,
            HeliosHistoryConditioning,
            HeliosPyramidSampler,
        ]


async def comfy_entrypoint() -> HeliosExtension:
    return HeliosExtension()

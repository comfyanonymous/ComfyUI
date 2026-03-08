import math
import torch

import nodes
import comfy.model_management
import comfy.model_patcher
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.latent_formats
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


_HELIOS_LATENT_FORMAT = comfy.latent_formats.Helios()


def _apply_helios_latent_space_noise(latent, sigma, generator=None):
    """Apply noise in Helios model latent space, then map back to VAE latent space."""
    latent_in = _HELIOS_LATENT_FORMAT.process_in(latent)
    noise = torch.randn(
        latent_in.shape,
        device=latent_in.device,
        dtype=latent_in.dtype,
        generator=generator,
    )
    noised_in = sigma * noise + (1.0 - sigma) * latent_in
    return _HELIOS_LATENT_FORMAT.process_out(noised_in).to(device=latent.device, dtype=latent.dtype)


def _tensor_stats_str(x):
    if x is None:
        return "None"
    if not torch.is_tensor(x):
        return f"non-tensor type={type(x)}"
    if x.numel() == 0:
        return f"shape={tuple(x.shape)} empty"
    xf = x.detach().to(torch.float32)
    return (
        f"shape={tuple(x.shape)} "
        f"mean={xf.mean().item():.6f} std={xf.std(unbiased=False).item():.6f} "
        f"min={xf.min().item():.6f} max={xf.max().item():.6f}"
    )


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


def _strict_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    # Reject non-bool numerics from stale workflows (e.g. 0.135).
    return bool(default)


def _extract_condition_value(conditioning, key):
    for c in conditioning:
        if len(c) < 2:
            continue
        value = c[1].get(key, None)
        if value is not None:
            return value
    return None


def _process_latent_in_preserve_zero_frames(model, latent, valid_mask=None):
    if latent is None or len(latent.shape) != 5:
        return latent
    if valid_mask is None:
        raise ValueError("Helios requires `helios_history_valid_mask` for history latent conversion.")
    vm = valid_mask
    if not torch.is_tensor(vm):
        vm = torch.tensor(vm, device=latent.device)
    vm = vm.to(device=latent.device)
    if vm.ndim == 2:
        nonzero = vm.any(dim=0)
    else:
        nonzero = vm.reshape(-1)
    nonzero = nonzero.bool()

    if nonzero.numel() == 0 or (not torch.any(nonzero)):
        return latent

    if nonzero.shape[0] != latent.shape[2]:
        raise ValueError(
            f"Helios history mask length mismatch: mask_t={nonzero.shape[0]} latent_t={latent.shape[2]}"
        )

    converted = model.model.process_latent_in(latent)
    out = latent.clone()
    out[:, :, nonzero, :, :] = converted[:, :, nonzero, :, :]
    return out


def _upsample_latent_5d(latent, scale=2):
    b, c, t, h, w = latent.shape
    x = latent.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = comfy.utils.common_upscale(x, w * scale, h * scale, "nearest", "disabled")
    x = x.reshape(b, t, c, h * scale, w * scale).permute(0, 2, 1, 3, 4)
    return x


def _downsample_latent_5d_bilinear_x2(latent):
    b, c, t, h, w = latent.shape
    x = latent.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = comfy.utils.common_upscale(x, max(1, w // 2), max(1, h // 2), "bilinear", "disabled") * 2.0
    x = x.reshape(b, t, c, max(1, h // 2), max(1, w // 2)).permute(0, 2, 1, 3, 4)
    return x


def _prepare_stage0_latent(batch, channels, frames, height, width, stage_count, add_noise, seed, dtype, layout, device):
    """Prepare initial latent for stage 0 with optional noise"""
    full_latent = torch.zeros((batch, channels, frames, height, width), dtype=dtype, layout=layout, device=device)
    if add_noise:
        full_latent = comfy.sample.prepare_noise(full_latent, seed).to(dtype)

    # Downsample to stage 0 resolution
    stage_latent = full_latent
    for _ in range(max(0, int(stage_count) - 1)):
        stage_latent = _downsample_latent_5d_bilinear_x2(stage_latent)
    return stage_latent


def _downsample_latent_for_stage0(latent, stage_count):
    """Downsample latent to stage 0 resolution."""
    stage_latent = latent
    for _ in range(max(0, int(stage_count) - 1)):
        stage_latent = _downsample_latent_5d_bilinear_x2(stage_latent)
    return stage_latent



def _sample_block_noise_like(latent, gamma, patch_size=(1, 2, 2), generator=None, seed=None):
    b, c, t, h, w = latent.shape
    _, ph, pw = patch_size
    block_size = ph * pw

    cov = torch.eye(block_size, device=latent.device) * (1.0 + gamma) - torch.ones(block_size, block_size, device=latent.device) * gamma
    cov += torch.eye(block_size, device=latent.device) * 1e-6

    h_blocks = h // ph
    w_blocks = w // pw
    block_number = b * c * t * h_blocks * w_blocks

    if generator is not None:
        # Exact sampling path (MultivariateNormal.sample), while consuming
        # from an explicit generator by temporarily swapping default RNG state.
        with torch.random.fork_rng(devices=[latent.device] if latent.device.type == "cuda" else []):
            if latent.device.type == "cuda":
                torch.cuda.set_rng_state(generator.get_state(), device=latent.device)
            else:
                torch.random.set_rng_state(generator.get_state())
            dist = torch.distributions.MultivariateNormal(
                torch.zeros(block_size, device=latent.device),
                covariance_matrix=cov,
            )
            noise = dist.sample((block_number,))
            if latent.device.type == "cuda":
                generator.set_state(torch.cuda.get_rng_state(device=latent.device))
            else:
                generator.set_state(torch.random.get_rng_state())
    elif seed is None:
        dist = torch.distributions.MultivariateNormal(torch.zeros(block_size, device=latent.device), covariance_matrix=cov)
        noise = dist.sample((block_number,))
    else:
        # Use deterministic RNG when seed is provided (for cross-framework alignment).
        with torch.random.fork_rng(devices=[latent.device] if latent.device.type == "cuda" else []):
            torch.manual_seed(int(seed))
            dist = torch.distributions.MultivariateNormal(torch.zeros(block_size, device=latent.device), covariance_matrix=cov)
            noise = dist.sample((block_number,))
    noise = noise.view(b, c, t, h_blocks, w_blocks, ph, pw)
    noise = noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(b, c, t, h, w)
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
        timesteps_per_stage[i] = torch.linspace(tmax, tmin, num_train_timesteps + 1)[:-1]
        # Fixed: use the same sigma range [0.999, 0] for all stages.
        sigmas_per_stage[i] = torch.linspace(0.999, 0.0, num_train_timesteps + 1)[:-1]


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
    sigmas = torch.linspace(float(stage_sigma_src[0].item()), float(stage_sigma_src[-1].item()), stage_steps)
    sigmas = torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype, device=sigmas.device)], dim=0)
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

        denoised_text = conds_out[0]
        denoised_uncond = conds_out[1]
        cfg = float(args.get("cond_scale", 1.0))
        x = args["input"]
        sigma = args["sigma"]

        sigma_reshaped = sigma.reshape(sigma.shape[0], *([1] * (denoised_text.ndim - 1)))
        sigma_safe = torch.clamp(sigma_reshaped, min=1e-8)

        flow_text = (x - denoised_text) / sigma_safe
        flow_uncond = (x - denoised_uncond) / sigma_safe

        positive_flat = flow_text.reshape(flow_text.shape[0], -1)
        negative_flat = flow_uncond.reshape(flow_uncond.shape[0], -1)
        alpha = _optimized_scale(positive_flat, negative_flat)
        alpha = alpha.reshape(flow_text.shape[0], *([1] * (flow_text.ndim - 1))).to(flow_text.dtype)

        if stage_idx == 0 and state["i"] <= int(zero_steps) and bool(use_zero_init):
            flow_final = flow_text * 0.0
        else:
            flow_final = flow_uncond * alpha + cfg * (flow_text - flow_uncond * alpha)

        denoised_final = x - flow_final * sigma_safe

        state["i"] += 1
        return [denoised_final, denoised_final]

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
    if prefix_latent is not None and (latent.device != prefix_latent.device or latent.dtype != prefix_latent.dtype):
        latent = latent.to(device=prefix_latent.device, dtype=prefix_latent.dtype)

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
        if prefix.device != latents_history_short_base.device or prefix.dtype != latents_history_short_base.dtype:
            prefix = prefix.to(device=latents_history_short_base.device, dtype=latents_history_short_base.dtype)
        latents_history_short = torch.cat([prefix, latents_history_short_base], dim=2)
    else:
        latents_history_short = latents_history_short_base

    idx_short = torch.arange(latents_history_short.shape[2], device=latent.device, dtype=torch.int64).unsqueeze(0).expand(latent.shape[0], -1)
    idx_mid = torch.arange(latents_history_mid.shape[2], device=latent.device, dtype=torch.int64).unsqueeze(0).expand(latent.shape[0], -1)
    idx_long = torch.arange(latents_history_long.shape[2], device=latent.device, dtype=torch.int64).unsqueeze(0).expand(latent.shape[0], -1)

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


def _build_helios_indices(batch, history_sizes, keep_first_frame, hidden_frames, device):
    sizes = list(history_sizes)
    if len(sizes) != 3:
        sizes = [16, 2, 1]
    sizes = [max(0, int(v)) for v in sizes]
    long_size, mid_size, short_base_size = sizes

    if keep_first_frame:
        total = 1 + long_size + mid_size + short_base_size + hidden_frames
        indices = torch.arange(total, device=device, dtype=torch.int64)
        splits = [1, long_size, mid_size, short_base_size, hidden_frames]
        indices_prefix, idx_long, idx_mid, idx_1x, idx_hidden = torch.split(indices, splits, dim=0)
        idx_short = torch.cat([indices_prefix, idx_1x], dim=0)
    else:
        total = long_size + mid_size + short_base_size + hidden_frames
        indices = torch.arange(total, device=device, dtype=torch.int64)
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
                io.Boolean.Input("include_history_in_output", default=False, advanced=True),
                io.Boolean.Input("debug_latent_stats", default=False, advanced=True),
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
        include_history_in_output=False,
        debug_latent_stats=False,
    ) -> io.NodeOutput:
        video_noise_sigma_min = 0.111
        video_noise_sigma_max = 0.135
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
        history_valid_mask = torch.zeros((batch_size, hist_len), device=latent.device, dtype=torch.bool)
        image_latent_prefix = None
        i2v_noise_gen = None
        noise_gen_state = None

        if start_image is not None:
            image = comfy.utils.common_upscale(start_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            img_latent = vae.encode(image[:, :, :, :3]).to(device=latent.device, dtype=torch.float32)
            img_latent = comfy.utils.repeat_to_batch_size(img_latent, batch_size)
            image_latent_prefix = img_latent[:, :, :1]

            if add_noise_to_image_latents:
                i2v_noise_gen = torch.Generator(device=img_latent.device)
                i2v_noise_gen.manual_seed(int(noise_seed))
                sigma = (
                    torch.rand((1,), device=img_latent.device, generator=i2v_noise_gen, dtype=img_latent.dtype).view(1, 1, 1, 1, 1)
                    * (float(image_noise_sigma_max) - float(image_noise_sigma_min))
                    + float(image_noise_sigma_min)
                )
                image_latent_prefix = _apply_helios_latent_space_noise(image_latent_prefix, sigma, generator=i2v_noise_gen)

            min_frames = max(1, (int(num_latent_frames_per_chunk) - 1) * 4 + 1)
            fake_video = image.repeat(min_frames, 1, 1, 1)
            fake_latents_full = vae.encode(fake_video).to(device=latent.device, dtype=torch.float32)
            fake_latent = comfy.utils.repeat_to_batch_size(fake_latents_full[:, :, -1:], batch_size)
            # when adding noise to image latents, fake_image_latents used for history are also noised.
            if add_noise_to_image_latents:
                if i2v_noise_gen is None:
                    i2v_noise_gen = torch.Generator(device=fake_latent.device)
                    i2v_noise_gen.manual_seed(int(noise_seed))
                # Keep backward compatibility with existing I2V node inputs:
                # this node exposes only image sigma controls; fake history latents
                # follow the video-noise defaults.
                fake_sigma = (
                    torch.rand((1,), device=fake_latent.device, generator=i2v_noise_gen, dtype=fake_latent.dtype).view(1, 1, 1, 1, 1)
                    * (float(video_noise_sigma_max) - float(video_noise_sigma_min))
                    + float(video_noise_sigma_min)
                )
                fake_latent = _apply_helios_latent_space_noise(fake_latent, fake_sigma, generator=i2v_noise_gen)
            history_latent[:, :, -1:] = fake_latent
            history_valid_mask[:, -1] = True
            if i2v_noise_gen is not None:
                noise_gen_state = i2v_noise_gen.get_state().clone()
            if debug_latent_stats:
                print(f"[HeliosDebug][I2V] image_latent_prefix: {_tensor_stats_str(image_latent_prefix)}")
                print(f"[HeliosDebug][I2V] fake_latent: {_tensor_stats_str(fake_latent)}")
                print(f"[HeliosDebug][I2V] history_latent: {_tensor_stats_str(history_latent)}")

        positive, negative = _set_helios_history_values(positive, negative, history_latent, sizes, keep_first_frame, prefix_latent=image_latent_prefix)
        return io.NodeOutput(
            positive,
            negative,
            {
                "samples": latent,
                "helios_history_latent": history_latent,
                "helios_image_latent_prefix": image_latent_prefix,
                "helios_history_valid_mask": history_valid_mask,
                "helios_num_frames": int(length),
                "helios_noise_gen_state": noise_gen_state,
                "helios_include_history_in_output": _strict_bool(include_history_in_output, default=False),
                "helios_debug_latent_stats": bool(debug_latent_stats),
            },
        )


class HeliosTextToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HeliosTextToVideo",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=640, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=384, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=132, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.String.Input("history_sizes", default="16,2,1", advanced=True),
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
        history_sizes="16,2,1",
    ) -> io.NodeOutput:
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1

        # Create zero latent as shape placeholder (noise will be generated in sampler)
        latent = torch.zeros(
            [batch_size, latent_channels, latent_t, height // spacial_scale, width // spacial_scale],
            device=comfy.model_management.intermediate_device(),
        )

        sizes = _parse_int_list(history_sizes, [16, 2, 1])
        if len(sizes) != 3:
            sizes = [16, 2, 1]
        sizes = sorted([max(0, int(v)) for v in sizes], reverse=True)
        hist_len = max(1, sum(sizes))
        # History latent starts as zeros (no history yet)
        history_latent = torch.zeros(
            [batch_size, latent_channels, hist_len, latent.shape[-2], latent.shape[-1]],
            device=latent.device,
            dtype=latent.dtype,
        )
        history_valid_mask = torch.zeros((batch_size, hist_len), device=latent.device, dtype=torch.bool)

        positive, negative = _set_helios_history_values(
            positive,
            negative,
            history_latent,
            sizes,
            False,
            prefix_latent=None,
        )
        return io.NodeOutput(
            positive,
            negative,
            {
                "samples": latent,
                "helios_history_latent": history_latent,
                "helios_image_latent_prefix": None,
                "helios_history_valid_mask": history_valid_mask,
                "helios_num_frames": int(length),
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
                io.Int.Input("num_latent_frames_per_chunk", default=9, min=1, max=256, advanced=True),
                io.Boolean.Input("add_noise_to_video_latents", default=True, advanced=True),
                io.Float.Input("video_noise_sigma_min", default=0.111, min=0.0, max=1.0, step=0.0001, round=False, advanced=True),
                io.Float.Input("video_noise_sigma_max", default=0.135, min=0.0, max=1.0, step=0.0001, round=False, advanced=True),
                io.Int.Input("noise_seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, advanced=True),
                io.Boolean.Input("include_history_in_output", default=True, advanced=True),
                io.Boolean.Input("debug_latent_stats", default=False, advanced=True),
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
        num_latent_frames_per_chunk=9,
        add_noise_to_video_latents=True,
        video_noise_sigma_min=0.111,
        video_noise_sigma_max=0.135,
        noise_seed=0,
        include_history_in_output=True,
        debug_latent_stats=False,
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
        history_valid_mask = torch.zeros((batch_size, hist_len), device=latent.device, dtype=torch.bool)
        image_latent_prefix = None
        noise_gen_state = None
        history_latent_output = history_latent

        if video is not None:
            video = comfy.utils.common_upscale(video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            num_frames = int(video.shape[0])
            min_frames = max(1, (int(num_latent_frames_per_chunk) - 1) * 4 + 1)
            num_chunks = num_frames // min_frames
            if num_chunks == 0:
                raise ValueError(
                    f"Video must have at least {min_frames} frames (got {num_frames} frames). "
                    f"Required: (num_latent_frames_per_chunk - 1) * 4 + 1 = ({int(num_latent_frames_per_chunk)} - 1) * 4 + 1 = {min_frames}"
                )

            first_frame = video[:1]
            first_frame_latent = vae.encode(first_frame[:, :, :, :3]).to(device=latent.device, dtype=torch.float32)

            total_valid_frames = num_chunks * min_frames
            start_frame = num_frames - total_valid_frames
            latents_chunks = []
            for i in range(num_chunks):
                chunk_start = start_frame + i * min_frames
                chunk_end = chunk_start + min_frames
                video_chunk = video[chunk_start:chunk_end]
                chunk_latents = vae.encode(video_chunk[:, :, :, :3]).to(device=latent.device, dtype=torch.float32)
                latents_chunks.append(chunk_latents)
            vid_latent = torch.cat(latents_chunks, dim=2)
            vid_latent_clean = vid_latent.clone()

            if add_noise_to_video_latents:
                g = torch.Generator(device=vid_latent.device)
                g.manual_seed(int(noise_seed))

                image_sigma = (
                    torch.rand((1,), device=first_frame_latent.device, generator=g, dtype=first_frame_latent.dtype).view(1, 1, 1, 1, 1)
                    * (float(video_noise_sigma_max) - float(video_noise_sigma_min))
                    + float(video_noise_sigma_min)
                )
                first_frame_latent = _apply_helios_latent_space_noise(first_frame_latent, image_sigma, generator=g)

                noisy_chunks = []
                num_latent_chunks = max(1, vid_latent.shape[2] // int(num_latent_frames_per_chunk))
                for i in range(num_latent_chunks):
                    chunk_start = i * int(num_latent_frames_per_chunk)
                    chunk_end = chunk_start + int(num_latent_frames_per_chunk)
                    latent_chunk = vid_latent[:, :, chunk_start:chunk_end, :, :]
                    if latent_chunk.shape[2] == 0:
                        continue
                    chunk_frames = latent_chunk.shape[2]
                    frame_sigmas = (
                        torch.rand((chunk_frames,), device=vid_latent.device, generator=g, dtype=vid_latent.dtype)
                        * (float(video_noise_sigma_max) - float(video_noise_sigma_min))
                        + float(video_noise_sigma_min)
                    ).view(1, 1, chunk_frames, 1, 1)
                    noisy_chunk = _apply_helios_latent_space_noise(latent_chunk, frame_sigmas, generator=g)
                    noisy_chunks.append(noisy_chunk)
                if len(noisy_chunks) > 0:
                    vid_latent = torch.cat(noisy_chunks, dim=2)
                noise_gen_state = g.get_state().clone()
            if debug_latent_stats:
                print(f"[HeliosDebug][V2V] first_frame_latent: {_tensor_stats_str(first_frame_latent)}")
                print(f"[HeliosDebug][V2V] video_latent: {_tensor_stats_str(vid_latent)}")

            vid_latent = comfy.utils.repeat_to_batch_size(vid_latent, batch_size)
            image_latent_prefix = comfy.utils.repeat_to_batch_size(first_frame_latent, batch_size)
            video_frames = vid_latent.shape[2]
            if video_frames < hist_len:
                keep_frames = hist_len - video_frames
                history_latent = torch.cat([history_latent[:, :, :keep_frames], vid_latent], dim=2)
                history_latent_output = torch.cat([history_latent_output[:, :, :keep_frames], comfy.utils.repeat_to_batch_size(vid_latent_clean, batch_size)], dim=2)
                history_valid_mask[:, keep_frames:] = True
            else:
                history_latent = vid_latent
                history_latent_output = comfy.utils.repeat_to_batch_size(vid_latent_clean, batch_size)
                history_valid_mask = torch.ones((batch_size, video_frames), device=latent.device, dtype=torch.bool)

        positive, negative = _set_helios_history_values(positive, negative, history_latent, sizes, keep_first_frame, prefix_latent=image_latent_prefix)
        return io.NodeOutput(
            positive,
            negative,
            {
                "samples": latent,
                "helios_history_latent": history_latent,
                "helios_history_latent_output": history_latent_output,
                "helios_image_latent_prefix": image_latent_prefix,
                "helios_history_valid_mask": history_valid_mask,
                "helios_num_frames": int(length),
                "helios_noise_gen_state": noise_gen_state,
                # Keep initial history segment and generated chunks together in sampler output.
                "helios_include_history_in_output": _strict_bool(include_history_in_output, default=True),
                "helios_debug_latent_stats": bool(debug_latent_stats),
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
                io.Boolean.Input("distilled", default=False),
                io.Boolean.Input("amplify_first_stage", default=False),
                io.Float.Input("gamma", default=1.0 / 3.0, min=0.0001, max=10.0, step=0.0001, round=False),
                io.String.Input("history_sizes", default="16,2,1", advanced=True),
                io.Boolean.Input("keep_first_frame", default=True, advanced=True),
                io.Int.Input("num_latent_frames_per_chunk", default=9, min=1, max=256, advanced=True),
                io.Boolean.Input("cfg_zero_star", default=True, advanced=True),
                io.Boolean.Input("use_zero_init", default=True, advanced=True),
                io.Int.Input("zero_steps", default=1, min=0, max=10000, advanced=True),
                io.Boolean.Input("skip_first_chunk", default=False, advanced=True),
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
        distilled,
        amplify_first_stage,
        gamma,
        history_sizes,
        keep_first_frame,
        num_latent_frames_per_chunk,
        cfg_zero_star,
        use_zero_init,
        zero_steps,
        skip_first_chunk,
    ) -> io.NodeOutput:
        # Keep these scheduler knobs internal (not exposed in node UI).
        shift = 1.0
        num_train_timesteps = 1000
        # Keep dynamic shifting always on for Helios parity; not exposed in node UI.
        use_dynamic_shifting = True
        time_shift_type = "exponential"
        base_image_seq_len = 256
        max_image_seq_len = 4096
        base_shift = 0.5
        max_shift = 1.15

        latent = latent_image.copy()
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent["samples"], latent.get("downscale_ratio_spacial", None))
        if not add_noise:
            latent_samples = _process_latent_in_preserve_zero_frames(model, latent_samples)

        stage_steps = _parse_int_list(pyramid_steps, [10, 10, 10])
        stage_steps = [max(1, int(s)) for s in stage_steps]
        stage_count = len(stage_steps)
        history_sizes_list = sorted([max(0, int(v)) for v in _parse_int_list(history_sizes, [16, 2, 1])], reverse=True)
        if not keep_first_frame and len(history_sizes_list) > 0:
            history_sizes_list[-1] += 1

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
        num_frames = int(latent.get("helios_num_frames", max(1, (int(t) - 1) * 4 + 1)))
        window_num_frames = (chunk_t - 1) * 4 + 1
        chunk_count = max(1, (num_frames + window_num_frames - 1) // window_num_frames)
        euler_sampler = comfy.samplers.KSAMPLER(_helios_euler_sample)
        target_device = comfy.model_management.get_torch_device()
        noise_gen = torch.Generator(device=target_device)
        noise_gen.manual_seed(int(noise_seed))
        noise_gen_state = latent.get("helios_noise_gen_state", None)
        if noise_gen_state is not None:
            try:
                noise_gen.set_state(noise_gen_state)
            except Exception:
                pass
        debug_latent_stats = bool(latent.get("helios_debug_latent_stats", False))

        image_latent_prefix = latent.get("helios_image_latent_prefix", None)
        history_valid_mask = latent.get("helios_history_valid_mask", None)
        if history_valid_mask is None:
            raise ValueError("Helios sampler requires `helios_history_valid_mask` in latent input.")
        history_full = None
        history_from_latent_applied = False
        if image_latent_prefix is not None:
            image_latent_prefix = model.model.process_latent_in(image_latent_prefix)
        if "helios_history_latent" in latent:
            history_in = _process_latent_in_preserve_zero_frames(model, latent["helios_history_latent"], valid_mask=history_valid_mask)
            history_full = history_in
            positive, negative = _set_helios_history_values(
                positive,
                negative,
                history_in,
                history_sizes_list,
                keep_first_frame,
                prefix_latent=image_latent_prefix,
            )
            history_from_latent_applied = True

        latents_history_short = _extract_condition_value(positive, "latents_history_short")
        latents_history_mid = _extract_condition_value(positive, "latents_history_mid")
        latents_history_long = _extract_condition_value(positive, "latents_history_long")
        if (not history_from_latent_applied) and latents_history_short is not None and latents_history_mid is not None and latents_history_long is not None:
            raise ValueError("Helios requires `helios_history_latent` + `helios_history_valid_mask`; direct history conditioning is not supported.")
        if latents_history_short is None and "helios_history_latent" in latent:
            history_in = _process_latent_in_preserve_zero_frames(model, latent["helios_history_latent"], valid_mask=history_valid_mask)
            positive, negative = _set_helios_history_values(
                positive,
                negative,
                history_in,
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
            short_base_size = history_sizes_list[-1] if len(history_sizes_list) > 0 else latents_history_short.shape[2]
            if keep_first_frame and latents_history_short.shape[2] > short_base_size:
                short_for_history = latents_history_short[:, :, -short_base_size:]
            else:
                short_for_history = latents_history_short
            rolling_history = torch.cat([latents_history_long, latents_history_mid, short_for_history], dim=2)
        elif "helios_history_latent" in latent:
            rolling_history = latent["helios_history_latent"]
            rolling_history = _process_latent_in_preserve_zero_frames(model, rolling_history, valid_mask=history_valid_mask)
        else:
            hist_len = max(1, sum(history_sizes_list))
            rolling_history = torch.zeros((b, c, hist_len, h, w), device=latent_samples.device, dtype=latent_samples.dtype)

        # When initial video latents are provided, seed history buffer
        # with those latents before the first denoising chunk.
        if not add_noise:
            hist_len = max(1, sum(history_sizes_list))
            rolling_history = rolling_history.to(device=latent_samples.device, dtype=latent_samples.dtype)
            video_latents = latent_samples
            video_frames = video_latents.shape[2]
            if video_frames < hist_len:
                keep_frames = hist_len - video_frames
                rolling_history = torch.cat([rolling_history[:, :, :keep_frames], video_latents], dim=2)
            else:
                rolling_history = video_latents[:, :, -hist_len:]

        # Keep history/prefix on the same device/dtype as denoising latents.
        rolling_history = rolling_history.to(device=target_device, dtype=torch.float32)
        if image_latent_prefix is not None:
            image_latent_prefix = image_latent_prefix.to(device=target_device, dtype=torch.float32)

        history_output = history_full if history_full is not None else rolling_history
        if "helios_history_latent_output" in latent:
            history_output = _process_latent_in_preserve_zero_frames(
                model,
                latent["helios_history_latent_output"],
                valid_mask=history_valid_mask,
            )
        history_output = history_output.to(device=target_device, dtype=torch.float32)
        if history_valid_mask is not None:
            if not torch.is_tensor(history_valid_mask):
                history_valid_mask = torch.tensor(history_valid_mask, device=target_device)
            history_valid_mask = history_valid_mask.to(device=target_device)
            if history_valid_mask.ndim == 2:
                initial_generated_latent_frames = int(history_valid_mask.any(dim=0).sum().item())
            else:
                initial_generated_latent_frames = int(history_valid_mask.reshape(-1).sum().item())
        else:
            initial_generated_latent_frames = 0
        total_generated_latent_frames = initial_generated_latent_frames

        for chunk_idx in range(chunk_count):
            # Extract chunk from input latents
            chunk_start = chunk_idx * chunk_t
            chunk_end = min(chunk_start + chunk_t, t)
            latent_chunk = latent_samples[:, :, chunk_start:chunk_end, :, :]

            # Prepare initial latent for this chunk
            if add_noise:
                noise_shape = (
                    latent_samples.shape[0],
                    latent_samples.shape[1],
                    chunk_t,
                    latent_samples.shape[3],
                    latent_samples.shape[4],
                )
                stage_latent = torch.randn(noise_shape, device=target_device, dtype=torch.float32, generator=noise_gen)
            else:
                # Use actual input latents; pad final short chunk to fixed size.
                stage_latent = latent_chunk.clone()
                if stage_latent.shape[2] < chunk_t:
                    if stage_latent.shape[2] == 0:
                        stage_latent = torch.zeros(
                            (
                                latent_samples.shape[0],
                                latent_samples.shape[1],
                                chunk_t,
                                latent_samples.shape[3],
                                latent_samples.shape[4],
                            ),
                            device=latent_samples.device,
                            dtype=torch.float32,
                        )
                    else:
                        pad = stage_latent[:, :, -1:].repeat(1, 1, chunk_t - stage_latent.shape[2], 1, 1)
                        stage_latent = torch.cat([stage_latent, pad], dim=2)
                stage_latent = stage_latent.to(dtype=torch.float32)

            # Downsample to stage 0 resolution
            for _ in range(max(0, int(stage_count) - 1)):
                stage_latent = _downsample_latent_5d_bilinear_x2(stage_latent)

            # Keep stage latents on model device for scheduler/noise path consistency.
            stage_latent = stage_latent.to(target_device)

            chunk_prefix = image_latent_prefix
            if keep_first_frame and image_latent_prefix is None and chunk_idx == 0:
                chunk_prefix = torch.zeros(
                    (
                        rolling_history.shape[0],
                        rolling_history.shape[1],
                        1,
                        rolling_history.shape[3],
                        rolling_history.shape[4],
                    ),
                    device=rolling_history.device,
                    dtype=rolling_history.dtype,
                )

            positive_chunk, negative_chunk = _set_helios_history_values(
                positive,
                negative,
                rolling_history,
                history_sizes_list,
                keep_first_frame,
                prefix_latent=chunk_prefix,
            )
            latents_history_short = _extract_condition_value(positive_chunk, "latents_history_short")
            latents_history_mid = _extract_condition_value(positive_chunk, "latents_history_mid")
            latents_history_long = _extract_condition_value(positive_chunk, "latents_history_long")
            if debug_latent_stats:
                print(f"[HeliosDebug][Sampler][chunk={chunk_idx}] latents_history_short: {_tensor_stats_str(latents_history_short)}")
                print(f"[HeliosDebug][Sampler][chunk={chunk_idx}] latents_history_mid: {_tensor_stats_str(latents_history_mid)}")
                print(f"[HeliosDebug][Sampler][chunk={chunk_idx}] latents_history_long: {_tensor_stats_str(latents_history_long)}")

            for stage_idx in range(stage_count):
                stage_latent = stage_latent.to(comfy.model_management.get_torch_device())
                sigmas = _helios_stage_sigmas(
                    stage_idx=stage_idx,
                    stage_steps=stage_steps[stage_idx],
                    stage_tables=stage_tables,
                    is_distilled=distilled,
                    is_amplify_first_stage=amplify_first_stage and chunk_idx == 0,
                ).to(device=stage_latent.device, dtype=torch.float32)
                timesteps = _helios_stage_timesteps(
                    stage_idx=stage_idx,
                    stage_steps=stage_steps[stage_idx],
                    stage_tables=stage_tables,
                    is_distilled=distilled,
                    is_amplify_first_stage=amplify_first_stage and chunk_idx == 0,
                ).to(device=stage_latent.device, dtype=torch.float32)
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
                    sigmas = _time_shift(sigmas, mu=mu, sigma=1.0, mode=time_shift_type).to(torch.float32)
                    tmin = torch.min(timesteps)
                    tmax = torch.max(timesteps)
                    timesteps = tmin + sigmas[:-1] * (tmax - tmin)
                else:
                    pass

                # Stage timesteps are computed before upsampling/renoise for stage > 0.
                if stage_idx > 0:
                    stage_latent = _upsample_latent_5d(stage_latent, scale=2)

                    ori_sigma = 1.0 - float(stage_tables["ori_start_sigmas"][stage_idx])
                    alpha = 1.0 / (math.sqrt(1.0 + (1.0 / gamma)) * (1.0 - ori_sigma) + ori_sigma)
                    beta = alpha * (1.0 - ori_sigma) / math.sqrt(gamma)

                    noise = _sample_block_noise_like(stage_latent, gamma, patch_size=(1, 2, 2), generator=noise_gen).to(stage_latent)
                    stage_latent = alpha * stage_latent + beta * noise

                indices_hidden_states, idx_short, idx_mid, idx_long = _build_helios_indices(
                    batch=stage_latent.shape[0],
                    history_sizes=history_sizes_list,
                    keep_first_frame=keep_first_frame,
                    hidden_frames=stage_latent.shape[2],
                    device=stage_latent.device,
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

                stage_time_values = {
                    "helios_stage_sigmas": sigmas,
                    "helios_stage_timesteps": timesteps,
                }
                positive_stage = node_helpers.conditioning_set_values(positive_stage, stage_time_values)
                negative_stage = node_helpers.conditioning_set_values(negative_stage, stage_time_values)

                cfg_use = 1.0 if distilled else cfg

                sigma0 = max(float(sigmas[0].item()), 1e-6)
                noise = stage_latent / sigma0
                latent_start = torch.zeros_like(stage_latent)

                stage_start_for_dmd = stage_latent.clone()

                if distilled:
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
                    sampler = euler_sampler

                callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
                stage_model = model
                if cfg_zero_star and not distilled:
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
                # sample_custom returns latent_format.process_out(samples); convert back to model-space
                # so subsequent pyramid stages and history conditioning stay in the same latent space.
                stage_latent = model.model.process_latent_in(stage_latent)

            if stage_latent.shape[-2] != h or stage_latent.shape[-1] != w:
                b2, c2, t2, h2, w2 = stage_latent.shape
                x = stage_latent.permute(0, 2, 1, 3, 4).reshape(b2 * t2, c2, h2, w2)
                x = comfy.utils.common_upscale(x, w, h, "nearest-exact", "disabled")
                stage_latent = x.reshape(b2, t2, c2, h, w).permute(0, 2, 1, 3, 4)
                stage_latent = stage_latent[:, :, :, :h, :w]

            generated_chunks.append(stage_latent)
            if keep_first_frame and ((chunk_idx == 0 and image_latent_prefix is None) or (skip_first_chunk and chunk_idx == 1)):
                image_latent_prefix = stage_latent[:, :, :1]
            rolling_history = torch.cat([rolling_history, stage_latent.to(rolling_history.device, rolling_history.dtype)], dim=2)
            keep_hist = max(1, sum(history_sizes_list))
            rolling_history = rolling_history[:, :, -keep_hist:]
            total_generated_latent_frames += stage_latent.shape[2]
            history_output = torch.cat([history_output, stage_latent.to(history_output.device, history_output.dtype)], dim=2)

        include_history_in_output = _strict_bool(latent.get("helios_include_history_in_output", False), default=False)
        if include_history_in_output and history_output is not None:
            keep_t = max(0, int(total_generated_latent_frames))
            stage_latent = history_output[:, :, -keep_t:] if keep_t > 0 else history_output[:, :, :0]
        elif len(generated_chunks) > 0:
            stage_latent = torch.cat(generated_chunks, dim=2)
        else:
            stage_latent = torch.zeros((b, c, 0, h, w), device=target_device, dtype=torch.float32)

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = model.model.process_latent_out(stage_latent)
        out["helios_chunk_decode"] = True
        out["helios_chunk_latent_frames"] = int(chunk_t)
        out["helios_chunk_count"] = int(len(generated_chunks))
        out["helios_window_num_frames"] = int(window_num_frames)
        out["helios_num_frames"] = int(num_frames)
        out["helios_prefix_latent_frames"] = int(initial_generated_latent_frames if include_history_in_output else 0)

        if "x0" in x0_output:
            x0_out = model.model.process_latent_out(x0_output["x0"].cpu())
            out_denoised = latent.copy()
            out_denoised["samples"] = x0_out
        else:
            out_denoised = out

        return io.NodeOutput(out, out_denoised)


class HeliosVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HeliosVAEDecode",
            category="latent",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, samples, vae) -> io.NodeOutput:
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]

        helios_chunk_decode = bool(samples.get("helios_chunk_decode", False))
        helios_chunk_latent_frames = int(samples.get("helios_chunk_latent_frames", 0) or 0)
        helios_prefix_latent_frames = int(samples.get("helios_prefix_latent_frames", 0) or 0)

        if (
            helios_chunk_decode
            and latent.ndim == 5
            and helios_chunk_latent_frames > 0
            and latent.shape[2] > 0
        ):
            decoded_chunks = []
            prefix_t = max(0, min(helios_prefix_latent_frames, latent.shape[2]))

            if prefix_t > 0:
                decoded_chunks.append(vae.decode(latent[:, :, :prefix_t]))

            body = latent[:, :, prefix_t:]
            for start in range(0, body.shape[2], helios_chunk_latent_frames):
                chunk = body[:, :, start:start + helios_chunk_latent_frames]
                if chunk.shape[2] == 0:
                    continue
                decoded_chunks.append(vae.decode(chunk))

            if len(decoded_chunks) > 0:
                images = torch.cat(decoded_chunks, dim=1)
            else:
                images = vae.decode(latent)
        else:
            images = vae.decode(latent)

        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return io.NodeOutput(images)


class HeliosExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            HeliosTextToVideo,
            HeliosImageToVideo,
            HeliosVideoToVideo,
            HeliosHistoryConditioning,
            HeliosPyramidSampler,
            HeliosVAEDecode,
        ]


async def comfy_entrypoint() -> HeliosExtension:
    return HeliosExtension()

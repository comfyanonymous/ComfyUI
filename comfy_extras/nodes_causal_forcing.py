"""
ComfyUI nodes for Causal Forcing autoregressive video generation.
  - LoadCausalForcingModel: load original HF/training or pre-converted checkpoints
    (auto-detects format and converts state dict at runtime)
  - CausalForcingSampler: autoregressive frame-by-frame sampling with KV cache
"""

import torch
import logging
import folder_paths
from typing_extensions import override

import comfy.model_management
import comfy.utils
import comfy.ops
import comfy.latent_formats
from comfy.model_patcher import ModelPatcher
from comfy.ldm.wan.causal_model import CausalWanModel
from comfy.ldm.wan.causal_convert import extract_state_dict
from comfy_api.latest import ComfyExtension, io

# ── Model size presets derived from Wan 2.1 configs ──────────────────────────
WAN_CONFIGS = {
    # dim → (ffn_dim, num_heads, num_layers, text_dim)
    1536:  (8960,  12, 30, 4096),  # 1.3B
    2048:  (8192,  16, 32, 4096),  # ~2B
    5120:  (13824, 40, 40, 4096),  # 14B
}


class LoadCausalForcingModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadCausalForcingModel",
            category="loaders/video_models",
            inputs=[
                io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("diffusion_models")),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL"),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name) -> io.NodeOutput:
        ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", ckpt_name)
        raw = comfy.utils.load_torch_file(ckpt_path)
        sd = extract_state_dict(raw, use_ema=True)
        del raw

        dim = sd["head.modulation"].shape[-1]
        out_dim = sd["head.head.weight"].shape[0] // 4  # prod(patch_size) * out_dim
        in_dim = sd["patch_embedding.weight"].shape[1]
        num_layers = 0
        while f"blocks.{num_layers}.self_attn.q.weight" in sd:
            num_layers += 1

        if dim in WAN_CONFIGS:
            ffn_dim, num_heads, expected_layers, text_dim = WAN_CONFIGS[dim]
        else:
            num_heads = dim // 128
            ffn_dim = sd["blocks.0.ffn.0.weight"].shape[0]
            text_dim = 4096
            logging.warning(f"CausalForcing: unknown dim={dim}, inferring num_heads={num_heads}, ffn_dim={ffn_dim}")

        cross_attn_norm = "blocks.0.norm3.weight" in sd

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        ops = comfy.ops.disable_weight_init

        model = CausalWanModel(
            model_type='t2v',
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=256,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=cross_attn_norm,
            eps=1e-6,
            device=offload_device,
            dtype=torch.bfloat16,
            operations=ops,
        )

        model.load_state_dict(sd, strict=False)
        model.eval()

        model_size = comfy.model_management.module_size(model)
        patcher = ModelPatcher(model, load_device=load_device,
                               offload_device=offload_device, size=model_size)
        patcher.model.latent_format = comfy.latent_formats.Wan21()
        return io.NodeOutput(patcher)


class CausalForcingSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CausalForcingSampler",
            category="sampling",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("num_frames", default=81, min=1, max=1024, step=4),
                io.Int.Input("num_frame_per_block", default=1, min=1, max=21),
                io.Float.Input("timestep_shift", default=5.0, min=0.1, max=20.0, step=0.1),
                io.String.Input("denoising_steps", default="1000,750,500,250"),
            ],
            outputs=[
                io.Latent.Output(display_name="LATENT"),
            ],
        )

    @classmethod
    def execute(cls, model, positive, seed, width, height,
                num_frames, num_frame_per_block, timestep_shift,
                denoising_steps) -> io.NodeOutput:

        device = comfy.model_management.get_torch_device()

        # Parse denoising steps
        step_values = [int(s.strip()) for s in denoising_steps.split(",")]

        # Build scheduler sigmas (FlowMatch with shift)
        num_train_timesteps = 1000
        raw_sigmas = torch.linspace(1.0, 0.003 / 1.002, num_train_timesteps + 1)[:-1]
        sigmas = timestep_shift * raw_sigmas / (1.0 + (timestep_shift - 1.0) * raw_sigmas)
        timesteps = sigmas * num_train_timesteps

        # Warp denoising step indices to actual timestep values
        all_timesteps = torch.cat([timesteps, torch.tensor([0.0])])
        warped_steps = all_timesteps[num_train_timesteps - torch.tensor(step_values, dtype=torch.long)]

        # Get the CausalWanModel from the patcher
        comfy.model_management.load_model_gpu(model)
        causal_model = model.model
        dtype = torch.bfloat16

        # Extract text embeddings from conditioning
        cond = positive[0][0].to(device=device, dtype=dtype)
        if cond.ndim == 2:
            cond = cond.unsqueeze(0)

        # Latent dimensions
        lat_h = height // 8
        lat_w = width // 8
        lat_t = ((num_frames - 1) // 4) + 1  # Wan VAE temporal compression
        in_channels = 16

        # Generate noise
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(1, in_channels, lat_t, lat_h, lat_w,
                            generator=generator, device="cpu").to(device=device, dtype=dtype)

        assert lat_t % num_frame_per_block == 0, \
            f"Latent frames ({lat_t}) must be divisible by num_frame_per_block ({num_frame_per_block})"
        num_blocks = lat_t // num_frame_per_block

        # Tokens per frame: (H/patch_h) * (W/patch_w) per temporal patch
        frame_seq_len = (lat_h // 2) * (lat_w // 2)  # patch_size = (1,2,2)
        max_seq_len = lat_t * frame_seq_len

        # Initialize caches
        kv_caches = causal_model.init_kv_caches(1, max_seq_len, device, dtype)
        crossattn_caches = causal_model.init_crossattn_caches(1, device, dtype)

        output = torch.zeros_like(noise)
        pbar = comfy.utils.ProgressBar(num_blocks * len(warped_steps) + num_blocks)

        current_start_frame = 0
        for block_idx in range(num_blocks):
            block_frames = num_frame_per_block
            frame_start = current_start_frame
            frame_end = current_start_frame + block_frames

            # Noise slice for this block: [B, C, block_frames, H, W]
            noisy_input = noise[:, :, frame_start:frame_end]

            # Denoising loop (e.g. 4 steps)
            for step_idx, current_timestep in enumerate(warped_steps):
                t_val = current_timestep.item()

                # Per-frame timestep tensor [B, block_frames]
                timestep_tensor = torch.full(
                    (1, block_frames), t_val, device=device, dtype=dtype)

                # Model forward
                flow_pred = causal_model.forward_block(
                    x=noisy_input,
                    timestep=timestep_tensor,
                    context=cond,
                    start_frame=current_start_frame,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                )

                # x0 = input - sigma * flow_pred
                sigma_t = _lookup_sigma(sigmas, timesteps, t_val)
                denoised = noisy_input - sigma_t * flow_pred

                if step_idx < len(warped_steps) - 1:
                    # Add noise for next step
                    next_t = warped_steps[step_idx + 1].item()
                    sigma_next = _lookup_sigma(sigmas, timesteps, next_t)
                    fresh_noise = torch.randn_like(denoised)
                    noisy_input = (1.0 - sigma_next) * denoised + sigma_next * fresh_noise

                    # Roll back KV cache end pointer so next step re-writes same positions
                    for cache in kv_caches:
                        cache["end"].fill_(cache["end"].item() - block_frames * frame_seq_len)
                else:
                    noisy_input = denoised

                pbar.update(1)

            output[:, :, frame_start:frame_end] = noisy_input

            # Cache update: forward at t=0 with clean output to fill KV cache
            with torch.no_grad():
                # Reset cache end to before this block so the t=0 pass writes clean K/V
                for cache in kv_caches:
                    cache["end"].fill_(cache["end"].item() - block_frames * frame_seq_len)

                t_zero = torch.zeros(1, block_frames, device=device, dtype=dtype)
                causal_model.forward_block(
                    x=noisy_input,
                    timestep=t_zero,
                    context=cond,
                    start_frame=current_start_frame,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                )

            pbar.update(1)
            current_start_frame += block_frames

        # Denormalize latents because VAEDecode expects raw latents.
        latent_format = comfy.latent_formats.Wan21()
        output_denorm = latent_format.process_out(output.float().cpu())
        return io.NodeOutput({"samples": output_denorm})


def _lookup_sigma(sigmas, timesteps, t_val):
    """Find the sigma corresponding to a timestep value."""
    idx = torch.argmin((timesteps - t_val).abs()).item()
    return sigmas[idx]


class CausalForcingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadCausalForcingModel,
            CausalForcingSampler,
        ]


async def comfy_entrypoint() -> CausalForcingExtension:
    return CausalForcingExtension()

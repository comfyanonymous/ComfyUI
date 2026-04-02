import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing_extensions import override

import comfy.model_patcher
import comfy.utils
import folder_paths
from comfy import model_management
from comfy_extras.rife_model.ifnet import IFNet, detect_rife_config, clear_warp_cache
from comfy_api.latest import ComfyExtension, io

FrameInterpolationModel = io.Custom("FRAME_INTERPOLATION_MODEL")


class FrameInterpolationModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FrameInterpolationModelLoader",
            display_name="Load Frame Interpolation Model",
            category="loaders",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("frame_interpolation")),
            ],
            outputs=[
                FrameInterpolationModel.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        model_path = folder_paths.get_full_path_or_raise("frame_interpolation", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)

        # Strip common prefixes (DataParallel, RIFE model wrapper)
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": "", "flownet.": ""})

        # Convert blockN.xxx keys to blocks.N.xxx if needed
        key_map = {}
        for k in sd:
            for i in range(5):
                prefix = f"block{i}."
                if k.startswith(prefix):
                    key_map[k] = f"blocks.{i}.{k[len(prefix):]}"
        if key_map:
            new_sd = {}
            for k, v in sd.items():
                new_sd[key_map.get(k, k)] = v
            sd = new_sd

        # Filter out training-only keys (teacher distillation, timestamp calibration)
        sd = {k: v for k, v in sd.items()
              if not k.startswith(("teacher.", "caltime."))}

        head_ch, channels = detect_rife_config(sd)
        model = IFNet(head_ch=head_ch, channels=channels)
        model.load_state_dict(sd)
        # RIFE is a small pixel-space model similar to VAE, bf16 produces artifacts due to low mantissa precision
        dtype = model_management.vae_dtype(device=model_management.get_torch_device(),
                                            allowed_dtypes=[torch.float16, torch.float32])
        model.eval().to(dtype)
        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=model_management.get_torch_device(),
            offload_device=model_management.unet_offload_device(),
        )
        return io.NodeOutput(patcher)


class FrameInterpolate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FrameInterpolate",
            display_name="Frame Interpolate",
            category="image/video",
            search_aliases=["rife", "frame interpolation", "slow motion", "interpolate frames"],
            inputs=[
                FrameInterpolationModel.Input("model"),
                io.Image.Input("images"),
                io.Int.Input("multiplier", default=2, min=2, max=16),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, images, multiplier) -> io.NodeOutput:
        offload_device = model_management.intermediate_device()

        num_frames = images.shape[0]
        if num_frames < 2 or multiplier < 2:
            return io.NodeOutput(images)

        model_management.load_model_gpu(model)
        device = model.load_device
        inference_model = model.model
        dtype = model.model_dtype()

        # BHWC -> BCHW
        frames = images.movedim(-1, 1).to(dtype=dtype, device=offload_device)
        _, C, H, W = frames.shape

        # Pad to multiple of 64
        pad_h = (64 - H % 64) % 64
        pad_w = (64 - W % 64) % 64
        if pad_h > 0 or pad_w > 0:
            frames = F.pad(frames, (0, pad_w, 0, pad_h), mode="reflect")

        # Count total interpolation passes for progress bar
        total_pairs = num_frames - 1
        num_interp = multiplier - 1
        total_steps = total_pairs * num_interp
        pbar = comfy.utils.ProgressBar(total_steps)
        tqdm_bar = tqdm(total=total_steps, desc="Frame interpolation")

        batch = num_interp
        t_values = [t / multiplier for t in range(1, multiplier)]
        _, _, pH, pW = frames.shape

        # Pre-allocate output tensor, pin for async GPU->CPU transfer
        total_out_frames = total_pairs * multiplier + 1
        result = torch.empty((total_out_frames, C, pH, pW), dtype=dtype, device=offload_device)
        use_pin = model_management.pin_memory(result)
        result[0] = frames[0]
        out_idx = 1

        # Pre-compute timestep tensor on device
        ts_full = torch.tensor(t_values, device=device, dtype=dtype).reshape(num_interp, 1, 1, 1)
        ts_full = ts_full.expand(-1, 1, pH, pW)

        try:
            for i in range(total_pairs):
                img0_single = frames[i:i + 1].to(device)
                img1_single = frames[i + 1:i + 2].to(device)

                j = 0
                while j < num_interp:
                    b = min(batch, num_interp - j)
                    try:
                        img0 = img0_single.expand(b, -1, -1, -1)
                        img1 = img1_single.expand(b, -1, -1, -1)
                        mids = inference_model(img0, img1, timestep=ts_full[j:j + b])
                        result[out_idx:out_idx + b].copy_(mids.to(dtype=dtype), non_blocking=use_pin)
                        out_idx += b
                        pbar.update(b)
                        tqdm_bar.update(b)
                        j += b
                    except model_management.OOM_EXCEPTION:
                        if batch <= 1:
                            raise
                        batch = max(1, batch // 2)
                        model_management.soft_empty_cache()

                result[out_idx].copy_(frames[i + 1])
                out_idx += 1
        finally:
            tqdm_bar.close()
            clear_warp_cache()
            if use_pin:
                model_management.synchronize()
                model_management.unpin_memory(result)

        # Crop padding and BCHW -> BHWC
        if pad_h > 0 or pad_w > 0:
            result = result[:, :, :H, :W]
        result = result.movedim(1, -1).clamp_(0.0, 1.0).to(dtype=model_management.intermediate_dtype())
        return io.NodeOutput(result)


class FrameInterpolationExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FrameInterpolationModelLoader,
            FrameInterpolate,
        ]


async def comfy_entrypoint() -> FrameInterpolationExtension:
    return FrameInterpolationExtension()

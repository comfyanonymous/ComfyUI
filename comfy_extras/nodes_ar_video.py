"""
ComfyUI nodes for autoregressive video generation (Causal Forcing, Self-Forcing, etc.).
  - LoadARVideoModel: load original HF/training or pre-converted checkpoints
    via the standard BaseModel + ModelPatcher pipeline
"""

import torch
import logging
import folder_paths
from typing_extensions import override

import comfy.model_management
import comfy.utils
import comfy.ops
import comfy.model_patcher
from comfy.ldm.wan.ar_convert import extract_state_dict
from comfy.supported_models import WAN21_CausalAR_T2V
from comfy_api.latest import ComfyExtension, io

# ── Model size presets derived from Wan 2.1 configs ──────────────────────────
WAN_CONFIGS = {
    # dim → (ffn_dim, num_heads, num_layers, text_dim)
    1536:  (8960,  12, 30, 4096),  # 1.3B
    2048:  (8192,  16, 32, 4096),  # ~2B
    5120:  (13824, 40, 40, 4096),  # 14B
}


class LoadARVideoModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadARVideoModel",
            category="loaders/video_models",
            inputs=[
                io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Int.Input("num_frame_per_block", default=1, min=1, max=21),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL"),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name, num_frame_per_block) -> io.NodeOutput:
        ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", ckpt_name)
        raw = comfy.utils.load_torch_file(ckpt_path)
        sd = extract_state_dict(raw, use_ema=True)
        del raw

        dim = sd["head.modulation"].shape[-1]
        out_dim = sd["head.head.weight"].shape[0] // 4
        in_dim = sd["patch_embedding.weight"].shape[1]
        num_layers = 0
        while f"blocks.{num_layers}.self_attn.q.weight" in sd:
            num_layers += 1

        if dim in WAN_CONFIGS:
            ffn_dim, num_heads, _, text_dim = WAN_CONFIGS[dim]
        else:
            num_heads = dim // 128
            ffn_dim = sd["blocks.0.ffn.0.weight"].shape[0]
            text_dim = 4096
            logging.warning(f"ARVideo: unknown dim={dim}, inferring num_heads={num_heads}, ffn_dim={ffn_dim}")

        cross_attn_norm = "blocks.0.norm3.weight" in sd

        unet_config = {
            "image_model": "wan2.1",
            "model_type": "t2v",
            "dim": dim,
            "ffn_dim": ffn_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "text_dim": text_dim,
            "cross_attn_norm": cross_attn_norm,
        }

        model_config = WAN21_CausalAR_T2V(unet_config)
        unet_dtype = comfy.model_management.unet_dtype(
            model_params=comfy.utils.calculate_parameters(sd),
            supported_dtypes=model_config.supported_inference_dtypes,
        )
        manual_cast_dtype = comfy.model_management.unet_manual_cast(
            unet_dtype,
            comfy.model_management.get_torch_device(),
            model_config.supported_inference_dtypes,
        )
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

        model = model_config.get_model(sd, "")
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        model_patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=load_device, offload_device=offload_device,
        )
        if not comfy.model_management.is_device_cpu(offload_device):
            model.to(offload_device)
        model.load_model_weights(sd, "")

        model_patcher.model_options.setdefault("transformer_options", {})["ar_config"] = {
            "num_frame_per_block": num_frame_per_block,
        }

        return io.NodeOutput(model_patcher)


class EmptyARVideoLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyARVideoLatent",
            category="latent/video",
            inputs=[
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=81, min=1, max=1024, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[
                io.Latent.Output(display_name="LATENT"),
            ],
        )

    @classmethod
    def execute(cls, width, height, length, batch_size) -> io.NodeOutput:
        lat_t = ((length - 1) // 4) + 1
        latent = torch.zeros(
            [batch_size, 16, lat_t, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return io.NodeOutput({"samples": latent})


class ARVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadARVideoModel,
            EmptyARVideoLatent,
        ]


async def comfy_entrypoint() -> ARVideoExtension:
    return ARVideoExtension()

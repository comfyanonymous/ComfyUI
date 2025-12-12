from __future__ import annotations

from ..cli_args_types import Configuration
from ..cli_args import default_configuration

MODEL_MANAGEMENT_ARGS = {
    "deterministic",
    "directml",
    "cpu",
    "disable_xformers",
    # todo: this is the default, so it will be omitted
    # "use_pytorch_cross_attention",
    "use_split_cross_attention",
    "use_quad_cross_attention",
    "supports_fp8_compute",
    "fast",
    "lowvram",
    "novram",
    "highvram",
    "gpu_only",
    "force_fp32",
    "force_fp16",
    "force_bf16",
    "reserve_vram",
    "disable_smart_memory",
    "disable_ipex_optimize",
    "disable_pinned_memory",
    "async_offload",
    "disable_async_offload",
    "force_non_blocking",
    "force_channels_last",
    "fp32_unet",
    "fp64_unet",
    "bf16_unet",
    "fp16_unet",
    "fp8_e4m3fn_unet",
    "fp8_e5m2_unet",
    "fp8_e8m0fnu_unet",
    "fp8_e4m3fn_text_enc",
    "fp8_e5m2_text_enc",
    "fp16_text_enc",
    "bf16_text_enc",
    "fp32_text_enc",
    "cpu_vae",
    "fp16_vae",
    "bf16_vae",
    "fp32_vae",
    "force_upcast_attention",
    "use_sage_attention",
    "use_flash_attention",
}


def requires_process_pool_executor(configuration: Configuration | None) -> bool:
    if configuration is None:
        return False
    
    default = default_configuration()
    for key in MODEL_MANAGEMENT_ARGS:
        # Check if key is in configuration and differs from default
        if key in configuration:
            val = configuration[key]
            # Use equality check, handling potential missing keys in default (though default should have them)
            if key not in default or val != default[key]:
                return True
    return False

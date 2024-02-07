# Define a class for your command-line arguments
import enum
from typing import Optional, List, TypedDict


class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"


class Configuration(dict):
    """
    Configuration options parsed from command-line arguments or config files.

    Attributes:
        config (Optional[str]): Path to the configuration file.
        cwd (Optional[str]): Working directory. Defaults to the current directory.
        listen (str): IP address to listen on. Defaults to "127.0.0.1".
        port (int): Port number for the server to listen on. Defaults to 8188.
        enable_cors_header (Optional[str]): Enables CORS with the specified origin.
        max_upload_size (float): Maximum upload size in MB. Defaults to 100.
        extra_model_paths_config (Optional[List[str]]): Extra model paths configuration files.
        output_directory (Optional[str]): Directory for output files.
        temp_directory (Optional[str]): Temporary directory for processing.
        input_directory (Optional[str]): Directory for input files.
        auto_launch (bool): Auto-launch UI in the default browser. Defaults to False.
        disable_auto_launch (bool): Disable auto-launching the browser.
        cuda_device (Optional[int]): CUDA device ID. None means default device.
        cuda_malloc (bool): Enable cudaMallocAsync. Defaults to True in applicable setups.
        disable_cuda_malloc (bool): Disable cudaMallocAsync.
        dont_upcast_attention (bool): Disable upcasting of attention.
        force_fp32 (bool): Force using FP32 precision.
        force_fp16 (bool): Force using FP16 precision.
        bf16_unet (bool): Use BF16 precision for UNet.
        fp16_unet (bool): Use FP16 precision for UNet.
        fp8_e4m3fn_unet (bool): Use FP8 precision (e4m3fn variant) for UNet.
        fp8_e5m2_unet (bool): Use FP8 precision (e5m2 variant) for UNet.
        fp16_vae (bool): Run the VAE in FP16 precision.
        fp32_vae (bool): Run the VAE in FP32 precision.
        bf16_vae (bool): Run the VAE in BF16 precision.
        cpu_vae (bool): Run the VAE on the CPU.
        fp8_e4m3fn_text_enc (bool): Use FP8 precision for the text encoder (e4m3fn variant).
        fp8_e5m2_text_enc (bool): Use FP8 precision for the text encoder (e5m2 variant).
        fp16_text_enc (bool): Use FP16 precision for the text encoder.
        fp32_text_enc (bool): Use FP32 precision for the text encoder.
        directml (Optional[int]): Use DirectML. -1 for auto-selection.
        disable_ipex_optimize (bool): Disable IPEX optimization for Intel GPUs.
        preview_method (LatentPreviewMethod): Method for generating previews. Defaults to "none".
        use_split_cross_attention (bool): Use split cross-attention optimization.
        use_quad_cross_attention (bool): Use sub-quadratic cross-attention optimization.
        use_pytorch_cross_attention (bool): Use PyTorch's cross-attention function.
        disable_xformers (bool): Disable xformers.
        gpu_only (bool): Run everything on the GPU.
        highvram (bool): Keep models in GPU memory.
        normalvram (bool): Default VRAM usage setting.
        lowvram (bool): Reduce UNet's VRAM usage.
        novram (bool): Minimize VRAM usage.
        cpu (bool): Use CPU for processing.
        disable_smart_memory (bool): Disable smart memory management.
        deterministic (bool): Use deterministic algorithms where possible.
        dont_print_server (bool): Suppress server output.
        quick_test_for_ci (bool): Enable quick testing mode for CI.
        windows_standalone_build (bool): Enable features for standalone Windows build.
        disable_metadata (bool): Disable saving metadata with outputs.
        multi_user (bool): Enable multi-user mode.
        plausible_analytics_base_url (Optional[str]): Base URL for server-side analytics.
        plausible_analytics_domain (Optional[str]): Domain for analytics events.
        analytics_use_identity_provider (bool): Use platform identifiers for analytics.
        write_out_config_file (bool): Enable writing out the configuration file.
    """

    config: Optional[str]
    cwd: Optional[str]
    listen: str
    port: int
    enable_cors_header: Optional[str]
    max_upload_size: float
    extra_model_paths_config: Optional[List[str]]
    output_directory: Optional[str]
    temp_directory: Optional[str]
    input_directory: Optional[str]
    auto_launch: bool
    disable_auto_launch: bool
    cuda_device: Optional[int]
    cuda_malloc: bool
    disable_cuda_malloc: bool
    dont_upcast_attention: bool
    force_fp32: bool
    force_fp16: bool
    bf16_unet: bool
    fp16_unet: bool
    fp8_e4m3fn_unet: bool
    fp8_e5m2_unet: bool
    fp16_vae: bool
    fp32_vae: bool
    bf16_vae: bool
    cpu_vae: bool
    fp8_e4m3fn_text_enc: bool
    fp8_e5m2_text_enc: bool
    fp16_text_enc: bool
    fp32_text_enc: bool
    directml: Optional[int]
    disable_ipex_optimize: bool
    preview_method: LatentPreviewMethod
    use_split_cross_attention: bool
    use_quad_cross_attention: bool
    use_pytorch_cross_attention: bool
    disable_xformers: bool
    gpu_only: bool
    highvram: bool
    normalvram: bool
    lowvram: bool
    novram: bool
    cpu: bool
    disable_smart_memory: bool
    deterministic: bool
    dont_print_server: bool
    quick_test_for_ci: bool
    windows_standalone_build: bool
    disable_metadata: bool
    multi_user: bool
    plausible_analytics_base_url: Optional[str]
    plausible_analytics_domain: Optional[str]
    analytics_use_identity_provider: bool
    write_out_config_file: bool

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

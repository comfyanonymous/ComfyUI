# Define a class for your command-line arguments
import enum
from typing import Optional, List, TypeAlias, Callable
import configargparse as argparse

ConfigurationExtender: TypeAlias = Callable[[argparse.ArgParser], Optional[argparse.ArgParser]]


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
        create_directories (bool): Creates the default models/, input/, output/ and temp/ directories, then exits.
        distributed_queue_connection_uri (Optional[str]): Servers and clients will connect to this AMQP URL to form a distributed queue and exchange prompt execution requests and progress updates.
        distributed_queue_frontend (bool): Frontends will start the web UI and connect to the provided AMQP URL to submit prompts.
        distributed_queue_worker (bool): Workers will pull requests off the AMQP URL.
        distributed_queue_name (str): This name will be used by the frontends and workers to exchange prompt requests and replies. Progress updates will be prefixed by the queue name, followed by a '.', then the user ID.
        external_address (str): Specifies a base URL for external addresses reported by the API, such as for image paths.
        verbose (bool): Shows extra output for debugging purposes such as import errors of custom nodes.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.cwd: Optional[str] = None
        self.listen: str = "127.0.0.1"
        self.port: int = 8188
        self.enable_cors_header: Optional[str] = None
        self.max_upload_size: float = 100.0
        self.extra_model_paths_config: Optional[List[str]] = []
        self.output_directory: Optional[str] = None
        self.temp_directory: Optional[str] = None
        self.input_directory: Optional[str] = None
        self.auto_launch: bool = False
        self.disable_auto_launch: bool = False
        self.cuda_device: Optional[int] = None
        self.cuda_malloc: bool = True
        self.disable_cuda_malloc: bool = False
        self.dont_upcast_attention: bool = False
        self.force_fp32: bool = False
        self.force_fp16: bool = False
        self.bf16_unet: bool = False
        self.fp16_unet: bool = False
        self.fp8_e4m3fn_unet: bool = False
        self.fp8_e5m2_unet: bool = False
        self.fp16_vae: bool = False
        self.fp32_vae: bool = False
        self.bf16_vae: bool = False
        self.cpu_vae: bool = False
        self.fp8_e4m3fn_text_enc: bool = False
        self.fp8_e5m2_text_enc: bool = False
        self.fp16_text_enc: bool = False
        self.fp32_text_enc: bool = False
        self.directml: Optional[int] = None
        self.disable_ipex_optimize: bool = False
        self.preview_method: str = "none"
        self.use_split_cross_attention: bool = False
        self.use_quad_cross_attention: bool = False
        self.use_pytorch_cross_attention: bool = False
        self.disable_xformers: bool = False
        self.gpu_only: bool = False
        self.highvram: bool = False
        self.normalvram: bool = False
        self.lowvram: bool = False
        self.novram: bool = False
        self.cpu: bool = False
        self.disable_smart_memory: bool = False
        self.deterministic: bool = False
        self.dont_print_server: bool = False
        self.quick_test_for_ci: bool = False
        self.windows_standalone_build: bool = False
        self.disable_metadata: bool = False
        self.multi_user: bool = False
        self.plausible_analytics_base_url: Optional[str] = None
        self.plausible_analytics_domain: Optional[str] = None
        self.analytics_use_identity_provider: bool = False
        self.write_out_config_file: bool = False
        self.create_directories: bool = False
        self.distributed_queue_connection_uri: Optional[str] = None
        self.distributed_queue_worker: bool = False
        self.distributed_queue_frontend: bool = False
        self.distributed_queue_name: str = "comfyui"
        self.external_address: Optional[str] = None
        for key, value in kwargs.items():
            self[key] = value

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

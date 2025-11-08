from __future__ import annotations

import enum
import logging
import os
from typing import Optional, List, Callable, Any, Union, Mapping, NamedTuple

import configargparse
import configargparse as argparse

ConfigurationExtender = Callable[[argparse.ArgParser], Optional[argparse.ArgParser]]


class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"


ConfigObserver = Callable[[str, Any], None]


def db_config() -> str:
    from .vendor.appdirs import user_data_dir

    logger = logging.getLogger(__name__)
    try:
        data_dir = user_data_dir(appname="comfyui")
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, "comfy.db")
        default_db_url = f"sqlite:///{db_path}"
    except Exception as e:
        # Fallback to an in-memory database if the user directory can't be accessed
        logger.warning(f"Could not determine user data directory for database, falling back to in-memory: {e}")
        default_db_url = "sqlite:///:memory:"
    return default_db_url


def is_valid_directory(path: str) -> str:
    """Validate if the given path is a directory, and check permissions."""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"The path '{path}' does not exist.")
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a directory.")
    if not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError(f"You do not have read permissions for '{path}'.")
    return path


class PerformanceFeature(enum.Enum):
    Fp16Accumulation = "fp16_accumulation"
    Fp8MatrixMultiplication = "fp8_matrix_mult"
    CublasOps = "cublas_ops"
    AutoTune = "autotune"


class Configuration(dict):
    """
    Configuration options parsed from command-line arguments or config files.

    Attributes:
        config_files (Optional[List[str]]): Path to the configuration file(s) that were set in the arguments.
        cwd (Optional[str]): Working directory. Defaults to the current directory. This is always treated as a base path for model files, and it will be the place where model files are downloaded.
        base_paths (Optional[list[str]]): Additional base paths for custom nodes, models and inputs.
        base_directory (Optional[str]): Set the ComfyUI base directory for models, custom_nodes, input, output, temp, and user directories.
        listen (str): IP address to listen on. Defaults to "127.0.0.1".
        port (int): Port number for the server to listen on. Defaults to 8188.
        enable_cors_header (Optional[str]): Enables CORS with the specified origin.
        max_upload_size (float): Maximum upload size in MB. Defaults to 100.
        extra_model_paths_config (Optional[List[str]]): Extra model paths configuration files.
        output_directory (Optional[str]): Directory for output files. This can also be a relative path to the cwd or current working directory.
        temp_directory (Optional[str]): Temporary directory for processing.
        input_directory (Optional[str]): Directory for input files. When this is a relative path, it will be looked up relative to the cwd (current working directory) and all of the base_paths.
        auto_launch (bool): Auto-launch UI in the default browser. Defaults to False.
        cuda_device (Optional[int]): CUDA device ID. None means default device.
        cuda_malloc (bool): Enable cudaMallocAsync. Defaults to True in applicable setups.
        disable_cuda_malloc (bool): Disable cudaMallocAsync.
        dont_upcast_attention (bool): Disable upcasting of attention.
        force_upcast_attention (bool): Force upcasting of attention.
        force_fp32 (bool): Force using FP32 precision.
        force_fp16 (bool): Force using FP16 precision.
        force_bf16 (bool): Force using BF16 precision.
        bf16_unet (bool): Use BF16 precision for UNet.
        fp16_unet (bool): Use FP16 precision for UNet.
        fp32_unet (bool): Run the diffusion model in fp32.
        fp64_unet (bool): Run the diffusion model in fp64.
        fp8_e4m3fn_unet (bool): Store unet weights in fp8_e4m3fn
        fp8_e5m2_unet (bool): Store unet weights in fp8_e5m2.
        fp16_vae (bool): Run the VAE in FP16 precision.
        fp32_vae (bool): Run the VAE in FP32 precision.
        bf16_vae (bool): Run the VAE in BF16 precision.
        cpu_vae (bool): Run the VAE on the CPU.
        fp8_e4m3fn_text_enc (bool): Use FP8 precision for the text encoder (e4m3fn variant).
        fp8_e5m2_text_enc (bool): Use FP8 precision for the text encoder (e5m2 variant).
        fp16_text_enc (bool): Use FP16 precision for the text encoder.
        fp32_text_enc (bool): Use FP32 precision for the text encoder.
        oneapi_device_selector (Optional[str]): Sets the oneAPI device(s) this instance will use.
        directml (Optional[int]): Use DirectML. -1 for auto-selection.
        disable_ipex_optimize (bool): Disable IPEX optimization for Intel GPUs.
        preview_method (LatentPreviewMethod): Method for generating previews. Defaults to "auto".
        cache_lru (int): Use LRU caching with a maximum of N node results cached. May use more RAM/VRAM.
        use_split_cross_attention (bool): Use split cross-attention optimization.
        use_quad_cross_attention (bool): Use sub-quadratic cross-attention optimization.
        use_pytorch_cross_attention (bool): Use PyTorch's cross-attention function.
        use_sage_attention (bool): Use Sage Attention
        use_flash_attention (bool): Use FlashAttention
        disable_xformers (bool): Disable xformers.
        gpu_only (bool): Run everything on the GPU.
        highvram (bool): Keep models in GPU memory.
        normalvram (bool): Default VRAM usage setting.
        lowvram (bool): Reduce UNet's VRAM usage.
        novram (bool): Minimize VRAM usage.
        cpu (bool): Use CPU for processing.
        fast (set[PerformanceFeature]): Enable some untested and potentially quality deteriorating optimizations. Pass a list specific optimizations if you only want to enable specific ones. Current valid optimizations: fp16_accumulation fp8_matrix_mult cublas_ops
        reserve_vram (Optional[float]): Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reserved depending on your OS
        disable_smart_memory (bool): Disable smart memory management.
        deterministic (bool): Use deterministic algorithms where possible.
        quick_test_for_ci (bool): Enable quick testing mode for CI.
        windows_standalone_build (bool): Enable features for standalone Windows build.
        disable_metadata (bool): Disable saving metadata with outputs.
        disable_all_custom_nodes (bool): Disable loading all custom nodes.
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
        logging_level (str): Specifies a log level
        disable_known_models (bool): Disables automatic downloads of known models and prevents them from appearing in the UI.
        max_queue_size (int): The API will reject prompt requests if the queue's size exceeds this value.
        otel_service_name (str): The name of the service or application that is generating telemetry data. Default: "comfyui".
        otel_service_version (str): The version of the service or application that is generating telemetry data. Default: "0.0.1".
        otel_exporter_otlp_endpoint (Optional[str]): A base endpoint URL for any signal type, with an optionally-specified port number. Helpful for when you're sending more than one signal to the same endpoint and want one environment variable to control the endpoint.
        force_channels_last (bool): Force channels last format when inferencing the models.
        force_hf_local_dir_mode (bool): Download repos from huggingface.co to the models/huggingface directory with the "local_dir" argument instead of models/huggingface_cache with the "cache_dir" argument, recreating the traditional file structure.
        executor_factory (str): Either ThreadPoolExecutor or ProcessPoolExecutor, defaulting to ThreadPoolExecutor
        preview_size (int): Sets the maximum preview size for sampler nodes. Defaults to 512.
        openai_api_key (str): Configures the OpenAI API Key for the OpenAI nodes. Visit https://platform.openai.com/api-keys to create this key.
        ideogram_api_key (str): Configures the Ideogram API Key for the Ideogram nodes. Visit https://ideogram.ai/manage-api to create this key.
        anthropic_api_key (str): Configures the Anthropic API key for its nodes related to Claude functionality. Visit https://console.anthropic.com/settings/keys to create this key.
        user_directory (Optional[str]): Set the ComfyUI user directory with an absolute path.
        log_stdout (bool): Send normal process output to stdout instead of stderr (default)
        panic_when (list[str]): List of fully qualified exception class names to panic (sys.exit(1)) when a workflow raises it.
        enable_compress_response_body (bool): Enable compressing response body.
        workflows (list[str]): Execute the API workflow(s) specified in the provided files. For each workflow, its outputs will be printed to a line to standard out. Application logging will be redirected to standard error. Use `-` to signify standard in.
        fp8_e8m0fnu_unet (bool): Store unet weights in fp8_e8m0fnu.
        bf16_text_enc (bool): Store text encoder weights in bf16.
        supports_fp8_compute (bool): ComfyUI will act like if the device supports fp8 compute.
        cache_classic (bool): WARNING: Unused. Use the old style (aggressive) caching.
        cache_none (bool): Reduced RAM/VRAM usage at the expense of executing every node for each run.
        async_offload (bool): Use async weight offloading.
        force_non_blocking (bool): Force ComfyUI to use non-blocking operations for all applicable tensors. This may improve performance on some non-Nvidia systems but can cause issues with some workflows.
        default_hashing_function (str): Allows you to choose the hash function to use for duplicate filename / contents comparison. Default is sha256.
        mmap_torch_files (bool): Use mmap when loading ckpt/pt files.
        disable_mmap (bool): Don't use mmap when loading safetensors.
        dont_print_server (bool): Don't print server output.
        disable_api_nodes (bool): Disable loading all api nodes.
        front_end_version (str): Specifies the version of the frontend to be used.
        front_end_root (Optional[str]): The local filesystem path to the directory where the frontend is located. Overrides --front-end-version.
        comfy_api_base (str): Set the base URL for the ComfyUI API. (default: https://api.comfy.org)
        database_url (str): Specify the database URL, e.g. for an in-memory database you can use 'sqlite:///:memory:'.
        blacklist_custom_nodes (list[str]): Specify custom node folders to never load. Accepts shell-style globs.
        whitelist_custom_nodes (list[str]): Specify custom node folders to load even when --disable-all-custom-nodes is enabled.
        default_device (Optional[int]): Set the id of the default device, all other devices will stay visible.
        block_runtime_package_installation (Optional[bool]): When set, custom nodes like ComfyUI Manager, Easy Use, Nunchaku and others will not be able to use pip or uv to install packages at runtime (experimental).
        enable_eval (Optional[bool]): Enable nodes that can evaluate Python code in workflows.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._observers: List[ConfigObserver] = []
        self.config_files = []
        self.cwd: Optional[str] = None
        self.base_paths: list[str] = []
        self.base_directory: Optional[str] = None
        self.listen: str = "127.0.0.1"
        self.port: int = 8188
        self.enable_cors_header: Optional[str] = None
        self.enable_compress_response_body: bool = False
        self.max_upload_size: float = 100.0
        self.extra_model_paths_config: Optional[List[str]] = []
        self.output_directory: Optional[str] = None
        self.temp_directory: Optional[str] = None
        self.input_directory: Optional[str] = None
        self.auto_launch: bool = False
        self.cuda_device: Optional[int] = None
        self.cuda_malloc: bool = True
        self.disable_cuda_malloc: bool = True
        self.dont_upcast_attention: bool = False
        self.force_upcast_attention: bool = False
        self.force_fp32: bool = False
        self.force_fp16: bool = False
        self.force_bf16: bool = False
        self.bf16_unet: bool = False
        self.fp16_unet: bool = False
        self.fp32_unet: bool = False
        self.fp64_unet: bool = False
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
        self.preview_method: LatentPreviewMethod = LatentPreviewMethod.Auto
        self.use_split_cross_attention: bool = False
        self.use_quad_cross_attention: bool = False
        self.use_pytorch_cross_attention: bool = False
        self.use_sage_attention: bool = False
        self.use_flash_attention: bool = False
        self.disable_xformers: bool = False
        self.gpu_only: bool = False
        self.highvram: bool = False
        self.normalvram: bool = False
        self.lowvram: bool = False
        self.novram: bool = False
        self.cpu: bool = False
        self.fast: set[PerformanceFeature] = set()
        self.reserve_vram: Optional[float] = None
        self.disable_smart_memory: bool = False
        self.deterministic: bool = False
        self.dont_print_server: bool = False
        self.quick_test_for_ci: bool = False
        self.windows_standalone_build: bool = False
        self.disable_metadata: bool = False
        self.disable_all_custom_nodes: bool = False
        self.blacklist_custom_nodes: list[str] = []
        self.whitelist_custom_nodes: list[str] = []
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
        self.disable_known_models: bool = False
        self.max_queue_size: int = 65536
        self.force_channels_last: bool = False
        self.force_hf_local_dir_mode = False
        self.preview_size: int = 512
        self.logging_level: str = "INFO"
        self.oneapi_device_selector: Optional[str] = None
        self.log_stdout: bool = False

        # from guill
        self.cache_lru: int = 0

        # from opentracing docs
        self.otel_service_name: str = "comfyui"
        self.otel_service_version: str = "0.0.1"
        self.otel_exporter_otlp_endpoint: Optional[str] = None
        self.executor_factory: str = "ThreadPoolExecutor"
        self.openai_api_key: Optional[str] = None
        self.ideogram_api_key: Optional[str] = None
        self.anthropic_api_key: Optional[str] = None
        self.user_directory: Optional[str] = None
        self.panic_when: list[str] = []
        self.workflows: list[str] = []

        self.fp8_e8m0fnu_unet: bool = False
        self.bf16_text_enc: bool = False
        self.supports_fp8_compute: bool = False
        self.cache_classic: bool = False
        self.cache_none: bool = False
        self.async_offload: bool = False
        self.force_non_blocking: bool = False
        self.default_hashing_function: str = 'sha256'
        self.mmap_torch_files: bool = False
        self.disable_mmap: bool = False
        self.disable_api_nodes: bool = False
        self.front_end_version: str = "comfyanonymous/ComfyUI@latest"
        self.front_end_root: Optional[str] = None
        self.comfy_api_base: str = "https://api.comfy.org"
        self.database_url: str = db_config()
        self.default_device: Optional[int] = None
        self.block_runtime_package_installation = None
        self.enable_eval: Optional[bool] = False

        for key, value in kwargs.items():
            self[key] = value
        # this must always be last

    def __getattr__(self, item):
        if item not in self:
            return None
        return self[item]

    def __setattr__(self, key, value):
        if key != "_observers":
            old_value = self.get(key)
            self[key] = value
            if old_value != value:
                self._notify_observers(key, value)
        else:
            super().__setattr__(key, value)

    def update(self, __m: Union[Mapping[str, Any], None] = None, **kwargs):
        if __m is None:
            __m = {}
        changes = {}
        for k, v in dict(__m, **kwargs).items():
            if k not in self or self[k] != v:
                changes[k] = v
        super().update(__m, **kwargs)
        for k, v in changes.items():
            self._notify_observers(k, v)
        # make this more pythonic
        return self

    def register_observer(self, observer: ConfigObserver):
        self._observers.append(observer)

    def unregister_observer(self, observer: ConfigObserver):
        self._observers.remove(observer)

    def _notify_observers(self, key, value):
        for observer in self._observers:
            observer(key, value)

    def __getstate__(self):
        state = self.copy()
        if "_observers" in state:
            state.pop("_observers")
        return state

    def __setstate__(self, state):
        self._observers = []
        self.update(state)

    @property
    def verbose(self) -> str:
        return self.logging_level

    @verbose.setter
    def verbose(self, value):
        if isinstance(value, bool):
            self.logging_level = "DEBUG"
        else:
            self.logging_level = value


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums in a case-insensitive manner.
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        self._enum = enum_type

        # Generate choices from the Enum for the help message
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        # We handle choices ourselves for case-insensitivity, so remove it before calling super.
        if "choices" in kwargs:
            del kwargs["choices"]

        super(EnumAction, self).__init__(**kwargs)
        self._choices = choices

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum, case-insensitively
        value_lower = values.lower()
        for member in self._enum:
            if member.value.lower() == value_lower:
                setattr(namespace, self.dest, member)
                return

        # If no match found, raise an error
        msg = f"invalid choice: {values!r} (choose from {', '.join(self._choices)})"
        raise argparse.ArgumentError(self, msg)


class ParsedArgs(NamedTuple):
    namespace: configargparse.Namespace
    unknown_args: list[str]
    config_file_paths: list[str]


class EnhancedConfigArgParser(configargparse.ArgParser):
    def parse_known_args_with_config_files(self, args=None, namespace=None, **kwargs) -> ParsedArgs:
        # usually the single method open
        prev_open_func = self._config_file_open_func
        config_files: List[str] = []

        try:
            self._config_file_open_func = lambda path: config_files.append(path)
            self._open_config_files(args)
        finally:
            self._config_file_open_func = prev_open_func

        namespace, unknown_args = super().parse_known_args(args, namespace, **kwargs)
        return ParsedArgs(namespace, unknown_args, config_files)


class FlattenAndAppendAction(argparse.Action):
    """
    Custom action to handle comma-separated values and multiple invocations
    of the same argument, flattening them into a single list.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        if items is None:
            items = []
        else:
            # Make a copy if it's not the first time, to avoid modifying the default.
            items = items[:]

        # 'values' will be a list of strings because of nargs='+'
        for value in values:
            # Split comma-separated strings and add them to the list
            items.extend(item.strip() for item in value.split(','))

        # Set the flattened list back to the namespace.
        setattr(namespace, self.dest, items)

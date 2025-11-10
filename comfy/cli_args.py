from __future__ import annotations

import logging
import sys
from importlib.metadata import entry_points
from types import ModuleType
from typing import Optional

import configargparse as argparse

from . import __version__
from .cli_args_types import LatentPreviewMethod, Configuration, ConfigurationExtender, EnumAction, \
    EnhancedConfigArgParser, PerformanceFeature, is_valid_directory, db_config, FlattenAndAppendAction
from .component_model.module_property import create_module_properties

# todo: move this
DEFAULT_VERSION_STRING = "comfyanonymous/ComfyUI@latest"

logger = logging.getLogger(__name__)

args: Configuration

_module_properties = create_module_properties()


def _create_parser() -> EnhancedConfigArgParser:
    parser = EnhancedConfigArgParser(default_config_files=['config.yaml', 'config.json', 'config.cfg', 'config.ini'],
                                     auto_env_var_prefix='COMFYUI_',
                                     args_for_setting_config_path=["-c", "--config"],
                                     add_env_var_help=True, add_config_file_help=True, add_help=True,
                                     args_for_writing_out_config_file=["--write-out-config-file"])

    parser.add_argument('-w', "--cwd", type=str, default=None,
                        help="Specify the working directory. If not set, this is the current working directory. models/, input/, output/ and other directories will be located here by default.")
    parser.add_argument("--base-paths", type=str, nargs='+', default=[], action=FlattenAndAppendAction, help="Additional base paths for custom nodes, models and inputs.")
    parser.add_argument('-H', "--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0,::",
                        help="Specify the IP address to listen on (default: 127.0.0.1). You can give a list of ip addresses by separating them with a comma like: 127.2.2.2,127.3.3.3 If --listen is provided without an argument, it defaults to 0.0.0.0,:: (listens on all ipv4 and ipv6)")
    parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
    parser.add_argument("--enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*",
                        help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'.")
    parser.add_argument("--max-upload-size", type=float, default=100, help="Set the maximum upload size in MB.")
    parser.add_argument("--base-directory", type=str, default=None, help="Set the ComfyUI base directory for models, custom_nodes, input, output, temp, and user directories.")
    parser.add_argument("--extra-model-paths-config", type=str, default=[], metavar="PATH", nargs='+',
                        action=FlattenAndAppendAction, help="Load one or more extra_model_paths.yaml files. Can be specified multiple times or as a comma-separated list.")
    parser.add_argument("--output-directory", type=str, default=None, help="Set the ComfyUI output directory. Overrides --base-directory.")
    parser.add_argument("--temp-directory", type=str, default=None,
                        help="Set the ComfyUI temp directory (default is in the ComfyUI directory). Overrides --base-directory.")
    parser.add_argument("--input-directory", type=str, default=None, help="Set the ComfyUI input directory. Overrides --base-directory.")
    parser.add_argument("--auto-launch", action="store_true",
                        help="Automatically launch ComfyUI in the default browser.")
    parser.add_argument("--disable-auto-launch", action="store_true", help="Disable auto launching the browser.")
    parser.add_argument("--cuda-device", type=int, default=None, metavar="DEVICE_ID",
                        help="Set the id of the cuda device this instance will use. All other devices will not be visible.")
    parser.add_argument("--default-device", type=int, default=None, metavar="DEFAULT_DEVICE_ID", help="Set the id of the default device, all other devices will stay visible.")
    cm_group = parser.add_mutually_exclusive_group()
    cm_group.add_argument("--cuda-malloc", action="store_true",
                          help="Enable cudaMallocAsync (enabled by default for torch 2.0 and up).")
    cm_group.add_argument("--disable-cuda-malloc", action="store_true", default=True, help="Disable cudaMallocAsync.")

    fp_group = parser.add_mutually_exclusive_group()
    fp_group.add_argument("--force-fp32", action="store_true",
                          help="Force fp32 (If this makes your GPU work better please report it).")
    fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")
    fp_group.add_argument("--force-bf16", action="store_true", help="Force bf16.")

    fpunet_group = parser.add_mutually_exclusive_group()
    fpunet_group.add_argument("--fp32-unet", action="store_true", help="Run the diffusion model in fp32.")
    fpunet_group.add_argument("--fp64-unet", action="store_true", help="Run the diffusion model in fp64.")
    fpunet_group.add_argument("--bf16-unet", action="store_true",
                              help="Run the diffusion model in bf16.")
    fpunet_group.add_argument("--fp16-unet", action="store_true", help="Run the diffusion model in fp16")
    fpunet_group.add_argument("--fp8_e4m3fn-unet", action="store_true", help="Store unet weights in fp8_e4m3fn.")
    fpunet_group.add_argument("--fp8_e5m2-unet", action="store_true", help="Store unet weights in fp8_e5m2.")
    fpunet_group.add_argument("--fp8_e8m0fnu-unet", action="store_true", help="Store unet weights in fp8_e8m0fnu.")

    fpvae_group = parser.add_mutually_exclusive_group()
    fpvae_group.add_argument("--fp16-vae", action="store_true", help="Run the VAE in fp16, might cause black images.")
    fpvae_group.add_argument("--fp32-vae", action="store_true", help="Run the VAE in full precision fp32.")
    fpvae_group.add_argument("--bf16-vae", action="store_true", help="Run the VAE in bf16.")

    parser.add_argument("--cpu-vae", action="store_true", help="Run the VAE on the CPU.")

    fpte_group = parser.add_mutually_exclusive_group()
    fpte_group.add_argument("--fp8_e4m3fn-text-enc", action="store_true",
                            help="Store text encoder weights in fp8 (e4m3fn variant).")
    fpte_group.add_argument("--fp8_e5m2-text-enc", action="store_true",
                            help="Store text encoder weights in fp8 (e5m2 variant).")
    fpte_group.add_argument("--fp16-text-enc", action="store_true", help="Store text encoder weights in fp16.")
    fpte_group.add_argument("--fp32-text-enc", action="store_true", help="Store text encoder weights in fp32.")
    fpte_group.add_argument("--bf16-text-enc", action="store_true", help="Store text encoder weights in bf16.")

    parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1,
                        help="Use torch-directml.")

    parser.add_argument("--oneapi-device-selector", type=str, default=None, metavar="SELECTOR_STRING", help="Sets the oneAPI device(s) this instance will use.")
    parser.add_argument("--disable-ipex-optimize", action="store_true",
                        help="Disables ipex.optimize default when loading models with Intel's Extension for Pytorch.")
    parser.add_argument("--supports-fp8-compute", action="store_true", help="ComfyUI will act like if the device supports fp8 compute.")

    parser.add_argument("--preview-method", type=LatentPreviewMethod, default=LatentPreviewMethod.Auto,
                        help="Default preview method for sampler nodes.", action=EnumAction)

    parser.add_argument("--preview-size", type=int, default=512, help="Sets the maximum preview size for sampler nodes.")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--cache-classic", action="store_true", help="WARNING: Unused. Use the old style (aggressive) caching.")
    cache_group.add_argument("--cache-lru", type=int, default=0, help="Use LRU caching with a maximum of N node results cached. May use more RAM/VRAM.")
    cache_group.add_argument("--cache-none", action="store_true", help="Reduced RAM/VRAM usage at the expense of executing every node for each run.")
    attn_group = parser.add_mutually_exclusive_group()
    attn_group.add_argument("--use-split-cross-attention", action="store_true",
                            help="Use the split cross attention optimization. Ignored when xformers is used.")
    attn_group.add_argument("--use-quad-cross-attention", action="store_true",
                            help="Use the sub-quadratic cross attention optimization . Ignored when xformers is used.")
    attn_group.add_argument("--use-pytorch-cross-attention", action="store_true",
                            help="Use the new pytorch 2.0 cross attention function (default).", default=True)
    attn_group.add_argument("--use-sage-attention", action="store_true", help="Use sage attention.")
    attn_group.add_argument("--use-flash-attention", action="store_true", help="Use FlashAttention.")

    parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")

    upcast = parser.add_mutually_exclusive_group()
    upcast.add_argument("--force-upcast-attention", action="store_true", help="Force enable attention upcasting, please report if it fixes black images.")
    upcast.add_argument("--dont-upcast-attention", action="store_true", help="Disable all upcasting of attention. Should be unnecessary except for debugging.")
    vram_group = parser.add_mutually_exclusive_group()
    vram_group.add_argument("--gpu-only", action="store_true",
                            help="Store and run everything (text encoders/CLIP models, etc... on the GPU).")
    vram_group.add_argument("--highvram", action="store_true",
                            help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.")
    vram_group.add_argument("--normalvram", action="store_true",
                            help="Used to force normal vram use if lowvram gets automatically enabled.")
    vram_group.add_argument("--lowvram", action="store_true", help="Split the unet in parts to use less vram.")
    vram_group.add_argument("--novram", action="store_true", help="When lowvram isn't enough.")
    vram_group.add_argument("--cpu", action="store_true", help="To use the CPU for everything (slow).")

    parser.add_argument("--reserve-vram", type=float, default=None, help="Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reserved depending on your OS.")
    parser.add_argument("--async-offload", action="store_true", help="Use async weight offloading.")
    parser.add_argument("--force-non-blocking", action="store_true", help="Force ComfyUI to use non-blocking operations for all applicable tensors. This may improve performance on some non-Nvidia systems but can cause issues with some workflows.")
    parser.add_argument("--default-hashing-function", type=str, choices=['md5', 'sha1', 'sha256', 'sha512'], default='sha256', help="Allows you to choose the hash function to use for duplicate filename / contents comparison. Default is sha256.")
    parser.add_argument("--disable-smart-memory", action="store_true",
                        help="Force ComfyUI to aggressively offload to regular ram instead of keeping models in VRAM when it can.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.")

    parser.add_argument("--fast", nargs="*", type=PerformanceFeature, help=f"Enable some untested and potentially quality deteriorating optimizations. Pass a list specific optimizations if you only want to enable specific ones. Current valid optimizations: {' '.join([f.value for f in PerformanceFeature])}", default=set())

    parser.add_argument("--mmap-torch-files", action="store_true", help="Use mmap when loading ckpt/pt files.")
    parser.add_argument("--disable-mmap", action="store_true", help="Don't use mmap when loading safetensors.")

    parser.add_argument("--dont-print-server", action="store_true", help="Don't print server output.")
    parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI. Raises an error if nodes cannot be imported,")
    parser.add_argument("--windows-standalone-build", default=hasattr(sys, 'frozen') and getattr(sys, 'frozen'),
                        action="store_true",
                        help="Windows standalone build: Enable convenient things that most people using the standalone windows build will probably enjoy (like auto opening the page on startup).")

    parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata in files.")
    parser.add_argument("--disable-all-custom-nodes", action="store_true", help="Disable loading all custom nodes.")
    parser.add_argument("--whitelist-custom-nodes", type=str, action=FlattenAndAppendAction, nargs='+', default=[], help="Specify custom node folders to load even when --disable-all-custom-nodes is enabled.")
    parser.add_argument("--blacklist-custom-nodes", type=str, action=FlattenAndAppendAction, nargs='+', default=[], help="Specify custom node folders to never load. Accepts shell-style globs.")
    parser.add_argument("--disable-api-nodes", action="store_true", help="Disable loading all api nodes.")
    parser.add_argument("--enable-eval", action="store_true", help="Enable nodes that can evaluate Python code in workflows.")

    parser.add_argument("--multi-user", action="store_true", help="Enables per-user storage.")
    parser.add_argument("--create-directories", action="store_true",
                        help="Creates the default models/, input/, output/ and temp/ directories, then exits.")
    parser.add_argument("--log-stdout", action="store_true", help="Send normal process output to stdout instead of stderr (default).")

    parser.add_argument("--plausible-analytics-base-url", required=False,
                        help="Enables server-side analytics events sent to the provided URL.")
    parser.add_argument("--plausible-analytics-domain", required=False,
                        help="Specifies the domain name for analytics events.")
    parser.add_argument("--analytics-use-identity-provider", action="store_true",
                        help="Uses platform identifiers for unique visitor analytics.")
    parser.add_argument("--distributed-queue-connection-uri", type=str, default=None,
                        help="EXAMPLE: \"amqp://guest:guest@127.0.0.1\" - Servers and clients will connect to this AMPQ URL to form a distributed queue and exchange prompt execution requests and progress updates.")
    parser.add_argument(
        '--distributed-queue-worker',
        required=False,
        action="store_true",
        help='Workers will pull requests off the AMQP URL.'
    )
    parser.add_argument(
        '--distributed-queue-frontend',
        required=False,
        action="store_true",
        help='Frontends will start the web UI and connect to the provided AMQP URL to submit prompts.'
    )
    parser.add_argument("--distributed-queue-name", type=str, default="comfyui",
                        help="This name will be used by the frontends and workers to exchange prompt requests and replies. Progress updates will be prefixed by the queue name, followed by a '.', then the user ID")
    parser.add_argument("--external-address", required=False,
                        help="Specifies a base URL for external addresses reported by the API, such as for image paths.")
    parser.add_argument("--logging-level", type=lambda x: str(x).upper(), default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level')
    parser.add_argument("--disable-known-models", action="store_true", help="Disables automatic downloads of known models and prevents them from appearing in the UI.")
    parser.add_argument("--max-queue-size", type=int, default=65536, help="The API will reject prompt requests if the queue's size exceeds this value.")
    # tracing
    parser.add_argument("--otel-service-name", type=str, default="comfyui", env_var="OTEL_SERVICE_NAME", help="The name of the service or application that is generating telemetry data.")
    parser.add_argument("--otel-service-version", type=str, default=__version__, env_var="OTEL_SERVICE_VERSION", help="The version of the service or application that is generating telemetry data.")
    parser.add_argument("--otel-exporter-otlp-endpoint", type=str, default=None, env_var="OTEL_EXPORTER_OTLP_ENDPOINT", help="A base endpoint URL for any signal type, with an optionally-specified port number. Helpful for when you're sending more than one signal to the same endpoint and want one environment variable to control the endpoint.")
    parser.add_argument("--force-channels-last", action="store_true", help="Force channels last format when inferencing the models.")
    parser.add_argument("--force-hf-local-dir-mode", action="store_true", help="Download repos from huggingface.co to the models/huggingface directory with the \"local_dir\" argument instead of models/huggingface_cache with the \"cache_dir\" argument, recreating the traditional file structure.")

    parser.add_argument(
        "--front-end-version",
        type=str,
        default=DEFAULT_VERSION_STRING,
        help="""
        Specifies the version of the frontend to be used. This command needs internet connectivity to query and
        download available frontend implementations from GitHub releases.
    
        The version string should be in the format of:
        [repoOwner]/[repoName]@[version]
        where version is one of: "latest" or a valid version number (e.g. "1.0.0")
        """,
    )

    parser.add_argument(
        '--panic-when',
        action=FlattenAndAppendAction,
        nargs='+',
        help="""
        List of fully qualified exception class names to panic (sys.exit(1)) when a workflow raises it.
        Example: --panic-when=torch.cuda.OutOfMemoryError. Can be specified multiple times or as a 
        comma-separated list.""",
        type=str,
        default=[]
    )

    parser.add_argument(
        "--front-end-root",
        type=is_valid_directory,
        default=None,
        help="The local filesystem path to the directory where the frontend is located. Overrides --front-end-version.",
    )

    parser.add_argument(
        "--executor-factory",
        type=str,
        default="ThreadPoolExecutor",
        help="When running ComfyUI as a distributed worker, this specifies the kind of executor that should be used to run the actual ComfyUI workflow worker. A ThreadPoolExecutor is the default. A ProcessPoolExecutor results in better memory management, since the process will be closed and large, contiguous blocks of CUDA memory can be freed."
    )

    parser.add_argument(
        "--openai-api-key",
        required=False,
        type=str,
        help="Configures the OpenAI API Key for the OpenAI nodes. Visit https://platform.openai.com/api-keys to create this key.",
        env_var="OPENAI_API_KEY",
        default=None
    )

    parser.add_argument(
        "--ideogram-api-key",
        required=False,
        type=str,
        help="Configures the Ideogram API Key for the Ideogram nodes. Visit https://ideogram.ai/manage-api to create this key.",
        env_var="IDEOGRAM_API_KEY",
        default=None
    )

    parser.add_argument(
        "--anthropic-api-key",
        required=False,
        type=str,
        help="Configures the Anthropic API key for its nodes related to Claude functionality. Visit https://console.anthropic.com/settings/keys to create this key.",
        env_var="ANTHROPIC_API_KEY"
    )

    parser.add_argument("--user-directory", type=is_valid_directory, default=None, help="Set the ComfyUI user directory with an absolute path. Overrides --base-directory.")

    parser.add_argument("--enable-compress-response-body", action="store_true", help="Enable compressing response body.")

    parser.add_argument(
        "--comfy-api-base",
        type=str,
        default="https://api.comfy.org",
        help="Set the base URL for the ComfyUI API.  (default: https://api.comfy.org)",
    )

    parser.add_argument(
        "--block-runtime-package-installation",
        action="store_true",
        help="When set, custom nodes like ComfyUI Manager, Easy Use, Nunchaku and others will not be able to use pip or uv to install packages at runtime (experimental)."
    )

    default_db_url = db_config()
    parser.add_argument("--database-url", type=str, default=default_db_url, help="Specify the database URL, e.g. for an in-memory database you can use 'sqlite:///:memory:'.")
    parser.add_argument("--workflows", type=str, action=FlattenAndAppendAction, nargs='+', default=[], help="Execute the API workflow(s) specified in the provided files. For each workflow, its outputs will be printed to a line to standard out. Application logging will be redirected to standard error. Use `-` to signify standard in.")

    # now give plugins a chance to add configuration
    for entry_point in entry_points().select(group='comfyui.custom_config'):
        try:
            plugin_callable: ConfigurationExtender | ModuleType = entry_point.load()
            if isinstance(plugin_callable, ModuleType):
                # todo: find the configuration extender in the module
                raise ValueError("unexpected or unsupported plugin configuration type")
            else:
                parser_result = plugin_callable(parser)
                if parser_result is not None:
                    parser = parser_result
        except Exception as exc:
            logger.error("Failed to load custom config plugin", exc_info=exc)

    return parser


def _parse_args(parser: Optional[argparse.ArgumentParser] = None, args_parsing: bool = False) -> Configuration:
    if parser is None:
        parser = _create_parser()

    if args_parsing:
        args, _, config_files = parser.parse_known_args_with_config_files()
    else:
        args, _, config_files = parser.parse_known_args_with_config_files([])

    if args.windows_standalone_build:
        args.auto_launch = True

    if args.disable_auto_launch:
        args.auto_launch = False

    if args.force_fp16:
        args.fp16_unet = True

    configuration_obj = Configuration(**vars(args))
    configuration_obj.config_files = config_files
    assert all(isinstance(config_file, str) for config_file in config_files)
    return configuration_obj


def default_configuration() -> Configuration:
    return _parse_args(_create_parser())


def cli_args_configuration() -> Configuration:
    return _parse_args(args_parsing=True)


@_module_properties.getter
def _args() -> Configuration:
    from .execution_context import current_execution_context
    return current_execution_context().configuration


__all__ = [
    "args",  # pylint: disable=undefined-all-variable, type: Configuration
    "default_configuration",
    "cli_args_configuration",
    "DEFAULT_VERSION_STRING"
]

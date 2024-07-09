from __future__ import annotations

import logging
import os
import sys
from importlib.metadata import entry_points
from types import ModuleType
from typing import Optional, List

import configargparse as argparse
from watchdog.observers import Observer

from . import __version__
from . import options
from .cli_args_types import LatentPreviewMethod, Configuration, ConfigurationExtender, ConfigChangeHandler, EnumAction, \
    EnhancedConfigArgParser


def _create_parser() -> EnhancedConfigArgParser:
    parser = EnhancedConfigArgParser(default_config_files=['config.yaml', 'config.json'],
                                     auto_env_var_prefix='COMFYUI_',
                                     args_for_setting_config_path=["-c", "--config"],
                                     add_env_var_help=True, add_config_file_help=True, add_help=True,
                                     args_for_writing_out_config_file=["--write-out-config-file"])

    parser.add_argument('-w', "--cwd", type=str, default=None,
                        help="Specify the working directory. If not set, this is the current working directory. models/, input/, output/ and other directories will be located here by default.")
    parser.add_argument('-H', "--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0",
                        help="Specify the IP address to listen on (default: 127.0.0.1). If --listen is provided without an argument, it defaults to 0.0.0.0. (listens on all)")
    parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
    parser.add_argument("--enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*",
                        help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'.")
    parser.add_argument("--max-upload-size", type=float, default=100, help="Set the maximum upload size in MB.")
    parser.add_argument("--extra-model-paths-config", type=str, default=None, metavar="PATH", nargs='+',
                        action='append', help="Load one or more extra_model_paths.yaml files.")
    parser.add_argument("--output-directory", type=str, default=None, help="Set the ComfyUI output directory.")
    parser.add_argument("--temp-directory", type=str, default=None,
                        help="Set the ComfyUI temp directory (default is in the ComfyUI directory).")
    parser.add_argument("--input-directory", type=str, default=None, help="Set the ComfyUI input directory.")
    parser.add_argument("--auto-launch", action="store_true",
                        help="Automatically launch ComfyUI in the default browser.")
    parser.add_argument("--disable-auto-launch", action="store_true", help="Disable auto launching the browser.")
    parser.add_argument("--cuda-device", type=int, default=None, metavar="DEVICE_ID",
                        help="Set the id of the cuda device this instance will use.")
    cm_group = parser.add_mutually_exclusive_group()
    cm_group.add_argument("--cuda-malloc", action="store_true",
                          help="Enable cudaMallocAsync (enabled by default for torch 2.0 and up).")
    cm_group.add_argument("--disable-cuda-malloc", action="store_true", help="Disable cudaMallocAsync.")

    fp_group = parser.add_mutually_exclusive_group()
    fp_group.add_argument("--force-fp32", action="store_true",
                          help="Force fp32 (If this makes your GPU work better please report it).")
    fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")
    fp_group.add_argument("--force-bf16", action="store_true", help="Force bf16.")

    fpunet_group = parser.add_mutually_exclusive_group()
    fpunet_group.add_argument("--bf16-unet", action="store_true",
                              help="Run the UNET in bf16. This should only be used for testing stuff.")
    fpunet_group.add_argument("--fp16-unet", action="store_true", help="Store unet weights in fp16.")
    fpunet_group.add_argument("--fp8_e4m3fn-unet", action="store_true", help="Store unet weights in fp8_e4m3fn.")
    fpunet_group.add_argument("--fp8_e5m2-unet", action="store_true", help="Store unet weights in fp8_e5m2.")

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

    parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1,
                        help="Use torch-directml.")

    parser.add_argument("--disable-ipex-optimize", action="store_true",
                        help="Disables ipex.optimize when loading models with Intel GPUs.")

    parser.add_argument("--preview-method", type=LatentPreviewMethod, default=LatentPreviewMethod.Auto,
                        help="Default preview method for sampler nodes.", action=EnumAction)

    attn_group = parser.add_mutually_exclusive_group()
    attn_group.add_argument("--use-split-cross-attention", action="store_true",
                            help="Use the split cross attention optimization. Ignored when xformers is used.")
    attn_group.add_argument("--use-quad-cross-attention", action="store_true",
                            help="Use the sub-quadratic cross attention optimization . Ignored when xformers is used.")
    attn_group.add_argument("--use-pytorch-cross-attention", action="store_true",
                            help="Use the new pytorch 2.0 cross attention function.")

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

    parser.add_argument("--disable-smart-memory", action="store_true",
                        help="Force ComfyUI to agressively offload to regular ram instead of keeping models in vram when it can.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.")

    parser.add_argument("--dont-print-server", action="store_true", help="Don't print server output.")
    parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI. Raises an error if nodes cannot be imported,")
    parser.add_argument("--windows-standalone-build", default=hasattr(sys, 'frozen') and getattr(sys, 'frozen'),
                        action="store_true",
                        help="Windows standalone build: Enable convenient things that most people using the standalone windows build will probably enjoy (like auto opening the page on startup).")

    parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata in files.")
    parser.add_argument("--disable-all-custom-nodes", action="store_true", help="Disable loading all custom nodes.")

    parser.add_argument("--multi-user", action="store_true", help="Enables per-user storage.")
    parser.add_argument("--create-directories", action="store_true",
                        help="Creates the default models/, input/, output/ and temp/ directories, then exits.")

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
    parser.add_argument("--verbose", action="store_true", help="Enables more debug prints.")
    parser.add_argument("--disable-known-models", action="store_true", help="Disables automatic downloads of known models and prevents them from appearing in the UI.")
    parser.add_argument("--max-queue-size", type=int, default=65536, help="The API will reject prompt requests if the queue's size exceeds this value.")
    # tracing
    parser.add_argument("--otel-service-name", type=str, default="comfyui", env_var="OTEL_SERVICE_NAME", help="The name of the service or application that is generating telemetry data.")
    parser.add_argument("--otel-service-version", type=str, default=__version__, env_var="OTEL_SERVICE_VERSION", help="The version of the service or application that is generating telemetry data.")
    parser.add_argument("--otel-exporter-otlp-endpoint", type=str, default=None, env_var="OTEL_EXPORTER_OTLP_ENDPOINT", help="A base endpoint URL for any signal type, with an optionally-specified port number. Helpful for when you're sending more than one signal to the same endpoint and want one environment variable to control the endpoint.")
    parser.add_argument("--force-channels-last", action="store_true", help="Force channels last format when inferencing the models.")
    parser.add_argument("--force-hf-local-dir-mode", action="store_true", help="Download repos from huggingface.co to the models/huggingface directory with the \"local_dir\" argument instead of models/huggingface_cache with the \"cache_dir\" argument, recreating the traditional file structure.")

    # now give plugins a chance to add configuration
    for entry_point in entry_points().select(group='comfyui.custom_config'):
        try:
            plugin_callable: ConfigurationExtender | ModuleType = entry_point.load()
            if isinstance(plugin_callable, ModuleType):
                # todo: find the configuration extender in the module
                plugin_callable = ...
            else:
                parser_result = plugin_callable(parser)
                if parser_result is not None:
                    parser = parser_result
        except Exception as exc:
            logging.error("Failed to load custom config plugin", exc_info=exc)

    return parser


def _parse_args(parser: Optional[argparse.ArgumentParser] = None) -> Configuration:
    if parser is None:
        parser = _create_parser()

    if options.args_parsing:
        args, _, config_files = parser.parse_known_args_with_config_files()
    else:
        args, _, config_files = parser.parse_known_args_with_config_files([])

    if args.windows_standalone_build:
        args.auto_launch = True

    if args.disable_auto_launch:
        args.auto_launch = False

    logging_level = logging.INFO
    if args.verbose:
        logging_level = logging.DEBUG

    logging.basicConfig(format="%(message)s", level=logging_level)
    configuration_obj = Configuration(**vars(args))
    configuration_obj.config_files = config_files
    assert all(isinstance(config_file, str) for config_file in config_files)
    # we always have to set up a watcher, even when there are no existing files
    if len(config_files) > 0:
        _setup_config_file_watcher(configuration_obj, parser, config_files)
    return configuration_obj


def _setup_config_file_watcher(config: Configuration, parser: EnhancedConfigArgParser, config_files: List[str]):
    def update_config():
        new_args, _, _ = parser.parse_known_args()
        new_config = vars(new_args)
        config.update(new_config)

    handler = ConfigChangeHandler(config_files, update_config)
    observer = Observer()

    for config_file in config_files:
        config_dir = os.path.dirname(config_file) or '.'
        observer.schedule(handler, path=config_dir, recursive=False)

    observer.start()

    # Ensure the observer is stopped when the program exits
    import atexit
    atexit.register(observer.stop)
    atexit.register(observer.join)


args = _parse_args()

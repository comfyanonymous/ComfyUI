from typing import Optional

from ..cli_args_types import Configuration
from ..cmd.extra_model_paths import load_extra_path_config
from .folder_path_types import FolderNames
from ..component_model.platform_path import construct_path
import itertools
import os

from ..distributed.executors import ContextVarExecutor, ContextVarProcessPoolExecutor


def configure_application_paths(args: Configuration, folder_names: Optional[FolderNames] = None):
    if folder_names is None:
        from ..cmd import folder_paths
        folder_names = folder_paths.folder_names_and_paths
    # configure paths
    if args.output_directory:
        folder_names.application_paths.output_directory = construct_path(args.output_directory)
    if args.input_directory:
        folder_names.application_paths.input_directory = construct_path(args.input_directory)
    if args.temp_directory:
        folder_names.application_paths.temp_directory = construct_path(args.temp_directory)
    if args.extra_model_paths_config:
        for config_path in args.extra_model_paths_config:
            load_extra_path_config(config_path)


async def executor_from_args(configuration:Optional[Configuration]=None):
    if configuration is None:
        from ..cli_args import args
        configuration = args

    if configuration.executor_factory in ("ThreadPoolExecutor", "ContextVarExecutor"):
        executor = ContextVarExecutor()
    elif configuration.executor_factory in ("ProcessPoolExecutor", "ContextVarProcessPoolExecutor"):
        executor = ContextVarProcessPoolExecutor()
    else:
        # default executor
        executor = ContextVarExecutor()
    return executor

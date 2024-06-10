import json
import multiprocessing
import pathlib
import time
import urllib
from typing import Tuple

import pytest

from comfy.cli_args_types import Configuration


def run_server(server_arguments: Configuration):
    from comfy.cmd.main import main
    from comfy.cli_args import args
    import asyncio
    for arg, value in server_arguments.items():
        args[arg] = value
    asyncio.run(main())


@pytest.fixture(scope="function", autouse=False)
def comfy_background_server(tmp_path) -> Tuple[Configuration, multiprocessing.Process]:
    import torch
    # Start server

    configuration = Configuration()
    configuration.listen = "localhost"
    configuration.output_directory = str(tmp_path)
    configuration.input_directory = str(tmp_path)

    server_process = multiprocessing.Process(target=run_server, args=(configuration,))
    server_process.start()
    # wait for http url to be ready
    success = False
    for i in range(60):
        try:
            with urllib.request.urlopen(f"http://localhost:{configuration['port']}/object_info") as response:
                success = response.status == 200
                if success:
                    break
        except:
            pass
        time.sleep(1)
    if not success:
        raise Exception("Failed to start background server")
    yield configuration, server_process
    server_process.terminate()
    torch.cuda.empty_cache()


def pytest_collection_modifyitems(items):
    # Modifies items so tests run in the correct order

    LAST_TESTS = ['test_quality']

    # Move the last items to the end
    last_items = []
    for test_name in LAST_TESTS:
        for item in items.copy():
            print(item.module.__name__, item)
            if item.module.__name__ == test_name:
                last_items.append(item)
                items.remove(item)

    items.extend(last_items)


@pytest.fixture(scope="module")
def vae():
    from comfy.nodes.base_nodes import VAELoader

    vae_file = "vae-ft-mse-840000-ema-pruned.safetensors"
    try:
        vae, = VAELoader().load_vae(vae_file)
    except FileNotFoundError:
        pytest.skip(f"{vae_file} not present on machine")
    return vae


@pytest.fixture(scope="module")
def clip():
    from comfy.nodes.base_nodes import CheckpointLoaderSimple

    checkpoint = "v1-5-pruned-emaonly.safetensors"
    try:
        return CheckpointLoaderSimple().load_checkpoint(checkpoint)[1]
    except FileNotFoundError:
        pytest.skip(f"{checkpoint} not present on machine")


@pytest.fixture(scope="module")
def model(clip):
    from comfy.nodes.base_nodes import CheckpointLoaderSimple
    checkpoint = "v1-5-pruned-emaonly.safetensors"
    try:
        return CheckpointLoaderSimple().load_checkpoint(checkpoint)[0]
    except FileNotFoundError:
        pytest.skip(f"{checkpoint} not present on machine")


@pytest.fixture(scope="function", autouse=True)
def use_temporary_output_directory(tmp_path: pathlib.Path):
    from comfy.cmd import folder_paths

    orig_dir = folder_paths.get_output_directory()
    folder_paths.set_output_directory(tmp_path)
    yield tmp_path
    folder_paths.set_output_directory(orig_dir)


@pytest.fixture(scope="function", autouse=True)
def use_temporary_input_directory(tmp_path: pathlib.Path):
    from comfy.cmd import folder_paths

    orig_dir = folder_paths.get_input_directory()
    folder_paths.set_input_directory(tmp_path)
    yield tmp_path
    folder_paths.set_input_directory(orig_dir)

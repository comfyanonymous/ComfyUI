import logging
import multiprocessing
import os
import pathlib
import subprocess
import urllib
from contextvars import ContextVar
from multiprocessing import Process
from typing import List, Any, Generator

import pytest
import requests
import sys
import time


os.environ['OTEL_METRICS_EXPORTER'] = 'none'
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "True"
# fixes issues with running the testcontainers rabbitmqcontainer on Windows
os.environ["TC_HOST"] = "localhost"

from comfy.cli_args import default_configuration
from comfy.cli_args_types import Configuration

logging.getLogger("pika").setLevel(logging.CRITICAL + 1)
logging.getLogger("aio_pika").setLevel(logging.CRITICAL + 1)


def run_server(server_arguments: Configuration):
    from comfy.cmd.main import _start_comfyui
    import asyncio
    asyncio.run(_start_comfyui(configuration=server_arguments))


@pytest.fixture(scope="function", autouse=False)
def has_gpu() -> bool:
    # mps
    has_gpu = False
    try:
        import torch
        has_gpu = torch.backends.mps.is_available() and torch.device("mps") is not None
        if has_gpu:
            from comfy import model_management
            from comfy.model_management import CPUState
            model_management.cpu_state = CPUState.MPS
    except ImportError:
        pass

    if not has_gpu:
        # ipex
        try:
            import intel_extension_for_pytorch as ipex
            has_gpu = ipex.xpu.device_count() > 0
        except ImportError:
            has_gpu = False

        if not has_gpu:
            # cuda
            try:
                import torch
                has_gpu = torch.device(torch.cuda.current_device()) is not None
            except:
                has_gpu = False

    if has_gpu:
        from comfy import model_management
        from comfy.model_management import CPUState
        if model_management.cpu_state != CPUState.MPS:
            model_management.cpu_state = CPUState.GPU if has_gpu else CPUState.CPU
    yield has_gpu


@pytest.fixture(scope="module", autouse=False, params=["ThreadPoolExecutor", "ProcessPoolExecutor"])
def frontend_backend_worker_with_rabbitmq(request, tmp_path_factory, num_workers: int = 1):
    from huggingface_hub import hf_hub_download
    from testcontainers.rabbitmq import RabbitMqContainer

    logging.getLogger("testcontainers.core.container").setLevel(logging.WARNING)
    logging.getLogger("testcontainers.core.waiting_utils").setLevel(logging.WARNING)

    hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors")
    hf_hub_download("stabilityai/stable-diffusion-xl-refiner-1.0", "sd_xl_refiner_1.0.safetensors")

    tmp_path = tmp_path_factory.mktemp("comfy_background_server")
    executor_factory = request.param
    processes_to_close: List[subprocess.Popen] = []

    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        params = rabbitmq.get_connection_params()
        connection_uri = f"amqp://guest:guest@127.0.0.1:{params.port}"

        # Check if OTEL endpoint is configured for integration testing
        otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

        env = os.environ.copy()
        if otel_endpoint:
            env["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_endpoint
            logging.info(f"Configuring services to export traces to: {otel_endpoint}")

        frontend_command = [
            "comfyui",
            "--listen=0.0.0.0",
            "--port=19001",
            "--cpu",
            "--distributed-queue-frontend",
            f"-w={str(tmp_path)}",
            f"--distributed-queue-connection-uri={connection_uri}",
        ]

        processes_to_close.append(subprocess.Popen(frontend_command, stdout=sys.stdout, stderr=sys.stderr, env=env))

        # Start multiple workers
        for i in range(num_workers):
            backend_command = [
                "comfyui-worker",
                f"--port={19002 + i}",
                f"-w={str(tmp_path)}",
                f"--distributed-queue-connection-uri={connection_uri}",
                f"--executor-factory={executor_factory}"
            ]
            processes_to_close.append(subprocess.Popen(backend_command, stdout=sys.stdout, stderr=sys.stderr, env=env))

        try:
            server_address = f"http://127.0.0.1:19001"
            start_time = time.time()
            connected = False
            while time.time() - start_time < 60:
                try:
                    response = requests.get(server_address)
                    if response.status_code == 200:
                        connected = True
                        break
                except requests.exceptions.ConnectionError:
                    pass
                time.sleep(1)
            if not connected:
                raise RuntimeError("could not connect to frontend")
            yield server_address
        finally:
            for process in processes_to_close:
                process.terminate()


@pytest.fixture(scope="module", autouse=False)
def comfy_background_server(tmp_path_factory) -> Generator[tuple[Configuration, Process], Any, None]:
    tmp_path = tmp_path_factory.mktemp("comfy_background_server")
    # Start server

    configuration = default_configuration()
    configuration.listen = "localhost"
    configuration.output_directory = str(tmp_path)
    configuration.input_directory = str(tmp_path)

    yield from comfy_background_server_from_config(configuration)


def comfy_background_server_from_config(configuration):
    server_process = multiprocessing.get_context('spawn').Process(target=run_server, args=(configuration,))
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
    import torch
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def skip_timing_checks(pytestconfig):
    """Fixture that returns whether timing checks should be skipped."""
    # todo: in the LTS, we don't need to skip timing checks, everything just works
    return False


def pytest_collection_modifyitems(items):
    # Modifies items so tests run in the correct order

    LAST_TESTS = ['test_quality']

    # Move the last items to the end
    last_items = []
    for test_name in LAST_TESTS:
        for item in items.copy():
            print(item.module.__name__, item)  # noqa: T201
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


@pytest.fixture(scope="function", autouse=False)
def use_temporary_output_directory(tmp_path: pathlib.Path):
    from comfy.cmd import folder_paths

    orig_dir = folder_paths.get_output_directory()
    folder_paths.set_output_directory(tmp_path)
    yield tmp_path
    folder_paths.set_output_directory(orig_dir)


@pytest.fixture(scope="function", autouse=False)
def use_temporary_input_directory(tmp_path: pathlib.Path):
    from comfy.cmd import folder_paths

    orig_dir = folder_paths.get_input_directory()
    folder_paths.set_input_directory(tmp_path)
    yield tmp_path
    folder_paths.set_input_directory(orig_dir)


current_test_name = ContextVar('current_test_name', default=None)


@pytest.fixture(autouse=True)
def set_test_name(request):
    token = current_test_name.set(request.node.name)
    yield
    current_test_name.reset(token)

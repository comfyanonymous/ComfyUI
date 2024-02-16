import signal
import sys

from .. import options

options.enable_args_parsing()

import os
import importlib.util

from ..cmd import cuda_malloc
from ..cmd import folder_paths
from ..analytics.analytics import initialize_event_tracking
import time


def execute_prestartup_script():
    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            print(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    node_prestartup_times = []
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path) if os.path.exists(custom_node_path) else []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        print("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
        print()


execute_prestartup_script()

# Main code
import asyncio
import itertools
import shutil
import threading
import gc

from ..cli_args import args

if os.name == "nt":
    import logging

    logging.getLogger("xformers").addFilter(
        lambda record: 'A matching Triton is not available' not in record.getMessage())

if args.cuda_device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    print("Set cuda device to:", args.cuda_device)

if args.deterministic:
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

from .. import utils
import yaml
from contextlib import AsyncExitStack

from ..cmd import execution
from ..cmd import server as server_module
from ..component_model.abstract_prompt_queue import AbstractPromptQueue
from ..component_model.queue_types import BinaryEventTypes, ExecutionStatus
from .. import model_management
from ..distributed.distributed_prompt_queue import DistributedPromptQueue
from ..component_model.executor_types import ExecutorToClientProgress
from ..distributed.server_stub import ServerStub


def prompt_worker(q: AbstractPromptQueue, _server: server_module.PromptServer):
    e = execution.PromptExecutor(_server)
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0
    current_time = 0.0
    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            _server.last_prompt_id = prompt_id

            e.execute(item[2], prompt_id, item[3], item[4])
            need_gc = True
            q.task_done(item_id,
                        e.outputs_ui,
                        status=ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages))
            if _server.client_id is not None:
                _server.send_sync("executing", { "node": None, "prompt_id": prompt_id }, _server.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            print("Prompt executed in {:.2f} seconds".format(execution_time))

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False


async def run(server, address='', port=8188, verbose=True, call_on_start=None):
    await asyncio.gather(server.start(address, port, verbose, call_on_start), server.publish_loop())


def hijack_progress(server: ExecutorToClientProgress):
    def hook(value: float, total: float, preview_image):
        model_management.throw_exception_if_processing_interrupted()
        progress = {"value": value, "max": total, "prompt_id": server.last_prompt_id, "node": server.last_node_id}

        server.send_sync("progress", progress, server.client_id)
        if preview_image is not None:
            server.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, server.client_id)

    utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    try:
        temp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except NameError:
        # __file__ was not defined
        pass


def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path is not None:
                    full_path = os.path.join(base_path, full_path)
                print("Adding extra search path", x, full_path)
                folder_paths.add_model_folder_path(x, full_path)


def cuda_malloc_warning():
    device = model_management.get_torch_device()
    device_name = model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            print(
                "\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")


async def main():
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        print(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    # create the default directories if we're instructed to, then exit
    # or, if it's a windows standalone build, the single .exe file should have its side-by-side directories always created
    if args.create_directories:
        folder_paths.create_directories()
        return

    if args.windows_standalone_build:
        folder_paths.create_directories()

    loop = asyncio.get_event_loop()
    server = server_module.PromptServer(loop)
    if args.external_address is not None:
        server.external_address = args.external_address
    if args.distributed_queue_connection_uri is not None:
        distributed = True
        q = DistributedPromptQueue(
            caller_server=server if args.distributed_queue_frontend else None,
            connection_uri=args.distributed_queue_connection_uri,
            is_caller=args.distributed_queue_frontend,
            is_callee=args.distributed_queue_worker,
            loop=loop,
            queue_name=args.distributed_queue_name
        )
        await q.init()
    else:
        distributed = False
        q = execution.PromptQueue(server)
    server.prompt_queue = q

    try:
        extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
        if os.path.isfile(extra_model_paths_config_path):
            load_extra_path_config(extra_model_paths_config_path)
    except NameError:
        pass

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)

    server.add_routes()
    hijack_progress(server)
    cuda_malloc_warning()

    # in a distributed setting, the default prompt worker will not be able to send execution events via the websocket
    worker_thread_server = server if not distributed else ServerStub()
    if not distributed or args.distributed_queue_worker:
        if distributed:
            logging.warning(f"Distributed workers started in the default thread loop cannot notify clients of progress updates. Instead of comfyui or main.py, use comfyui-worker.")
        threading.Thread(target=prompt_worker, daemon=True, args=(q, worker_thread_server,)).start()

    # server has been imported and things should be looking good
    initialize_event_tracking(loop)

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        print(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        print(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.quick_test_for_ci:
        exit(0)

    call_on_start = None
    if args.auto_launch:
        def startup_server(address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0' or address == '':
                address = '127.0.0.1'
            webbrowser.open(f"http://{address}:{port}")

        call_on_start = startup_server

    server.address = args.listen
    server.port = args.port
    try:
        await run(server, address=args.listen, port=args.port, verbose=not args.dont_print_server,
                                    call_on_start=call_on_start)
    except asyncio.CancelledError:
        if distributed:
            await q.close()
        print("\nStopped server")

    cleanup_temp()


def entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()

import asyncio
import gc
import itertools
import logging
import os
import shutil
import threading
import time

# main_pre must be the earliest import since it suppresses some spurious warnings
from .main_pre import args
from .extra_model_paths import load_extra_path_config
from .. import model_management
from ..analytics.analytics import initialize_event_tracking
from ..cmd import cuda_malloc
from ..cmd import folder_paths
from ..cmd import server as server_module
from ..component_model.abstract_prompt_queue import AbstractPromptQueue
from ..component_model.queue_types import ExecutionStatus
from ..distributed.distributed_prompt_queue import DistributedPromptQueue
from ..distributed.server_stub import ServerStub
from ..nodes.package import import_all_nodes_in_workspace


def prompt_worker(q: AbstractPromptQueue, _server: server_module.PromptServer):
    from ..cmd.execution import PromptExecutor

    e = PromptExecutor(_server)
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
                _server.send_sync("executing", {"node": None, "prompt_id": prompt_id}, _server.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

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


def cleanup_temp():
    try:
        temp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except NameError:
        # __file__ was not defined
        pass


def cuda_malloc_warning():
    device = model_management.get_torch_device()
    device_name = model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning(
                "\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")


async def main():
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.debug(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    # configure extra model paths earlier
    try:
        extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
        if os.path.isfile(extra_model_paths_config_path):
            load_extra_path_config(extra_model_paths_config_path)
    except NameError:
        pass

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)

    # create the default directories if we're instructed to, then exit
    # or, if it's a windows standalone build, the single .exe file should have its side-by-side directories always created
    if args.create_directories:
        folder_paths.create_directories()
        return

    if args.windows_standalone_build:
        folder_paths.create_directories()
        try:
            from . import new_updater
            new_updater.update_windows_updater()
        except:
            pass

    loop = asyncio.get_event_loop()
    server = server_module.PromptServer(loop)
    if args.external_address is not None:
        server.external_address = args.external_address

    # at this stage, it's safe to import nodes
    server.nodes = import_all_nodes_in_workspace()
    # as a side effect, this also populates the nodes for execution

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
        from .execution import PromptQueue
        q = PromptQueue(server)
    server.prompt_queue = q

    server.add_routes()
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
        logging.debug(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.debug(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.quick_test_for_ci:
        # for CI purposes, try importing all the nodes
        import_all_nodes_in_workspace(raise_on_failure=True)
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
    except (asyncio.CancelledError, KeyboardInterrupt):
        logging.debug("\nStopped server")
    finally:
        if distributed:
            await q.close()
    cleanup_temp()


def entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()

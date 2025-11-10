import asyncio
import contextvars
import gc

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

# main_pre must be the earliest import
from .main_pre import tracer
from ..cli_args_types import Configuration
from ..component_model.file_counter import cleanup_temp as fc_cleanup_temp
from ..execution_context import current_execution_context
from . import hook_breaker_ac10a0
from .extra_model_paths import load_extra_path_config
from .. import model_management
from ..analytics.analytics import initialize_event_tracking
from ..cmd import cuda_malloc
from ..cmd import folder_paths
from ..cmd import server as server_module
from ..component_model.abstract_prompt_queue import AbstractPromptQueue
from ..component_model.entrypoints_common import configure_application_paths, executor_from_args
from ..distributed.distributed_prompt_queue import DistributedPromptQueue
from ..distributed.server_stub import ServerStub
from ..nodes.package import import_all_nodes_in_workspace
from ..nodes_context import get_nodes

logger = logging.getLogger(__name__)


def cuda_malloc_warning():
    device = model_management.get_torch_device()
    device_name = model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logger.warning(
                "\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")


def prompt_worker(q: AbstractPromptQueue, server_instance: server_module.PromptServer):
    asyncio.run(_prompt_worker(q, server_instance))


async def _prompt_worker(q: AbstractPromptQueue, server_instance: server_module.PromptServer):
    from ..cmd import execution
    from ..component_model import queue_types
    from .. import model_management
    args = current_execution_context().configuration
    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_none:
        cache_type = execution.CacheType.DEPENDENCY_AWARE

    e = execution.PromptExecutor(server_instance, cache_type=cache_type, cache_size=args.cache_lru)
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
            server_instance.last_prompt_id = prompt_id

            await e.execute_async(item[2], prompt_id, item[3], item[4])
            need_gc = True

            # Extract error details from status_messages if there's an error
            error_details = None
            if not e.success:
                for event, data in e.status_messages:
                    if event == "execution_error":
                        error_details = data
                        break

            # Convert status_messages tuples to string messages for backward compatibility
            messages = [f"{event}: {data.get('exception_message', str(data))}" if isinstance(data, dict) and 'exception_message' in data else f"{event}" for event, data in e.status_messages]

            q.task_done(item_id,
                        e.history_result,
                        status=queue_types.ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=messages),
                        error_details=error_details)
            if server_instance.client_id is not None:
                server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id},
                                          server_instance.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            # Log Time in a more readable way after 10 minutes
            if execution_time > 600:
                execution_time = time.strftime("%H:%M:%S", time.gmtime(execution_time))
                logger.info(f"Prompt executed in {execution_time}")
            else:
                logger.info("Prompt executed in {:.2f} seconds".format(execution_time))

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
                hook_breaker_ac10a0.restore_functions()


async def run(server_instance, address='', port=8188, verbose=True, call_on_start=None):
    addresses = []
    for addr in address.split(","):
        addresses.append((addr, port))
    await asyncio.gather(server_instance.start_multi_address(addresses, call_on_start), server_instance.publish_loop())


def cleanup_temp():
    try:
        folder_paths.get_temp_directory()
        temp_dir = folder_paths.get_temp_directory()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except NameError:
        # __file__ was not defined
        pass


def start_comfyui(asyncio_loop: asyncio.AbstractEventLoop = None):
    asyncio_loop = asyncio_loop or asyncio.get_event_loop()
    asyncio_loop.run_until_complete(_start_comfyui())


def setup_database():
    from ..app.database.db import dependencies_available, init_db
    if dependencies_available():
        init_db()


async def _start_comfyui(from_script_dir: Optional[Path] = None, configuration: Optional[Configuration] = None):
    from ..execution_context import context_configuration
    from ..cli_args import cli_args_configuration
    configuration = configuration or cli_args_configuration()
    with (
        context_configuration(configuration),
        fc_cleanup_temp()
    ):
        await __start_comfyui(from_script_dir=from_script_dir)


async def __start_comfyui(from_script_dir: Optional[Path] = None):
    """
    Runs ComfyUI's frontend and backend like upstream.
    :param from_script_dir: when set to a path, assumes that you are running ComfyUI's legacy main.py entrypoint at the root of the git repository located at the path
    """
    args = current_execution_context().configuration
    if not from_script_dir:
        os_getcwd = os.getcwd()
    else:
        os_getcwd = str(from_script_dir)

    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logger.debug(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logger.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)

    # configure extra model paths earlier
    try:
        extra_model_paths_config_path = os.path.join(os_getcwd, "extra_model_paths.yaml")
        if os.path.isfile(extra_model_paths_config_path):
            load_extra_path_config(extra_model_paths_config_path)
    except NameError:
        pass

    if args.extra_model_paths_config:
        for config_path in args.extra_model_paths_config:
            load_extra_path_config(config_path)

    if args.create_directories:
        # then, import and exit
        import_all_nodes_in_workspace(raise_on_failure=False)
        folder_paths.create_directories()
        exit(0)
    elif args.quick_test_for_ci:
        import_all_nodes_in_workspace(raise_on_failure=True)
        exit(0)

    if args.windows_standalone_build:
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
    hook_breaker_ac10a0.save_functions()
    nodes_to_import = get_nodes()
    logger.debug(f"Imported {len(nodes_to_import)} nodes")
    server.nodes = nodes_to_import
    hook_breaker_ac10a0.restore_functions()
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
    setup_database()

    # in a distributed setting, the default prompt worker will not be able to send execution events via the websocket
    worker_thread_server = server if not distributed else ServerStub()
    if not distributed or args.distributed_queue_worker:
        if distributed:
            logger.warning(
                f"Distributed workers started in the default thread loop cannot notify clients of progress updates. Instead of comfyui or main.py, use comfyui-worker.")
        # todo: this should really be using an executor instead of doing things this jankilicious way
        ctx = contextvars.copy_context()
        threading.Thread(target=lambda _q, _worker_thread_server: ctx.run(prompt_worker, _q, _worker_thread_server),
                         daemon=True, args=(q, worker_thread_server,)).start()

    # server has been imported and things should be looking good
    initialize_event_tracking(loop)

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logger.debug(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models",
                                       os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logger.debug(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    # now that nodes are loaded, create directories
    folder_paths.create_directories()

    if len(args.workflows) > 0:
        configure_application_paths(args)
        executor = await executor_from_args(args)
        from ..entrypoints.workflow import run_workflows
        await run_workflows(executor, args.workflows)
        return

    # replaced my folder_paths.create_directories
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme="http", address="localhost", port=8188):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0' or address == '':
                address = '127.0.0.1'
            if ':' in address:
                address = "[{}]".format(address)
            webbrowser.open(f"{scheme}://{address}:{port}")

        call_on_start = startup_server

    first_listen_addr = args.listen.split(',')[0] if ',' in args.listen else args.listen
    server.address = first_listen_addr
    server.port = args.port

    try:
        await server.setup()
        await run(server, address=first_listen_addr, port=args.port, verbose=not args.dont_print_server,
                  call_on_start=call_on_start)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.debug("Stopped server")
    finally:
        if distributed:
            await q.close()


def entrypoint():
    try:
        asyncio.run(_start_comfyui())
    except KeyboardInterrupt:
        logger.info(f"Gracefully shutting down due to KeyboardInterrupt")


def main():
    entrypoint()


if __name__ == "__main__":
    entrypoint()

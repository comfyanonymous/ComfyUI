


import comfy.options
comfy.options.enable_args_parsing()

import os
import importlib.util
import folder_paths
import time
from framework.app_log import AppLog
from aiyo_executor.message_sender import MessageManager
from aiyo_executor.aiyo_executor import AIYoExecutor
import aiyo_project_init
from framework.app_log import AppLog
AppLog.init()

def execute_prestartup_script():
    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            AppLog.info(f"Failed to execute startup-script: {script_path} / {e}")
        return False
    
    # aiyo project init
    aiyo_project_init.aiyo_proj_init()

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

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
        AppLog.info("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            AppLog.info("{:6.1f} seconds{}:{}".format(n[0], import_message, n[1]))

execute_prestartup_script()


# Main code
import asyncio
import itertools
import shutil
import threading
import gc

from comfy.cli_args import args

if os.name == "nt":
    import logging
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

def prepare_cuda_env():
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        AppLog.info("Set cuda device to:", args.cuda_device)

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import cuda_malloc

import comfy.utils
import yaml

import execution
import server
from server import BinaryEventTypes
from nodes import init_custom_nodes
import comfy.model_management
from framework.flow_execution import FlowExecutor

def cuda_malloc_warning():
    prepare_cuda_env()
    
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            AppLog.info("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")



def hijack_progress(server):
    def hook(value, total, preview_image):
        comfy.model_management.throw_exception_if_processing_interrupted()
        server.send_sync("progress", {"value": value, "max": total}, server.client_id)
        if preview_image is not None:
            server.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, server.client_id)
    comfy.utils.set_progress_bar_global_hook(hook)



def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


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
                AppLog.info(f"Adding extra search path: {x},  {full_path}")
                folder_paths.add_model_folder_path(x, full_path)



async def run(msg_sender):
    await msg_sender.publish_loop()



def aiyo_executor_main():
    
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        AppLog.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    msg_sender = MessageManager(loop)
    executor = AIYoExecutor(msg_sender)
    
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)

    init_custom_nodes()

    cuda_malloc_warning()
    
    hijack_progress(msg_sender)


    threading.Thread(target=executor.run, daemon=True).start()

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        AppLog.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    #These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        AppLog.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    try:
        # loop.run_until_complete(msg_sender.publish_loop)
        loop.run_until_complete(run(msg_sender))
    except KeyboardInterrupt:
        AppLog.info("\nStopped server")

    cleanup_temp()
    
    
    
if __name__ == "__main__":
    aiyo_executor_main()
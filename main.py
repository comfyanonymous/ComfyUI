import argparse
import asyncio
import os
import shutil
import sys
import threading

if os.name == "nt":
    import logging
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script Arguments")

    parser.add_argument("--listen", type=str, default="127.0.0.1", help="Listen on IP or 0.0.0.0 if none given so the UI can be accessed from other computers.")
    parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
    parser.add_argument("--extra-model-paths-config", type=str, default=None, help="Load an extra_model_paths.yaml file.")
    parser.add_argument("--output-directory", type=str, default=None, help="Set the ComfyUI output directory.")
    parser.add_argument("--dont-upcast-attention", action="store_true", help="Disable upcasting of attention. Can boost speed but increase the chances of black images.")
    parser.add_argument("--use-split-cross-attention", action="store_true", help="Use the split cross attention optimization instead of the sub-quadratic one. Ignored when xformers is used.")
    parser.add_argument("--use-pytorch-cross-attention", action="store_true", help="Use the new pytorch 2.0 cross attention function.")
    parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")
    parser.add_argument("--cuda-device", type=int, default=None, help="Set the id of the cuda device this instance will use.")
    parser.add_argument("--highvram", action="store_true", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.")
    parser.add_argument("--normalvram", action="store_true", help="Used to force normal vram use if lowvram gets automatically enabled.")
    parser.add_argument("--lowvram", action="store_true", help="Split the unet in parts to use less vram.")
    parser.add_argument("--novram", action="store_true", help="When lowvram isn't enough.")
    parser.add_argument("--cpu", action="store_true", help="To use the CPU for everything (slow).")
    parser.add_argument("--dont-print-server", action="store_true", help="Don't print server output.")
    parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI.")
    parser.add_argument("--windows-standalone-build", action="store_true", help="Windows standalone build.")

    args = parser.parse_args()

    if args.dont_upcast_attention:
        print("disabling upcasting of attention")
        os.environ['ATTN_PRECISION'] = "fp16"

    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        print("Set cuda device to:", args.cuda_device)


import yaml

import execution
import folder_paths
import server
from nodes import init_custom_nodes


def prompt_worker(q, server):
    e = execution.PromptExecutor(server)
    while True:
        item, item_id = q.get()
        e.execute(item[-2], item[-1])
        q.task_done(item_id, e.outputs)

async def run(server, address='', port=8188, verbose=True, call_on_start=None):
    await asyncio.gather(server.start(address, port, verbose, call_on_start), server.publish_loop())

def hijack_progress(server):
    from tqdm.auto import tqdm
    orig_func = getattr(tqdm, "update")
    def wrapped_func(*args, **kwargs):
        pbar = args[0]
        v = orig_func(*args, **kwargs)
        server.send_sync("progress", { "value": pbar.n, "max": pbar.total}, server.client_id)            
        return v
    setattr(tqdm, "update", wrapped_func)

def cleanup_temp():
    temp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp")
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
                print("Adding extra search path", x, full_path)
                folder_paths.add_model_folder_path(x, full_path)

if __name__ == "__main__":
    cleanup_temp()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = server.PromptServer(loop)
    q = execution.PromptQueue(server)

    init_custom_nodes()
    server.add_routes()
    hijack_progress(server)

    threading.Thread(target=prompt_worker, daemon=True, args=(q,server,)).start()

    address = args.listen

    dont_print = args.dont_print_server

    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        load_extra_path_config(args.extra_model_paths_config)

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        print("setting output directory to:", output_dir)
        folder_paths.set_output_directory(output_dir)

    port = args.port

    if args.quick_test_for_ci:
        exit(0)

    call_on_start = None
    if args.windows_standalone_build:
        def startup_server(address, port):
            import webbrowser
            webbrowser.open("http://{}:{}".format(address, port))
        call_on_start = startup_server

    if os.name == "nt":
        try:
            loop.run_until_complete(run(server, address=address, port=port, verbose=not dont_print, call_on_start=call_on_start))
        except KeyboardInterrupt:
            pass
    else:
        loop.run_until_complete(run(server, address=address, port=port, verbose=not dont_print, call_on_start=call_on_start))

    cleanup_temp()

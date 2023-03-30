import os
import sys
import shutil

import threading
import asyncio

if os.name == "nt":
    import logging
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

if __name__ == "__main__":
    if '--help' in sys.argv:
        print("Valid Command line Arguments:")
        print("\t--listen [ip]\t\t\tListen on ip or 0.0.0.0 if none given so the UI can be accessed from other computers.")
        print("\t--port 8188\t\t\tSet the listen port.")
        print("\t--dont-upcast-attention\t\tDisable upcasting of attention \n\t\t\t\t\tcan boost speed but increase the chances of black images.\n")
        print("\t--use-split-cross-attention\tUse the split cross attention optimization instead of the sub-quadratic one.\n\t\t\t\t\tIgnored when xformers is used.")
        print("\t--use-pytorch-cross-attention\tUse the new pytorch 2.0 cross attention function.")
        print("\t--disable-xformers\t\tdisables xformers")
        print("\t--cuda-device 1\t\tSet the id of the cuda device this instance will use.")
        print()
        print("\t--highvram\t\t\tBy default models will be unloaded to CPU memory after being used.\n\t\t\t\t\tThis option keeps them in GPU memory.\n")
        print("\t--normalvram\t\t\tUsed to force normal vram use if lowvram gets automatically enabled.")
        print("\t--lowvram\t\t\tSplit the unet in parts to use less vram.")
        print("\t--novram\t\t\tWhen lowvram isn't enough.")
        print()
        print("\t--cpu\t\t\tTo use the CPU for everything (slow).")
        exit()

    if '--dont-upcast-attention' in sys.argv:
        print("disabling upcasting of attention")
        os.environ['ATTN_PRECISION'] = "fp16"

    try:
        index = sys.argv.index('--cuda-device')
        device = sys.argv[index + 1]
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        print("Set cuda device to:", device)
    except:
        pass

import execution
import server
import folder_paths
import yaml

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

    hijack_progress(server)

    threading.Thread(target=prompt_worker, daemon=True, args=(q,server,)).start()
    try:
        address = '0.0.0.0'
        p_index = sys.argv.index('--listen')
        try:
            ip = sys.argv[p_index + 1]
            if ip[:2] != '--':
                address = ip
        except:
            pass
    except:
        address = '127.0.0.1'


    dont_print = False
    if '--dont-print-server' in sys.argv:
        dont_print = True

    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        load_extra_path_config(extra_model_paths_config_path)

    if '--extra-model-paths-config' in sys.argv:
        indices = [(i + 1) for i in range(len(sys.argv) - 1) if sys.argv[i] == '--extra-model-paths-config']
        for i in indices:
            load_extra_path_config(sys.argv[i])

    port = 8188
    try:
        p_index = sys.argv.index('--port')
        port = int(sys.argv[p_index + 1])
    except:
        pass

    if '--quick-test-for-ci' in sys.argv:
        exit(0)

    call_on_start = None
    if "--windows-standalone-build" in sys.argv:
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

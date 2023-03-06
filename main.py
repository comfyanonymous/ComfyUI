import os
import sys

import threading
import asyncio

if os.name == "nt":
    import logging
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

import execution
import server

if __name__ == "__main__":
    if '--help' in sys.argv:
        print("Valid Command line Arguments:")
        print("\t--listen\t\t\tListen on 0.0.0.0 so the UI can be accessed from other computers.")
        print("\t--port 8188\t\t\tSet the listen port.")
        print("\t--dont-upcast-attention\t\tDisable upcasting of attention \n\t\t\t\t\tcan boost speed but increase the chances of black images.\n")
        print("\t--use-split-cross-attention\tUse the split cross attention optimization instead of the sub-quadratic one.\n\t\t\t\t\tIgnored when xformers is used.")
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

def prompt_worker(q, server):
    e = execution.PromptExecutor(server)
    while True:
        item, item_id = q.get()
        e.execute(item[-2], item[-1])
        q.task_done(item_id, e.outputs)

async def run(server, address='', port=8188, verbose=True):
    await asyncio.gather(server.start(address, port, verbose), server.publish_loop())

def hijack_progress(server):
    from tqdm.auto import tqdm
    orig_func = getattr(tqdm, "update")
    def wrapped_func(*args, **kwargs):
        pbar = args[0]
        v = orig_func(*args, **kwargs)
        server.send_sync("progress", { "value": pbar.n, "max": pbar.total}, server.client_id)            
        return v
    setattr(tqdm, "update", wrapped_func)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = server.PromptServer(loop)
    q = execution.PromptQueue(server)

    hijack_progress(server)

    threading.Thread(target=prompt_worker, daemon=True, args=(q,server,)).start()
    if '--listen' in sys.argv:
        address = '0.0.0.0'
    else:
        address = '127.0.0.1'

    dont_print = False
    if '--dont-print-server' in sys.argv:
        dont_print = True

    port = 8188
    try:
        p_index = sys.argv.index('--port')
        port = int(sys.argv[p_index + 1])
    except:
        pass

    if os.name == "nt":
        try:
            loop.run_until_complete(run(server, address=address, port=port, verbose=not dont_print))
        except KeyboardInterrupt:
            pass
    else:
        loop.run_until_complete(run(server, address=address, port=port, verbose=not dont_print))


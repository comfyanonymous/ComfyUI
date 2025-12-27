"""Core execution logic shared between normal and subprocess execution modes."""

import logging
import time

_active_worker = None


def create_worker(server_instance):
    """Create worker backend. Returns NativeWorker or SubprocessWorker."""
    global _active_worker
    from comfy.cli_args import args

    server = WorkerServer(server_instance)

    if args.use_subprocess_workers:
        from comfy.worker_process import SubprocessWorker
        worker = SubprocessWorker(server, timeout=args.subprocess_timeout)
    else:
        from comfy.worker_native import NativeWorker
        worker = NativeWorker(server)

    _active_worker = worker
    return worker


async def init_execution_environment():
    """Load nodes and custom nodes. Returns number of node types loaded."""
    import nodes
    from comfy.cli_args import args

    await nodes.init_extra_nodes(
        init_custom_nodes=(not args.disable_all_custom_nodes) or len(args.whitelist_custom_nodes) > 0,
        init_api_nodes=not args.disable_api_nodes
    )
    return len(nodes.NODE_CLASS_MAPPINGS)


def setup_progress_hook(server_instance, interrupt_checker):
    """Set up global progress hook. interrupt_checker must raise on interrupt."""
    import comfy.utils
    from comfy_execution.progress import get_progress_state
    from comfy_execution.utils import get_executing_context

    def hook(value, total, preview_image, prompt_id=None, node_id=None):
        ctx = get_executing_context()
        if ctx:
            prompt_id = prompt_id or ctx.prompt_id
            node_id = node_id or ctx.node_id

        interrupt_checker()

        prompt_id = prompt_id or server_instance.last_prompt_id
        node_id = node_id or server_instance.last_node_id

        get_progress_state().update_progress(node_id, value, total, preview_image)
        server_instance.send_sync("progress", {"value": value, "max": total, "prompt_id": prompt_id, "node": node_id}, server_instance.client_id)

    comfy.utils.set_progress_bar_global_hook(hook)


class WorkerServer:
    """Protocol boundary: client_id, last_node_id, last_prompt_id, sockets_metadata, send_sync(), queue_updated()"""

    _WRITABLE = {'client_id', 'last_node_id', 'last_prompt_id'}

    def __init__(self, server):
        object.__setattr__(self, '_server', server)

    def __setattr__(self, name, value):
        if name in self._WRITABLE:
            setattr(self._server, name, value)
        else:
            raise AttributeError(f"WorkerServer does not accept attribute '{name}'")

    @property
    def client_id(self):
        return self._server.client_id

    @property
    def last_node_id(self):
        return self._server.last_node_id

    @property
    def last_prompt_id(self):
        return self._server.last_prompt_id

    @property
    def sockets_metadata(self):
        return self._server.sockets_metadata

    def send_sync(self, event, data, sid=None):
        self._server.send_sync(event, data, sid or self.client_id)

    def queue_updated(self):
        self._server.queue_updated()

def interrupt_processing(value=True):
    _active_worker.interrupt(value)


def _strip_sensitive(prompt):
    return prompt[:5] + prompt[6:]


def prompt_worker(q, worker):
    """Main prompt execution loop."""
    import execution

    server = worker.server_instance

    while True:
        queue_item = q.get(timeout=worker.get_gc_timeout())
        if queue_item is not None:
            item, item_id = queue_item
            start_time = time.perf_counter()
            prompt_id = item[1]
            server.last_prompt_id = prompt_id

            extra_data = {**item[3], **item[5]}

            result = worker.execute_prompt(item[2], prompt_id, extra_data, item[4], server=server)
            worker.mark_needs_gc()

            q.task_done(
                item_id,
                result['history_result'],
                status=execution.PromptQueue.ExecutionStatus(
                    status_str='success' if result['success'] else 'error',
                    completed=result['success'],
                    messages=result['status_messages']
                ),
                process_item=_strip_sensitive
            )

            if server.client_id is not None:
                server.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server.client_id)

            elapsed = time.perf_counter() - start_time
            if elapsed > 600:
                logging.info(f"Prompt executed in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
            else:
                logging.info(f"Prompt executed in {elapsed:.2f} seconds")

        worker.handle_flags(q.get_flags())

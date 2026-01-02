"""Subprocess worker child process entry point."""

import logging
import multiprocessing as mp
import traceback


class IPCMessageServer:
    """IPC-based message server for subprocess workers."""

    def __init__(self, message_queue, client_id=None, sockets_metadata=None):
        self.message_queue = message_queue
        self.client_id = client_id
        self.last_node_id = None
        self.last_prompt_id = None
        self.sockets_metadata = sockets_metadata or {}

    def send_sync(self, event, data, sid=None):
        from protocol import BinaryEventTypes
        from io import BytesIO
        import base64

        if event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA and isinstance(data, tuple):
            preview_image, metadata = data
            image_type, pil_image, max_size = preview_image

            buffer = BytesIO()
            pil_image.save(buffer, format=image_type)

            data = {
                '_serialized': True,
                'image_type': image_type,
                'image_bytes': base64.b64encode(buffer.getvalue()).decode('utf-8'),
                'max_size': max_size,
                'metadata': metadata
            }

        self.message_queue.put_nowait({'event': event, 'data': data, 'sid': sid})

    def queue_updated(self):
        pass


def worker_main(job_queue, result_queue, message_queue, interrupt_event, worker_id):
    """Subprocess worker entry point - spawned fresh for each execution."""
    job_data = None
    try:
        logging.basicConfig(level=logging.INFO, format=f'[Worker-{worker_id}] %(levelname)s: %(message)s')
        logging.info(f"Worker {worker_id} starting (PID: {mp.current_process().pid})")

        import asyncio
        import comfy.model_management
        from comfy.worker_native import NativeWorker
        from comfy.execution_core import WorkerServer

        logging.info(f"Worker {worker_id} initialized. Device: {comfy.model_management.get_torch_device()}")

        job_data = job_queue.get(timeout=30)
        client_id = job_data.get('extra_data', {}).get('client_id')
        client_metadata = job_data.get('client_sockets_metadata', {})

        sockets_metadata = {client_id: client_metadata} if client_id and client_metadata else {}
        ipc_server = IPCMessageServer(message_queue, client_id, sockets_metadata)
        server = WorkerServer(ipc_server)

        def check_interrupt():
            if interrupt_event.is_set():
                raise comfy.model_management.InterruptProcessingException()

        worker = NativeWorker(server, interrupt_checker=check_interrupt)

        import comfy.execution_core
        comfy.execution_core._active_worker = worker

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        node_count = loop.run_until_complete(worker.initialize())
        logging.info(f"Worker {worker_id} loaded {node_count} node types")

        result = worker.execute_prompt(
            job_data['prompt'],
            job_data['prompt_id'],
            job_data.get('extra_data', {}),
            job_data.get('execute_outputs', [])
        )
        result['worker_id'] = worker_id

        logging.info(f"Worker {worker_id} completed successfully")
        result_queue.put(result)

    except Exception as e:
        logging.error(f"Worker {worker_id} failed: {e}\n{traceback.format_exc()}")
        result_queue.put({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'history_result': {},
            'status_messages': [],
            'worker_id': worker_id,
            'prompt_id': job_data.get('prompt_id', 'unknown') if job_data else 'unknown'
        })

    finally:
        logging.info(f"Worker {worker_id} exiting")

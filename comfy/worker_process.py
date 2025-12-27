"""Subprocess worker for isolated prompt execution with complete GPU/ROCm reset."""

import logging
import multiprocessing as mp
import time
import traceback

mp.set_start_method('spawn', force=True)


def _deserialize_preview(msg):
    """Deserialize preview image from IPC transport."""
    if not (isinstance(msg['data'], dict) and msg['data'].get('_serialized')):
        return msg

    from PIL import Image
    from io import BytesIO
    import base64

    s = msg['data']
    pil_image = Image.open(BytesIO(base64.b64decode(s['image_bytes'])))
    msg['data'] = ((s['image_type'], pil_image, s['max_size']), s['metadata'])
    return msg


def _error_result(worker_id, prompt_id, error, tb=None):
    return {
        'success': False,
        'error': error,
        'traceback': tb,
        'history_result': {},
        'status_messages': [],
        'worker_id': worker_id,
        'prompt_id': prompt_id
    }


def _kill_worker(worker, worker_id):
    if not worker.is_alive():
        return
    worker.terminate()
    worker.join(timeout=2)
    if worker.is_alive():
        logging.warning(f"Worker {worker_id} didn't terminate, killing")
        worker.kill()
        worker.join()


class SubprocessWorker:
    """Executes each prompt in an isolated subprocess with fresh GPU context."""

    def __init__(self, server_instance, timeout=600):
        self.server_instance = server_instance
        self.timeout = timeout
        self.worker_counter = 0
        self.current_worker = None
        self.interrupt_event = None
        logging.info("SubprocessWorker created - each job will run in isolated process")

    async def initialize(self):
        """Load node definitions for prompt validation. Returns node count."""
        from comfy.execution_core import init_execution_environment
        return await init_execution_environment()

    def handle_flags(self, flags):
        pass

    def mark_needs_gc(self):
        pass

    def get_gc_timeout(self):
        return 1000.0

    def interrupt(self, value=True):
        if not value:
            return
        if self.interrupt_event:
            self.interrupt_event.set()
        if self.current_worker and self.current_worker.is_alive():
            self.current_worker.join(timeout=2)
            _kill_worker(self.current_worker, self.worker_counter)
            self.current_worker = None

    def _relay_messages(self, message_queue, server):
        """Relay queued messages to UI."""
        while not message_queue.empty():
            try:
                msg = _deserialize_preview(message_queue.get_nowait())
                if server:
                    server.send_sync(msg['event'], msg['data'], msg['sid'])
            except:
                break

    def execute_prompt(self, prompt, prompt_id, extra_data={}, execute_outputs=[], server=None):
        self.worker_counter += 1
        worker_id = self.worker_counter

        job_queue = mp.Queue()
        result_queue = mp.Queue()
        message_queue = mp.Queue()
        self.interrupt_event = mp.Event()

        client_id = extra_data.get('client_id')
        client_metadata = {}
        if client_id and hasattr(server, 'sockets_metadata'):
            client_metadata = server.sockets_metadata.get(client_id, {})

        job_data = {
            'prompt': prompt,
            'prompt_id': prompt_id,
            'extra_data': extra_data,
            'execute_outputs': execute_outputs,
            'client_sockets_metadata': client_metadata
        }

        from comfy.worker_process_child import worker_main
        worker = mp.Process(
            target=worker_main,
            args=(job_queue, result_queue, message_queue, self.interrupt_event, worker_id),
            name=f'ComfyUI-Worker-{worker_id}'
        )

        logging.info(f"Starting worker {worker_id} for prompt {prompt_id}")
        self.current_worker = worker
        worker.start()
        job_queue.put(job_data)

        try:
            start_time = time.time()
            result = None

            while result is None:
                if self.interrupt_event.is_set():
                    logging.info(f"Worker {worker_id} interrupted")
                    if server:
                        server.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server.client_id)
                    return _error_result(worker_id, prompt_id, 'Execution interrupted by user')

                if time.time() - start_time > self.timeout:
                    raise TimeoutError()

                self._relay_messages(message_queue, server)

                try:
                    result = result_queue.get(timeout=0.1)
                except mp.queues.Empty:
                    pass

            self._relay_messages(message_queue, server)

            worker.join(timeout=5)
            if worker.is_alive():
                _kill_worker(worker, worker_id)

            logging.info(f"Worker {worker_id} cleaned up (exit code: {worker.exitcode})")
            self.current_worker = None
            return result

        except TimeoutError:
            error = f"Worker {worker_id} timed out after {self.timeout}s. Try --subprocess-timeout to increase."
            logging.error(error)
            _kill_worker(worker, worker_id)
            self.current_worker = None
            return _error_result(worker_id, prompt_id, error)

        except Exception as e:
            error = f"Worker {worker_id} IPC error: {e}"
            logging.error(f"{error}\n{traceback.format_exc()}")
            _kill_worker(worker, worker_id)
            self.current_worker = None
            return _error_result(worker_id, prompt_id, error, traceback.format_exc())

        finally:
            for q in (job_queue, result_queue, message_queue):
                q.close()
                try:
                    q.join_thread()
                except:
                    pass

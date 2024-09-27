from __future__ import annotations

import asyncio
import contextvars
import gc
import json
import uuid
from asyncio import get_event_loop
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import RLock
from typing import Optional

from opentelemetry import context, propagate
from opentelemetry.context import Context, attach, detach
from opentelemetry.trace import Status, StatusCode

from .client_types import V1QueuePromptResponse
from ..api.components.schema.prompt import PromptDict
from ..cli_args_types import Configuration
from ..cmd.main_pre import tracer
from ..component_model.executor_types import ExecutorToClientProgress, Executor
from ..component_model.make_mutable import make_mutable
from ..distributed.process_pool_executor import ProcessPoolExecutor
from ..distributed.server_stub import ServerStub

_prompt_executor = contextvars.ContextVar('prompt_executor')


def _execute_prompt(
        prompt: dict,
        prompt_id: str,
        client_id: str,
        span_context: dict,
        progress_handler: ExecutorToClientProgress | None,
        configuration: Configuration | None) -> dict:
    span_context: Context = propagate.extract(span_context)
    token = attach(span_context)
    try:
        return __execute_prompt(prompt, prompt_id, client_id, span_context, progress_handler, configuration)
    finally:
        detach(token)


def __execute_prompt(
        prompt: dict,
        prompt_id: str,
        client_id: str,
        span_context: Context,
        progress_handler: ExecutorToClientProgress | None,
        configuration: Configuration | None) -> dict:
    from .. import options
    progress_handler = progress_handler or ServerStub()

    try:
        prompt_executor = _prompt_executor.get()
    except LookupError:
        if configuration is None:
            options.enable_args_parsing()
        else:
            from ..cmd.main_pre import args
            args.clear()
            args.update(configuration)

        from ..cmd.execution import PromptExecutor
        with tracer.start_as_current_span("Initialize Prompt Executor", context=span_context) as span:
            prompt_executor = PromptExecutor(progress_handler, lru_size=configuration.cache_lru if configuration is not None else 0)
            prompt_executor.raise_exceptions = True
            _prompt_executor.set(prompt_executor)

    with tracer.start_as_current_span("Execute Prompt", context=span_context) as span:
        try:
            prompt_mut = make_mutable(prompt)
            from ..cmd.execution import validate_prompt
            validation_tuple = validate_prompt(prompt_mut)
            if not validation_tuple.valid:
                validation_error_dict = {"message": "Unknown", "details": ""} if not validation_tuple.node_errors or len(validation_tuple.node_errors) == 0 else validation_tuple.node_errors
                raise ValueError(json.dumps(validation_error_dict))

            if client_id is None:
                prompt_executor.server = ServerStub()
            else:
                prompt_executor.server = progress_handler

            prompt_executor.execute(prompt_mut, prompt_id, {"client_id": client_id},
                                    execute_outputs=validation_tuple.good_output_node_ids)
            return prompt_executor.outputs_ui
        except Exception as exc_info:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(exc_info)
            raise exc_info


def _cleanup():
    from .. import model_management
    model_management.unload_all_models()
    gc.collect()
    try:
        model_management.soft_empty_cache()
    except:
        pass


class EmbeddedComfyClient:
    """
    Embedded client for comfy executing prompts as a library.

    This client manages a single-threaded executor to run long-running or blocking tasks
    asynchronously without blocking the asyncio event loop. It initializes a PromptExecutor
    in a dedicated thread for executing prompts and handling server-stub communications.
    Example usage:

    Asynchronous (non-blocking) usage with async-await:
    ```
    # Write a workflow, or enable Dev Mode in the UI settings, then Save (API Format) to get the workflow in your
    # workspace.
    prompt_dict = {
      "1": {"class_type": "KSamplerAdvanced", ...}
      ...
    }
    # Validate your workflow (the prompt)
    from comfy.api.components.schema.prompt import Prompt
    prompt = Prompt.validate(prompt_dict)
    # Then use the client to run your workflow. This will start, then stop, a local ComfyUI workflow executor.
    # It does not connect to a remote server.
    async def main():
        async with EmbeddedComfyClient() as client:
            outputs = await client.queue_prompt(prompt)
            print(outputs)
        print("Now that we've exited the with statement, all your VRAM has been cleared from ComfyUI")
    if __name__ == "__main__"
        asyncio.run(main())
    ```

    In order to use this in blocking methods, learn more about asyncio online.
    """

    def __init__(self, configuration: Optional[Configuration] = None, progress_handler: Optional[ExecutorToClientProgress] = None, max_workers: int = 1, executor: Executor = None):
        self._progress_handler = progress_handler or ServerStub()
        self._executor = executor or ThreadPoolExecutor(max_workers=max_workers)
        self._configuration = configuration
        self._is_running = False
        self._task_count_lock = RLock()
        self._task_count = 0

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def task_count(self) -> int:
        return self._task_count

    async def __aenter__(self):
        self._is_running = True
        return self

    async def __aexit__(self, *args):

        while self.task_count > 0:
            await asyncio.sleep(0.1)

        await get_event_loop().run_in_executor(self._executor, _cleanup)

        self._executor.shutdown(wait=True)
        self._is_running = False

    async def queue_prompt_api(self,
                               prompt: PromptDict) -> V1QueuePromptResponse:
        outputs = await self.queue_prompt(prompt)
        return V1QueuePromptResponse(urls=[], outputs=outputs)

    @tracer.start_as_current_span("Queue Prompt")
    async def queue_prompt(self,
                           prompt: PromptDict | dict,
                           prompt_id: Optional[str] = None,
                           client_id: Optional[str] = None) -> dict:
        with self._task_count_lock:
            self._task_count += 1
        prompt_id = prompt_id or str(uuid.uuid4())
        client_id = client_id or self._progress_handler.client_id or None
        span_context = context.get_current()
        carrier = {}
        propagate.inject(carrier, span_context)
        try:
            return await get_event_loop().run_in_executor(
                self._executor,
                _execute_prompt,
                make_mutable(prompt),
                prompt_id,
                client_id,
                carrier,
                # todo: a proxy object or something more sophisticated will have to be done here to restore progress notifications for ProcessPoolExecutors
                None if isinstance(self._executor, ProcessPoolExecutor) else self._progress_handler,
                self._configuration,
            )
        finally:
            with self._task_count_lock:
                self._task_count -= 1

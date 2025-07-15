from __future__ import annotations

import asyncio
import copy
import gc
import json
import threading
import uuid
from asyncio import get_event_loop
from multiprocessing import RLock
from typing import Optional

from opentelemetry import context, propagate
from opentelemetry.context import Context, attach, detach
from opentelemetry.trace import Status, StatusCode

from .client_types import V1QueuePromptResponse
from ..api.components.schema.prompt import PromptDict
from ..cli_args_types import Configuration
from ..cmd.folder_paths import init_default_paths  # pylint: disable=import-error
from ..cmd.main_pre import tracer
from ..component_model.executor_types import ExecutorToClientProgress
from ..component_model.make_mutable import make_mutable
from ..distributed.executors import ContextVarExecutor
from ..distributed.process_pool_executor import ProcessPoolExecutor
from ..distributed.server_stub import ServerStub
from ..execution_context import current_execution_context

_prompt_executor = threading.local()


def _execute_prompt(
        prompt: dict,
        prompt_id: str,
        client_id: str,
        span_context: dict,
        progress_handler: ExecutorToClientProgress | None,
        configuration: Configuration | None) -> dict:
    configuration = copy.deepcopy(configuration) if configuration is not None else None
    execution_context = current_execution_context()
    if len(execution_context.folder_names_and_paths) == 0 or configuration is not None:
        init_default_paths(execution_context.folder_names_and_paths, configuration, replace_existing=True)
    span_context: Context = propagate.extract(span_context)
    token = attach(span_context)
    try:
        # there is never an event loop running on a thread or process pool thread here
        # this also guarantees nodes will be able to successfully call await
        return asyncio.run(__execute_prompt(prompt, prompt_id, client_id, span_context, progress_handler, configuration))
    finally:
        detach(token)


async def __execute_prompt(
        prompt: dict,
        prompt_id: str,
        client_id: str,
        span_context: Context,
        progress_handler: ExecutorToClientProgress | None,
        configuration: Configuration | None) -> dict:
    from .. import options
    from ..cmd.execution import PromptExecutor

    progress_handler = progress_handler or ServerStub()
    prompt_executor: PromptExecutor = None
    try:
        prompt_executor: PromptExecutor = _prompt_executor.executor
    except (LookupError, AttributeError):
        if configuration is None:
            options.enable_args_parsing()
        else:
            from ..cmd.main_pre import args
            args.clear()
            args.update(configuration)

        with tracer.start_as_current_span("Initialize Prompt Executor", context=span_context):
            # todo: deal with new caching features
            prompt_executor = PromptExecutor(progress_handler)
            prompt_executor.raise_exceptions = True
            _prompt_executor.executor = prompt_executor

    with tracer.start_as_current_span("Execute Prompt", context=span_context) as span:
        try:
            prompt_mut = make_mutable(prompt)
            from ..cmd.execution import validate_prompt
            validation_tuple = await validate_prompt(prompt_id, prompt_mut)
            if not validation_tuple.valid:
                if validation_tuple.node_errors is not None and len(validation_tuple.node_errors) > 0:
                    validation_error_dict = validation_tuple.node_errors
                elif validation_tuple.error is not None:
                    validation_error_dict = validation_tuple.error
                else:
                    validation_error_dict = {"message": "Unknown", "details": ""}
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
    from ..cmd.execution import PromptExecutor
    try:
        prompt_executor: PromptExecutor = _prompt_executor.executor
        # this should clear all references to output tensors and make it easier to collect back the memory
        prompt_executor.reset()
    except (LookupError, AttributeError):
        pass
    from .. import model_management
    model_management.unload_all_models()
    gc.collect()
    try:
        model_management.soft_empty_cache()
    except:
        pass


class Comfy:
    """
    This manages a single-threaded executor to run long-running or blocking workflows
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

    def __init__(self, configuration: Optional[Configuration] = None, progress_handler: Optional[ExecutorToClientProgress] = None, max_workers: int = 1, executor: ProcessPoolExecutor | ContextVarExecutor = None):
        self._progress_handler = progress_handler or ServerStub()
        self._executor = executor or ContextVarExecutor(max_workers=max_workers)
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
                               prompt: PromptDict | str | dict) -> V1QueuePromptResponse:
        """
        Queues a prompt for execution, returning the output when it is complete.
        :param prompt: a PromptDict, string or dictionary containing a so-called Workflow API prompt
        :return: a response of URLs for Save-related nodes and the node outputs
        """
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        if isinstance(prompt, dict):
            from comfy.api.components.schema.prompt import Prompt
            prompt = Prompt.validate(prompt)
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


EmbeddedComfyClient = Comfy

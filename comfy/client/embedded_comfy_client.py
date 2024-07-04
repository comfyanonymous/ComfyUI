from __future__ import annotations

import asyncio
import gc
import uuid
from asyncio import get_event_loop
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from opentelemetry import context
from opentelemetry.trace import Span, Status, StatusCode

from ..api.components.schema.prompt import PromptDict
from ..cli_args_types import Configuration
from ..cmd.main_pre import tracer
from ..component_model.executor_types import ExecutorToClientProgress
from ..component_model.make_mutable import make_mutable
from ..distributed.server_stub import ServerStub

_server_stub_instance = ServerStub()


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

    def __init__(self, configuration: Optional[Configuration] = None, progress_handler: Optional[ExecutorToClientProgress] = None, max_workers: int = 1):
        self._progress_handler = progress_handler or ServerStub()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._configuration = configuration
        # we don't want to import the executor yet
        self._prompt_executor: Optional["comfy.cmd.execution.PromptExecutor"] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def __aenter__(self):
        await self._initialize_prompt_executor()
        self._is_running = True
        return self

    async def __aexit__(self, *args):
        # Perform cleanup here
        def cleanup():
            from .. import model_management
            model_management.unload_all_models()
            gc.collect()
            try:
                model_management.soft_empty_cache()
            except:
                pass

        # wait until the queue is done
        while self._executor._work_queue.qsize() > 0:
            await asyncio.sleep(0.1)

        await get_event_loop().run_in_executor(self._executor, cleanup)

        self._executor.shutdown(wait=True)
        self._is_running = False

    async def _initialize_prompt_executor(self):
        # This method must be async since it's used in __aenter__
        def create_executor_in_thread():
            from .. import options
            if self._configuration is None:
                options.enable_args_parsing()
            else:
                from ..cmd.main_pre import args
                args.clear()
                args.update(self._configuration)

            from ..cmd.execution import PromptExecutor

            self._prompt_executor = PromptExecutor(self._progress_handler)
            self._prompt_executor.raise_exceptions = True

        await get_event_loop().run_in_executor(self._executor, create_executor_in_thread)

    @tracer.start_as_current_span("Queue Prompt")
    async def queue_prompt(self,
                           prompt: PromptDict | dict,
                           prompt_id: Optional[str] = None,
                           client_id: Optional[str] = None) -> dict:
        prompt_id = prompt_id or str(uuid.uuid4())
        client_id = client_id or self._progress_handler.client_id or None
        span_context = context.get_current()

        def execute_prompt() -> dict:
            spam: Span
            with tracer.start_as_current_span("Execute Prompt", context=span_context) as span:
                from ..cmd.execution import PromptExecutor, validate_prompt
                try:
                    prompt_mut = make_mutable(prompt)
                    validation_tuple = validate_prompt(prompt_mut)
                    if not validation_tuple[0]:
                        validation_error_dict = validation_tuple[1] or {"message": "Unknown", "details": ""}
                        raise ValueError("\n".join([validation_error_dict["message"], validation_error_dict["details"]]))

                    prompt_executor: PromptExecutor = self._prompt_executor

                    if client_id is None:
                        prompt_executor.server = _server_stub_instance
                    else:
                        prompt_executor.server = self._progress_handler

                    prompt_executor.execute(prompt_mut, prompt_id, {"client_id": client_id},
                                            execute_outputs=validation_tuple[2])
                    return prompt_executor.outputs_ui
                except Exception as exc_info:
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(exc_info)
                    raise exc_info

        return await get_event_loop().run_in_executor(self._executor, execute_prompt)

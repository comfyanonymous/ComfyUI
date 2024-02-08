from __future__ import annotations

import asyncio
import gc
import uuid
from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from ..api.components.schema.prompt import PromptDict
from ..cli_args_types import Configuration
from ..component_model.make_mutable import make_mutable
from ..component_model.queue_types import BinaryEventTypes
from ..component_model.executor_types import ExecutorToClientProgress, StatusMessage, ExecutingMessage


class ServerStub(ExecutorToClientProgress):
    """
    This class is a stub implementation of ExecutorToClientProgress. This will handle progress events.
    """
    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.last_node_id = None
        self.last_prompt_id = None

    def send_sync(self,
                  event: Literal["status", "executing"] | BinaryEventTypes | str | None,
                  data: StatusMessage | ExecutingMessage | bytes | bytearray | None, sid: str | None = None):
        pass

    def queue_updated(self):
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

    def __init__(self, configuration: Optional[Configuration] = None, loop: Optional[AbstractEventLoop] = None,
                 max_workers: int = 1):
        self._server_stub = ServerStub()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._loop = loop or asyncio.get_event_loop()
        self._configuration = configuration
        # we don't want to import the executor yet
        self._prompt_executor: Optional["comfy.cmd.execution.PromptExecutor"] = None

    async def __aenter__(self):
        # Perform asynchronous initialization here, if needed
        await self._initialize_prompt_executor()
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

        await self._loop.run_in_executor(self._executor, cleanup)

        self._executor.shutdown(wait=True)

    async def _initialize_prompt_executor(self):
        # This method must be async since it's used in __aenter__
        def create_executor_in_thread():
            from .. import options
            if self._configuration is None:
                options.enable_args_parsing()
            else:
                from ..cli_args import args
                args.clear()
                args.update(self._configuration)

            from ..cmd.execution import PromptExecutor

            self._prompt_executor = PromptExecutor(self._server_stub)

        await self._loop.run_in_executor(self._executor, create_executor_in_thread)

    async def queue_prompt(self, prompt: PromptDict) -> dict:
        prompt_id = str(uuid.uuid4())

        def execute_prompt() -> dict:
            from ..cmd.execution import validate_prompt
            prompt_mut = make_mutable(prompt)
            validation_tuple = validate_prompt(prompt_mut)

            self._prompt_executor.execute(prompt_mut, prompt_id, {"client_id": self._server_stub.client_id},
                                          execute_outputs=validation_tuple[2])
            if self._prompt_executor.success:
                return self._prompt_executor.outputs_ui
            else:
                raise RuntimeError("\n".join(event for (event, data) in self._prompt_executor.status_messages))

        return await self._loop.run_in_executor(self._executor, execute_prompt)

from __future__ import annotations

from ..cmd.main_pre import tracer

import asyncio
import concurrent.futures
import contextlib
import copy
import gc
import json
import logging
import threading
import uuid
from asyncio import get_event_loop
from multiprocessing import RLock
from typing import Optional, Literal

from opentelemetry import context, propagate
from opentelemetry.context import Context, attach, detach
from opentelemetry.trace import Status, StatusCode

from .async_progress_iterable import QueuePromptWithProgress
from .client_types import V1QueuePromptResponse
from ..api.components.schema.prompt import PromptDict
from ..cli_args_types import Configuration
from ..cli_args import default_configuration
from ..cmd.folder_paths import init_default_paths  # pylint: disable=import-error
from ..component_model.executor_types import ExecutorToClientProgress
from ..component_model.make_mutable import make_mutable
from ..component_model.queue_types import QueueItem, ExecutionStatus, TaskInvocation, QueueTuple, ExtraData
from ..distributed.executors import ContextVarExecutor
from ..distributed.history import History
from ..distributed.process_pool_executor import ProcessPoolExecutor
from ..distributed.server_stub import ServerStub
from ..component_model.configuration import MODEL_MANAGEMENT_ARGS, requires_process_pool_executor

_prompt_executor = threading.local()

logger = logging.getLogger(__name__)


def _execute_prompt(
        prompt: dict,
        prompt_id: str,
        client_id: str,
        span_context: dict,
        progress_handler: ExecutorToClientProgress | None,
        configuration: Configuration | None,
        partial_execution_targets: Optional[list[str]] = None) -> dict:
    configuration = copy.deepcopy(configuration) if configuration is not None else None
    from ..execution_context import current_execution_context
    execution_context = current_execution_context()
    if len(execution_context.folder_names_and_paths) == 0 or configuration is not None:
        init_default_paths(execution_context.folder_names_and_paths, configuration, replace_existing=True)
    span_context: Context = propagate.extract(span_context)
    token = attach(span_context)
    try:
        # there is never an event loop running on a thread or process pool thread here
        # this also guarantees nodes will be able to successfully call await
        return asyncio.run(__execute_prompt(prompt, prompt_id, client_id, span_context, progress_handler, configuration, partial_execution_targets))
    finally:
        detach(token)


async def __execute_prompt(
        prompt: dict,
        prompt_id: str,
        client_id: str,
        span_context: Context,
        progress_handler: ExecutorToClientProgress | None,
        configuration: Configuration | None,
        partial_execution_targets: list[str] | None) -> dict:
    from ..execution_context import context_configuration
    with context_configuration(configuration):
        return await ___execute_prompt(prompt, prompt_id, client_id, span_context, progress_handler, partial_execution_targets)


async def ___execute_prompt(
        prompt: dict,
        prompt_id: str,
        client_id: str,
        span_context: Context,
        progress_handler: ExecutorToClientProgress | None,
        partial_execution_targets: list[str] | None) -> dict:
    from ..cmd.execution import PromptExecutor

    progress_handler = progress_handler or ServerStub()
    prompt_executor: PromptExecutor = None
    try:
        prompt_executor: PromptExecutor = _prompt_executor.executor
    except (LookupError, AttributeError):
        with tracer.start_as_current_span("Initialize Prompt Executor", context=span_context):
            # todo: deal with new caching features
            prompt_executor = PromptExecutor(progress_handler)
            prompt_executor.raise_exceptions = True
            _prompt_executor.executor = prompt_executor

    with tracer.start_as_current_span("Execute Prompt", context=span_context) as span:
        try:
            prompt_mut = make_mutable(prompt)
            from ..cmd.execution import validate_prompt
            validation_tuple = await validate_prompt(prompt_id, prompt_mut, partial_execution_targets)
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

            await prompt_executor.execute_async(prompt_mut, prompt_id, {"client_id": client_id},
                                                execute_outputs=validation_tuple.good_output_node_ids)
            return prompt_executor.outputs_ui
        except Exception as exc_info:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(exc_info)
            raise exc_info


def _cleanup(invalidate_nodes=True):
    from ..cmd.execution import PromptExecutor
    from ..nodes_context import invalidate
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
    if invalidate_nodes:
        try:
            invalidate()
        except:
            pass


class Comfy:
    """
    A client for running ComfyUI workflows within a Python application.

    This client allows you to execute ComfyUI workflows (in API JSON format) programmatically.
    It manages the execution environment, including model loading and resource cleanup.

    ### Configuration and Executors

    ComfyUI relies on global state for model management (e.g., loaded models in VRAM). To handle this safely, `Comfy`
    executes workflows using one of two strategies based on your `configuration`:

    1.  **ContextVarExecutor (Default)**: Runs in a thread pool within the current process.
        -   **Pros**: Efficient, low overhead.
        -   **Cons**: Modifies global state in the current process.
        -   **Use Case**: Standard workflows where you are happy with the default ComfyUI settings or sharing state.

    2.  **ProcessPoolExecutor**: Runs in a separate process.
        -   **Pros**: Complete isolation. Configuration changes (like `lowvram`) do not affect the main process.
        -   **Cons**: Higher overhead (process startup).
        -   **Use Case**: Required when `configuration` overrides arguments that affect global model management state.
            These arguments include: `lowvram`, `highvram`, `cpu`, `gpu_only`, `deterministic`, `directml`,
            various `fp8`/`fp16`/`bf16` settings, and attention optimizations (e.g., `use_flash_attention`).

    The client automatically selects `ProcessPoolExecutor` if you provide a `configuration` that modifies any of these
    global settings, unless you explicitly pass an `executor`.

    ### Parameters

    -   **configuration** (`Optional[Configuration]`): A dictionary of arguments to override defaults.
        See `comfy.cli_args_types.Configuration`.
        Example: `{"lowvram": True}` or `{"gpu_only": True}`.
    -   **progress_handler** (`Optional[ExecutorToClientProgress]`): callback handler for progress updates and previews.
    -   **max_workers** (`int`): Maximum number of concurrent workflows (default: 1).
    -   **executor** (`Optional[Union[Executor, str]]`): Explicitly define the executor to use.
        -   Pass an instance of `ProcessPoolExecutor` or `ContextVarExecutor`.
        -   Pass the string `"ProcessPoolExecutor"` or `"ContextVarExecutor"` to force initialization of that type.
        -   If `None` (default), the best executor is chosen based on `configuration`.

    ### Examples

    #### 1. Running a Workflow (Basic)

    This example executes a simple workflow and prints the path of the saved image.

    ```python
    import asyncio
    from comfy.client.embedded_comfy_client import Comfy

    # A simple API format workflow (simplified for brevity)
    prompt_dict = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 8566257, "steps": 20, "cfg": 8, "sampler_name": "euler",
                "scheduler": "normal", "denoise": 1,
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "masterpiece best quality girl", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad hands", "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI_API", "images": ["8", 0]}}
    }

    async def main():
        # Using default configuration (runs in-process)
        async with Comfy() as client:
            # Queue the prompt and await the result
            outputs = await client.queue_prompt(prompt_dict)

            # Retrieve the output path from the SaveImage node (Node ID "9")
            image_path = outputs["9"]["images"][0]["abs_path"]
            print(f"Image saved to: {image_path}")

    # asyncio.run(main())
    ```

    #### 2. Using Custom Configuration (Isolated Process)

    To run with specific settings like `lowvram`, pass the configuration. This implies `ProcessPoolExecutor`.

    ```python
    async def run_lowvram():
        # This will spawn a new process with lowvram enabled
        async with Comfy(configuration={"lowvram": True}) as client:
            outputs = await client.queue_prompt(prompt_dict)
            print("Finished lowvram generation")
    ```

    #### 3. Programmatically Building Workflows

    You can use `GraphBuilder` constructing workflows with a more pythonic API.

    ```python
    from comfy_execution.graph_utils import GraphBuilder

    def build_graph():
        builder = GraphBuilder()
        checkpoint = builder.node("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")
        latent = builder.node("EmptyLatentImage", width=512, height=512, batch_size=1)
        pos = builder.node("CLIPTextEncode", text="masterpiece", clip=checkpoint.out(1))
        neg = builder.node("CLIPTextEncode", text="bad quality", clip=checkpoint.out(1))
        
        sampler = builder.node("KSampler", 
            seed=42, steps=20, cfg=8, sampler_name="euler", scheduler="normal", denoise=1,
            model=checkpoint.out(0), positive=pos.out(0), negative=neg.out(0), latent_image=latent.out(0)
        )
        vae = builder.node("VAEDecode", samples=sampler.out(0), vae=checkpoint.out(2))
        builder.node("SaveImage", filename_prefix="Generated", images=vae.out(0))
        return builder.finalize()

    async def run_builder():
        prompt = build_graph()
        async with Comfy() as client:
            await client.queue_prompt(prompt)
    ```

    #### 4. Streaming Progress and Previews

    To receive real-time progress updates and preview images (e.g., step-by-step decoding).

    ```python
    from comfy.component_model.queue_types import BinaryEventTypes

    async def run_streaming():
        async with Comfy() as client:
            # Get a task that supports progress iteration
            task = client.queue_with_progress(prompt_dict)
            
            async for notification in task.progress():
                if notification.event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
                    # 'data' contains the PIL Image and metadata
                    image, metadata = notification.data
                    print(f"Received preview: {image.size}")
                elif notification.event == "progress":
                    print(f"Step: {notification.data['value']}/{notification.data['max']}")

            # Await final result
            result = await task.get()
    ```
    """

    def __init__(self, configuration: Optional[Configuration] = None, progress_handler: Optional[ExecutorToClientProgress] = None, max_workers: int = 1, executor: ProcessPoolExecutor | ContextVarExecutor | Literal["ProcessPoolExecutor", "ContextVarExecutor"] = None):
        self._progress_handler = progress_handler or ServerStub()
        self._default_configuration = default_configuration()
        self._configuration = configuration

        need_process_pool = requires_process_pool_executor(configuration)

        if executor is None:
            if need_process_pool:
                self._executor = ProcessPoolExecutor(max_workers=max_workers)
                self._owns_executor = True
            else:
                self._executor = ContextVarExecutor(max_workers=max_workers)
                self._owns_executor = True
        elif isinstance(executor, str):
            self._owns_executor = True
            if executor == "ProcessPoolExecutor":
                self._executor = ProcessPoolExecutor(max_workers=max_workers)
            elif executor == "ContextVarExecutor":
                if need_process_pool:
                    raise ValueError(f"Configuration requires ProcessPoolExecutor but ContextVarExecutor was requested. Configuration keys causing this: {[k for k in MODEL_MANAGEMENT_ARGS if configuration.get(k) != self._default_configuration.get(k)]}")
                self._executor = ContextVarExecutor(max_workers=max_workers)
            else:
                raise ValueError(f"Unknown executor type string: {executor}")
        else:
            # Executor instance passed
            self._owns_executor = False
            self._executor = executor
            if need_process_pool and not isinstance(executor, ProcessPoolExecutor):
                raise ValueError(f"Configuration requires ProcessPoolExecutor but {type(executor).__name__} was passed. Configuration keys causing this: {[k for k in MODEL_MANAGEMENT_ARGS if configuration.get(k) != self._default_configuration.get(k)]}")

        self._is_running = False
        self._task_count_lock = RLock()
        self._task_count = 0
        self._history = History()
        self._exit_stack = None
        self._async_exit_stack = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def task_count(self) -> int:
        return self._task_count

    def __enter__(self):
        self._exit_stack = contextlib.ExitStack()
        self._is_running = True
        from ..execution_context import context_configuration
        cm = context_configuration(self._configuration)
        self._exit_stack.enter_context(cm)
        if self._owns_executor:
            self._exit_stack.enter_context(self._executor)
        return self

    @property
    def history(self) -> History:
        return self._history

    async def clear_cache(self):
        await get_event_loop().run_in_executor(self._executor, _cleanup, False)

    def __exit__(self, *args):
        get_event_loop().run_in_executor(self._executor, _cleanup)
        self._is_running = False
        self._exit_stack.__exit__(*args)

    async def __aenter__(self):
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._is_running = True
        from ..execution_context import context_configuration
        cm = context_configuration(self._configuration)
        self._async_exit_stack.enter_context(cm)
        if self._owns_executor:
            self._async_exit_stack.enter_context(self._executor)
        return self

    async def __aexit__(self, *args):

        while self.task_count > 0:
            await asyncio.sleep(0.1)

        await get_event_loop().run_in_executor(self._executor, _cleanup)

        self._is_running = False
        await self._async_exit_stack.__aexit__(*args)

    async def queue_prompt_api(self,
                               prompt: PromptDict | str | dict,
                               progress_handler: Optional[ExecutorToClientProgress] = None,
                               prompt_id: Optional[str] = None) -> V1QueuePromptResponse:
        """
        Queues a prompt for execution, returning the output when it is complete.
        :param prompt: a PromptDict, string or dictionary containing a so-called Workflow API prompt
        :return: a response of URLs for Save-related nodes and the node outputs
        """
        prompt_id = prompt_id or str(uuid.uuid4())
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        if isinstance(prompt, dict):
            from ..api.components.schema.prompt import Prompt
            prompt = Prompt.validate(prompt)
        outputs = await self.queue_prompt(prompt, progress_handler=progress_handler, prompt_id=prompt_id)
        return V1QueuePromptResponse(urls=[], outputs=outputs, prompt_id=prompt_id)

    def queue_with_progress(self, prompt: PromptDict | str | dict) -> QueuePromptWithProgress:
        """
        Queues a prompt with progress notifications.

        >>> from comfy.client.embedded_comfy_client import Comfy
        >>> from comfy.client.client_types import ProgressNotification
        >>> async with Comfy() as comfy:
        >>>     task = comfy.queue_with_progress({ ... })
        >>>     # Raises an exception while iterating
        >>>     notification: ProgressNotification
        >>>     async for notification in task.progress():
        >>>         print(notification.data)
        >>>     # If you get this far, no errors occurred.
        >>>     result = await task.get()
        :param prompt:
        :return:
        """
        handler = QueuePromptWithProgress()
        task = asyncio.create_task(self.queue_prompt_api(prompt, progress_handler=handler.progress_handler))
        task.add_done_callback(handler.complete)
        return handler

    @tracer.start_as_current_span("Queue Prompt")
    async def queue_prompt(self,
                           prompt: PromptDict | dict,
                           prompt_id: Optional[str] = None,
                           client_id: Optional[str] = None,
                           partial_execution_targets: Optional[list[str]] = None,
                           progress_handler: Optional[ExecutorToClientProgress] = None) -> dict:
        if isinstance(self._executor, ProcessPoolExecutor) and progress_handler is not None:
            logger.debug(f"a progress_handler={progress_handler} was passed, it must be pickleable to support ProcessPoolExecutor")
        progress_handler = progress_handler or self._progress_handler
        with self._task_count_lock:
            self._task_count += 1
        prompt_id = prompt_id or str(uuid.uuid4())
        assert prompt_id is not None
        client_id = client_id or self._progress_handler.client_id or None
        span_context = context.get_current()
        carrier = {}
        propagate.inject(carrier, span_context)
        # setup history
        prompt = make_mutable(prompt)

        try:
            outputs = await get_event_loop().run_in_executor(
                self._executor,
                _execute_prompt,
                prompt,
                prompt_id,
                client_id,
                carrier,
                # todo: a proxy object or something more sophisticated will have to be done here to restore progress notifications for ProcessPoolExecutors
                None if isinstance(self._executor, ProcessPoolExecutor) else progress_handler,
                self._configuration,
                partial_execution_targets,
            )

            fut = concurrent.futures.Future()
            fut.set_result(TaskInvocation(prompt_id, copy.deepcopy(outputs), ExecutionStatus('success', True, [])))
            self._history.put(QueueItem(queue_tuple=QueueTuple(float(self._task_count), prompt_id, prompt, ExtraData(), [], {}), completed=fut), outputs, ExecutionStatus('success', True, []))
            return outputs
        except Exception as exc_info:
            fut = concurrent.futures.Future()
            fut.set_exception(exc_info)
            self._history.put(QueueItem(queue_tuple=QueueTuple(float(self._task_count), prompt_id, prompt, ExtraData(), [], {}), completed=fut), {}, ExecutionStatus('error', False, [str(exc_info)]))
            raise exc_info
        finally:
            with self._task_count_lock:
                self._task_count -= 1

    def __str__(self):
        diff = {k: v for k, v in (self._configuration or {}).items() if v != self._default_configuration.get(k)}
        return f"<Comfy task_count={self.task_count} configuration={diff} executor={self._executor}>"


EmbeddedComfyClient = Comfy

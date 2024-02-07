import asyncio
from asyncio import AbstractEventLoop
from typing import Optional

from aio_pika import connect_robust
from aio_pika.patterns import RPC

from ..api.components.schema.prompt import Prompt
from ..cli_args_types import Configuration
from ..client.embedded_comfy_client import EmbeddedComfyClient
from ..component_model.queue_types import TaskInvocation, QueueTuple, QueueItem, ExecutionStatus


class DistributedPromptWorker:
    """
    A work in progress distributed prompt worker.
    """

    def __init__(self, embedded_comfy_client: EmbeddedComfyClient,
                 connection_uri: str = "amqp://localhost:5672/",
                 queue_name: str = "comfyui",
                 loop: Optional[AbstractEventLoop] = None, configuration: Configuration = None):
        self._queue_name = queue_name
        self._configuration = configuration
        self._connection_uri = connection_uri
        self._loop = loop or asyncio.get_event_loop()
        self._embedded_comfy_client = embedded_comfy_client

    async def _do_work_item(self, item: QueueTuple) -> TaskInvocation:
        item_without_completer = QueueItem(item, completed=None)
        try:
            output_dict = await self._embedded_comfy_client.queue_prompt(Prompt.validate(item_without_completer.prompt))
            return TaskInvocation(item_without_completer.prompt_id, outputs=output_dict,
                                  status=ExecutionStatus("success", True, []))
        except Exception as e:
            return TaskInvocation(item_without_completer.prompt_id, outputs={},
                                  status=ExecutionStatus("error", False, [str(e)]))

    async def __aenter__(self) -> "DistributedPromptWorker":
        self._connection = await connect_robust(self._connection_uri, loop=self._loop)
        self._channel = await self._connection.channel()
        self._rpc = await RPC.create(channel=self._channel)
        self._rpc.host_exceptions = True
        await self._rpc.register(self._queue_name, self._do_work_item)
        return self

    async def __aexit__(self, *args):
        await self._rpc.close()
        await self._channel.close()
        await self._connection.close()

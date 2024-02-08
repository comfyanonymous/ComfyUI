import asyncio
from asyncio import AbstractEventLoop
from contextlib import AsyncExitStack
from dataclasses import asdict
from typing import Optional

from aio_pika import connect_robust
from aio_pika.patterns import JsonRPC

from .distributed_types import RpcRequest, RpcReply
from ..client.embedded_comfy_client import EmbeddedComfyClient
from ..component_model.queue_types import ExecutionStatus


class DistributedPromptWorker:
    """
    A work in progress distributed prompt worker.
    """

    def __init__(self, embedded_comfy_client: Optional[EmbeddedComfyClient] = None,
                 connection_uri: str = "amqp://localhost:5672/",
                 queue_name: str = "comfyui",
                 loop: Optional[AbstractEventLoop] = None):
        self._exit_stack = AsyncExitStack()
        self._queue_name = queue_name
        self._connection_uri = connection_uri
        self._loop = loop or asyncio.get_event_loop()
        self._embedded_comfy_client = embedded_comfy_client or EmbeddedComfyClient()

    async def _do_work_item(self, request: dict) -> dict:
        try:
            request_obj = RpcRequest.from_dict(request)
        except Exception as e:
            request_dict_prompt_id_recovered = request["prompt_id"] \
                if request is not None and "prompt_id" in request else ""
            return asdict(RpcReply(request_dict_prompt_id_recovered, "", {},
                            ExecutionStatus("error", False, [str(e)])))
        try:
            output_dict = await self._embedded_comfy_client.queue_prompt(request_obj.prompt,
                                                                         request_obj.prompt_id,
                                                                         client_id=request_obj.user_id)
            return asdict(RpcReply(request_obj.prompt_id, request_obj.user_token, output_dict, ExecutionStatus("success", True, [])))
        except Exception as e:
            return asdict(RpcReply(request_obj.prompt_id, request_obj.user_token, {}, ExecutionStatus("error", False, [str(e)])))

    async def __aenter__(self) -> "DistributedPromptWorker":
        await self._exit_stack.__aenter__()
        if not self._embedded_comfy_client.is_running:
            await self._exit_stack.enter_async_context(self._embedded_comfy_client)

        self._connection = await connect_robust(self._connection_uri, loop=self._loop)
        self._channel = await self._connection.channel()
        self._rpc = await JsonRPC.create(channel=self._channel)
        self._rpc.host_exceptions = True

        await self._rpc.register(self._queue_name, self._do_work_item)
        return self

    async def __aexit__(self, *args):
        await self._rpc.close()
        await self._channel.close()
        await self._connection.close()
        return await self._exit_stack.__aexit__(*args)

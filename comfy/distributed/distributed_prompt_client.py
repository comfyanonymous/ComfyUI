import asyncio
from asyncio import AbstractEventLoop
from dataclasses import asdict
from typing import Optional

from aio_pika import connect_robust
from aio_pika.patterns import RPC

from comfy.distributed.distributed_types import RpcRequest, RpcReply


class DistributedPromptClient:
    def __init__(self, queue_name: str = "comfyui",
                 connection_uri="amqp://localhost/",
                 loop: Optional[AbstractEventLoop] = None):
        self.queue_name = queue_name
        self.connection_uri = connection_uri
        self.loop = loop or asyncio.get_event_loop()

    async def __aenter__(self):
        self.connection = await connect_robust(self.connection_uri, loop=self.loop)
        self.channel = await self.connection.channel()
        self.rpc = await RPC.create(channel=self.channel)
        self.rpc.host_exceptions = True

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.channel.close()
        await self.rpc.close()
        await self.connection.close()

    async def queue_prompt(self, request: RpcRequest) -> RpcReply:
        return RpcReply(**(await self.rpc.call(self.queue_name, {"request": asdict(request)})))

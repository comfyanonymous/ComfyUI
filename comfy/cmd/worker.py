import asyncio

from .. import options
from ..distributed.distributed_prompt_worker import DistributedPromptWorker

options.enable_args_parsing()

from ..cli_args import args


async def main():
    # assume we are a worker
    args.distributed_queue_worker = True
    args.distributed_queue_frontend = False
    assert args.distributed_queue_connection_uri is not None, "Set the --distributed-queue-connection-uri argument to your RabbitMQ server"

    async with DistributedPromptWorker(connection_uri=args.distributed_queue_connection_uri,
                                       queue_name=args.distributed_queue_name):
        stop = asyncio.Event()
        try:
            await stop.wait()
        except asyncio.CancelledError:
            pass


def entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()

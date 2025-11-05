import asyncio

from ..cmd.main_pre import args
from ..component_model.file_counter import cleanup_temp
from ..component_model.entrypoints_common import configure_application_paths, executor_from_args


async def main():
    # assume we are a worker
    from ..distributed.distributed_prompt_worker import DistributedPromptWorker

    args.distributed_queue_worker = True
    args.distributed_queue_frontend = False

    # in workers, there is a different default
    if args.block_runtime_package_installation is None:
        args.block_runtime_package_installation = True

    assert args.distributed_queue_connection_uri is not None, "Set the --distributed-queue-connection-uri argument to your RabbitMQ server"

    configure_application_paths(args)
    executor = await executor_from_args(args)

    async with (
        DistributedPromptWorker(connection_uri=args.distributed_queue_connection_uri,
                                queue_name=args.distributed_queue_name,
                                executor=executor),
    ):
        with cleanup_temp():
            stop = asyncio.Event()
            try:
                await stop.wait()
            except (asyncio.CancelledError, InterruptedError, KeyboardInterrupt):
                pass


def entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()

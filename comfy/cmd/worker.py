import asyncio
import os

from .. import options

options.enable_args_parsing()

from ..cli_args import args


async def main():
    # assume we are a worker
    args.distributed_queue_worker = True
    args.distributed_queue_frontend = False
    assert args.distributed_queue_connection_uri is not None, "Set the --distributed-queue-connection-uri argument to your RabbitMQ server"


    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        print("Set cuda device to:", args.cuda_device)

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    # configure paths
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        print(f"Setting output directory to: {output_dir}")
        from ..cmd import folder_paths
        
        folder_paths.set_output_directory(output_dir)
    
    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        print(f"Setting input directory to: {input_dir}")
        from ..cmd import folder_paths
    
        folder_paths.set_input_directory(input_dir)

    if args.temp_directory:
        temp_dir = os.path.abspath(args.temp_directory)
        print(f"Setting temp directory to: {temp_dir}")
        from ..cmd import folder_paths

        folder_paths.set_temp_directory(temp_dir)

    from ..distributed.distributed_prompt_worker import DistributedPromptWorker
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

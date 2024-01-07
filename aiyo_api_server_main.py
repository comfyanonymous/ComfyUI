

import comfy.options
comfy.options.enable_args_parsing()

import os
import folder_paths
import aiyo_api_server.aiyo_api_server
from framework.app_log import AppLog

# Main code
from comfy.cli_args import args



def aiyo_server_main():
    AppLog.init()
    
    server = aiyo_api_server.aiyo_api_server.AIYoApiServer()
    
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        AppLog.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    call_on_start = None
    try:
        server.start(address=args.listen, port=args.port, verbose=True, call_on_start=call_on_start)
    except KeyboardInterrupt:
        AppLog.info("\nStopped server")


if __name__ == "__main__":
    aiyo_server_main()
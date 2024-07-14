import os

cli_mode_flag = os.path.join(os.path.dirname(__file__), '.enable-cli-only-mode')

if not os.path.exists(cli_mode_flag):
    from .glob import manager_server
    WEB_DIRECTORY = "js"
else:
    print(f"\n[ComfyUI-Manager] !! cli-only-mode is enabled !!\n")

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']




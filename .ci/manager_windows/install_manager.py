import sys
import os

repo_path = str(sys.argv[1])
repo_manager_req_path = os.path.join(repo_path, "manager_requirements.txt")

if os.path.exists(repo_manager_req_path):
    import subprocess
    # if not installed, we get 'WARNING: Package(s) not found: comfyui_manager'
    # if installed, there will be a line like 'Version: 0.1.0' = False
    try:
        output = subprocess.check_output([sys.executable, '-s', '-m', 'pip', 'show', 'comfyui_manager'])
        if 'Version:' in output.decode('utf-8'):
            print("comfyui_manager is already installed, will attempt to update to matching version of ComfyUI.")  # noqa: T201
        else:
            print("comfyui_manager is not installed, will install it now.")  # noqa: T201
    except:
        pass

    try:
        subprocess.check_call([sys.executable, '-s', '-m', 'pip', 'install', '-r', repo_manager_req_path])
        print("comfyui_manager installed successfully.")  # noqa: T201
    except:
        print("Failed to install comfyui_manager, please install it manually.")  # noqa: T201

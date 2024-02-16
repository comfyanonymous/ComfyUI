import os
import sys
import threading
import locale
import subprocess  # don't remove this
from tqdm.auto import tqdm
from urllib.parse import urlparse
from datetime import datetime
import platform
import subprocess
import time
import git
from git.remote import RemoteProgress

scanner_path = os.path.dirname(__file__)
root_path = os.path.dirname(os.path.dirname(scanner_path))
comfy_path = os.path.join(root_path,'comfyui-fork')
communication_file = os.path.join(comfy_path, 'communication_file.txt') 
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")

class GitProgress(RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def update(self, op_code, cur_count, max_count=None, message=''):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.pos = 0
        self.pbar.refresh()

def gitclone_install(files):
    print(f"install: {files}")
    for url in files:
        # if not is_valid_url(url):
        #     print(f"Invalid git url: '{url}'")
        #     return False

        if url.endswith("/"):
            url = url[:-1]
        try:
            print(f"Download: git clone '{url}'")
            repo_name = os.path.splitext(os.path.basename(url))[0]
            repo_path = os.path.join(custom_nodes_path, repo_name)

            # Clone the repository from the remote URL
            if None:
            # if platform.system() == 'Windows':
                res = run_script([sys.executable, git_script_path, "--clone", custom_nodes_path, url])
                if res != 0:
                    return False
            else:
                repo = git.Repo.clone_from(url, repo_path, recursive=True, progress=GitProgress())
                repo.git.clear_cache()
                repo.close()

            if not execute_install_script(url, repo_path):
                return False

        except Exception as e:
            print(f"Install(git-clone) error: {url} / {e}", file=sys.stderr)
            return False

    print("Installation was successful.")
    return True

def run_script(cmd, cwd='.'):
    if len(cmd) > 0 and cmd[0].startswith("#"):
        print(f"[ComfyUI-Manager] Unexpected behavior: `{cmd}`")
        return 0

    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    stdout_thread = threading.Thread(target=handle_stream, args=(process.stdout, ""))
    stderr_thread = threading.Thread(target=handle_stream, args=(process.stderr, "[!]"))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()


def handle_stream(stream, prefix):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')
    for msg in stream:
        if prefix == '[!]' and ('it/s]' in msg or 's/it]' in msg) and ('%|' in msg or 'it [' in msg):
            if msg.startswith('100%'):
                print('\r' + msg, end="", file=sys.stderr),
            else:
                print('\r' + msg[:-1], end="", file=sys.stderr),
        else:
            if prefix == '[!]':
                print(prefix, msg, end="", file=sys.stderr)
            else:
                print(prefix, msg, end="")

def execute_install_script(url, repo_path, lazy_mode=False):
    install_script_path = os.path.join(repo_path, "install.py")
    requirements_path = os.path.join(repo_path, "requirements.txt")

    if lazy_mode:
        install_cmd = ["#LAZY-INSTALL-SCRIPT",  sys.executable]
        try_install_script(url, repo_path, install_cmd)
    else:
        if os.path.exists(requirements_path):
            print("Install: pip packages")
            with open(requirements_path, "r") as requirements_file:
                for line in requirements_file:
                    package_name = line.strip()
                    if package_name:
                        install_cmd = [sys.executable, "-m", "pip", "install", package_name]
                        if package_name.strip() != "":
                            try_install_script(url, repo_path, install_cmd)

        if os.path.exists(install_script_path):
            print(f"Install: install script")
            install_cmd = [sys.executable, "install.py"]
            try_install_script(url, repo_path, install_cmd)

    return True

def try_install_script(url, repo_path, install_cmd):
    if None:
    # if platform.system() == "Windows" and comfy_ui_commit_datetime.date() >= comfy_ui_required_commit_datetime.date():
        # if not os.path.exists(startup_script_path):
        #     os.makedirs(startup_script_path)

        # script_path = os.path.join(startup_script_path, "install-scripts.txt")
        # with open(script_path, "a") as file:
        #     obj = [repo_path] + install_cmd
        #     file.write(f"{obj}\n")

        return True
    else:
        print(f"\n## ComfyUI-Manager: EXECUTE => {install_cmd}")
        code = run_script(install_cmd, cwd=repo_path)

        # if platform.system() == "Windows":
        #     try:
        #         if comfy_ui_commit_datetime.date() < comfy_ui_required_commit_datetime.date():
        #             print("\n\n###################################################################")
        #             print(f"[WARN] ComfyUI-Manager: Your ComfyUI version ({comfy_ui_revision})[{comfy_ui_commit_datetime.date()}] is too old. Please update to the latest version.")
        #             print(f"[WARN] The extension installation feature may not work properly in the current installed ComfyUI version on Windows environment.")
        #             print("###################################################################\n\n")
        #     except:
        #         pass

        if code != 0:
            if url is None:
                url = os.path.dirname(repo_path)
            print(f"install script failed: {url}")
            return False

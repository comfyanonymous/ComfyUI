import asyncio
import os
import stat
import subprocess
import threading
from asyncio import Task
from typing import NamedTuple

import requests

from ..cmd.folder_paths import init_default_paths, folder_names_and_paths
# experimental workarounds for colab
from ..cmd.main import _start_comfyui
from ..execution_context import *


class _ColabTuple(NamedTuple):
    tunnel: "CloudflaredTunnel"
    server: Task


_colab_instances: list[_ColabTuple] = []


class CloudflaredTunnel:
    """
    A class to manage a cloudflared tunnel subprocess.

    Provides methods to start, stop, and manage the lifecycle of the tunnel.
    It can be used as a context manager.
    """

    def __init__(self, port: int):
        self._port: int = port
        self._executable_path: str = "./cloudflared"
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

        # Download and set permissions for the executable
        self._setup_executable()

        # Start the tunnel process and capture the URL
        self.url: str = self._start_tunnel()

    def _setup_executable(self):
        """Downloads cloudflared and makes it executable if it doesn't exist."""
        if not os.path.exists(self._executable_path):
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(self._executable_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Make the file executable (add execute permission for the owner)
            current_permissions = os.stat(self._executable_path).st_mode
            os.chmod(self._executable_path, current_permissions | stat.S_IEXEC)

    def _start_tunnel(self) -> str:
        """Starts the tunnel and returns the public URL."""
        command = [self._executable_path, "tunnel", "--url", f"http://localhost:{self._port}", "--no-autoupdate"]

        # Using DEVNULL for stderr to keep the output clean, stdout is piped
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )

        for line in iter(self._process.stdout.readline, ""):
            if ".trycloudflare.com" in line:
                # The line format is typically: "INFO |  https://<subdomain>.trycloudflare.com |"
                try:
                    url = line.split("|")[1].strip()
                    print(f"Tunnel is live at: {url}")
                    return url
                except IndexError:
                    continue

        # If the loop finishes without finding a URL
        self.stop()
        raise RuntimeError("Failed to start cloudflared tunnel or find URL.")

    def stop(self):
        """Stops the cloudflared tunnel process."""
        if self._process and self._process.poll() is None:
            print("Stopping cloudflared tunnel...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
                print("Tunnel stopped successfully.")
            except subprocess.TimeoutExpired:
                print("Tunnel did not terminate gracefully, forcing kill.")
                self._process.kill()
        self._process = None

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, ensuring the tunnel is stopped."""
        self.stop()


def start_tunnel(port: int) -> CloudflaredTunnel:
    """
    Initializes and starts a cloudflared tunnel.

    Args:
        port: The local port number to expose.

    Returns:
        A CloudflaredTunnel object that controls the tunnel process.
        This object has a `url` attribute and a `stop()` method.
    """
    return CloudflaredTunnel(port)


def start_server_in_colab() -> str:
    """
    returns the URL of the tunnel and the running context
    :return:
    """
    if len(_colab_instances) == 0:
        comfyui_execution_context.set(ExecutionContext(server=ServerStub(), folder_names_and_paths=FolderNames(is_root=True), custom_nodes=ExportedNodes(), progress_registry=ProgressRegistryStub()))

        async def colab_server_loop():
            init_default_paths(folder_names_and_paths)
            await _start_comfyui()

        _loop = asyncio.get_running_loop()
        task = _loop.create_task(colab_server_loop())

        tunnel = start_tunnel(8188)
        _colab_instances.append(_ColabTuple(tunnel, task))
    return _colab_instances[0].tunnel.url

import asyncio
import os
import stat
import subprocess
import threading
from asyncio import Task
from typing import NamedTuple, Optional

import requests


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
        self._setup_executable()
        self.url: Optional[str] = None

    def start_tunnel(self):
        if self.url is None:
            self.url: str = self._start_tunnel()

    async def __aenter__(self):
        self.start_tunnel()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __enter__(self):
        self.start_tunnel()
        return self

    def __exit__(self, *args):
        self.stop()

    def _setup_executable(self):
        """Downloads cloudflared and makes it executable if it doesn't exist."""
        if not os.path.exists(self._executable_path):
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(self._executable_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        current_permissions = os.stat(self._executable_path).st_mode
        os.chmod(self._executable_path, current_permissions | stat.S_IEXEC)

    def _start_tunnel(self) -> str:
        """Starts the tunnel and returns the public URL."""
        command = [self._executable_path, "tunnel", "--url", f"http://localhost:{self._port}", "--no-autoupdate"]

        self._process = subprocess.Popen(
            command,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in self._process.stdout:
            if ".trycloudflare.com" in line:
                # the line format is typically: "INFO |  https://<subdomain>.trycloudflare.com |"
                try:
                    url = line.split("|")[1].strip()
                    print(f"Tunnel is live at: {url}")
                    return url
                except IndexError:
                    continue

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


def start_tunnel(port: int) -> CloudflaredTunnel:
    """
    Initializes and starts a cloudflared tunnel.

    Args:
        port: The local port number to expose.

    Returns:
        A CloudflaredTunnel object that controls the tunnel process.
        This object has a `url` attribute and a `stop()` method.
    """
    tunnel = CloudflaredTunnel(port)
    tunnel.start_tunnel()
    return tunnel


def start_server_in_colab() -> str:
    """
    returns the URL of the tunnel and the running context
    :return:
    """
    if len(_colab_instances) == 0:
        from ..execution_context import ExecutionContext, ServerStub, comfyui_execution_context
        from ..component_model.folder_path_types import FolderNames
        from ..nodes.package_typing import ExportedNodes
        from ..progress_types import ProgressRegistryStub
        comfyui_execution_context.set(ExecutionContext(server=ServerStub(), folder_names_and_paths=FolderNames(is_root=True), custom_nodes=ExportedNodes(), progress_registry=ProgressRegistryStub()))

        # now we're ready to import
        from ..cmd.folder_paths import init_default_paths, folder_names_and_paths  # pylint: disable=import-error
        # experimental workarounds for colab
        from ..cmd.main import _start_comfyui

        async def colab_server_loop():
            init_default_paths(folder_names_and_paths)
            await _start_comfyui()

        _loop = asyncio.get_running_loop()
        task = _loop.create_task(colab_server_loop())

        tunnel = start_tunnel(8188)
        _colab_instances.append(_ColabTuple(tunnel, task))
    return _colab_instances[0].tunnel.url

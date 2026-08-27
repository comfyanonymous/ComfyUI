"""Boot ComfyUI with --disable-assets and verify the assets system is inert."""
import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    """Override the package-level autouse cleaner: it boots the regular
    (assets-enabled) server, which this module does not need."""
    yield


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_assets_disabled(base: str, proc: subprocess.Popen, timeout: float = 90.0) -> None:
    start = time.time()
    last_err = None
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early with code {proc.returncode}")
        try:
            r = requests.get(base + "/api/assets", timeout=5)
            if r.status_code == 503:
                return
            last_err = RuntimeError(f"unexpected status {r.status_code}")
        except Exception as e:
            last_err = e
        time.sleep(0.25)
    raise RuntimeError(f"ComfyUI HTTP did not become ready: {last_err}")


@pytest.fixture(scope="module")
def disabled_comfy(tmp_path_factory: pytest.TempPathFactory):
    """Boot a ComfyUI subprocess with --disable-assets.

    Yields (base_url, db_path) where db_path is where the sqlite database
    would be created if the assets system initialized it.
    """
    base_dir = tmp_path_factory.mktemp("comfyui-assets-disabled")
    for sub in ("models", "custom_nodes", "input", "output", "temp", "user"):
        (base_dir / sub).mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "assets-disabled.sqlite3"

    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    out_log = open(logs_dir / "stdout.log", "w", buffering=1)
    err_log = open(logs_dir / "stderr.log", "w", buffering=1)

    comfy_root = Path(__file__).resolve().parent.parent.parent
    port = _free_port()
    try:
        proc = subprocess.Popen(
            args=[
                sys.executable,
                "main.py",
                f"--base-directory={str(base_dir)}",
                f"--database-url=sqlite:///{db_path}",
                "--disable-assets",
                "--listen",
                "127.0.0.1",
                "--port",
                str(port),
                "--cpu",
            ],
            stdout=out_log,
            stderr=err_log,
            cwd=str(comfy_root),
        )
    except Exception:
        out_log.close()
        err_log.close()
        raise

    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_assets_disabled(base_url, proc)
        yield base_url, db_path
    finally:
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=15)
        out_log.close()
        err_log.close()


def test_assets_routes_return_503(disabled_comfy):
    base_url, _ = disabled_comfy
    for path in ("/api/assets", "/api/tags", "/api/assets/seed/status"):
        r = requests.get(base_url + path, timeout=30)
        assert r.status_code == 503, (path, r.text)
        assert r.json()["error"]["code"] == "SERVICE_DISABLED"


def test_database_not_created(disabled_comfy):
    _, db_path = disabled_comfy
    assert not db_path.exists()


def test_seed_scan_rejected(disabled_comfy):
    base_url, _ = disabled_comfy
    r = requests.post(base_url + "/api/assets/seed", json={"roots": ["models"]}, timeout=30)
    assert r.status_code == 503


def test_view_blake3_hash_returns_404(disabled_comfy):
    base_url, _ = disabled_comfy
    r = requests.get(base_url + "/view", params={"filename": "blake3:" + "0" * 64}, timeout=30)
    assert r.status_code == 404


def test_upload_image_skips_asset_registration(disabled_comfy):
    base_url, db_path = disabled_comfy
    files = {"image": ("disabled-mode-test.png", b"\x89PNG fake bytes", "image/png")}
    r = requests.post(base_url + "/upload/image", files=files, timeout=30)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "disabled-mode-test.png"
    assert "asset" not in body
    assert not db_path.exists()

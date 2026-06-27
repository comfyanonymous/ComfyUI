#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_WORKFLOW = REPO_ROOT / "comfy" / "micro" / "demo_workflow.json"
DEMO_INPUT_NAME = "micro_demo_input.png"


class ServerProcess:
    def __init__(self, name: str, port: int, input_dir: Path, output_dir: Path, temp_dir: Path, db_path: Path, log_path: Path, micro_worker_url: str | None = None):
        self.name = name
        self.port = port
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.db_path = db_path
        self.log_path = log_path
        self.micro_worker_url = micro_worker_url
        self._log_file = None
        self.proc = None

    def start(self):
        cmd = [
            sys.executable,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--input-directory",
            str(self.input_dir),
            "--output-directory",
            str(self.output_dir),
            "--temp-directory",
            str(self.temp_dir),
            "--database-url",
            f"sqlite:///{self.db_path}",
            "--disable-all-custom-nodes",
            "--disable-api-nodes",
            "--disable-auto-launch",
            "--disable-metadata",
            "--cpu",
        ]
        if self.micro_worker_url is not None:
            cmd.extend(["--micro-worker-url", self.micro_worker_url])

        self._log_file = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_for_ready(self.port, self.proc, self.log_path)
        return self

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self._log_file is not None:
            self._log_file.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_ready(port: int, proc: subprocess.Popen, log_path: Path, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/system_stats"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server on port {port} exited early\n{tail_log(log_path)}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"server on port {port} did not become ready\n{tail_log(log_path)}")


def tail_log(log_path: Path, lines: int = 80) -> str:
    if not log_path.exists():
        return ""
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def write_demo_input(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (17, 13))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = ((x * 17 + y * 3) % 256, (x * 5 + y * 19) % 256, (x * 11 + y * 7) % 256)
    path = input_dir / DEMO_INPUT_NAME
    image.save(path)
    return path


def load_demo_workflow(prefix: str) -> dict:
    workflow = json.loads(DEMO_WORKFLOW.read_text(encoding="utf-8"))
    workflow["5"]["inputs"]["filename_prefix"] = prefix
    return workflow


def build_reference_workflow(prefix: str) -> dict:
    demo = load_demo_workflow(prefix)
    scale_inputs = dict(demo["3"]["inputs"])
    scale_inputs["image"] = ["1", 0]
    return {
        "1": demo["1"],
        "2": {
            "class_type": "ImageScale",
            "inputs": scale_inputs,
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["2", 0],
                "filename_prefix": prefix,
            },
        },
    }


def post_json(port: int, path: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {path} failed with HTTP {exc.code}: {detail}") from exc


def get_json(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def run_prompt(port: int, workflow: dict, output_dir: Path, save_node_id: str, timeout: float = 120.0) -> bytes:
    queued = post_json(port, "/prompt", {"prompt": workflow})
    if "prompt_id" not in queued:
        raise RuntimeError(f"prompt did not queue: {queued}")

    prompt_id = queued["prompt_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = get_json(port, f"/history/{prompt_id}")
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("status_str") != "success":
                raise RuntimeError(f"prompt {prompt_id} failed: {status}")
            outputs = record.get("outputs", {})
            node_output = outputs.get(save_node_id, {})
            images = node_output.get("images", [])
            if not images:
                raise RuntimeError(f"prompt {prompt_id} produced no saved images: {record}")
            return read_output_image(output_dir, images[0])
        time.sleep(0.25)

    raise RuntimeError(f"prompt {prompt_id} did not finish within {timeout} seconds")


def read_output_image(output_dir: Path, image_info: dict) -> bytes:
    if image_info.get("type") != "output":
        raise RuntimeError(f"expected output image, got {image_info}")
    subfolder = image_info.get("subfolder", "")
    filename = image_info["filename"]
    path = output_dir / subfolder / filename
    return path.read_bytes()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_server(name: str, root: Path, port: int, micro_worker_url: str | None = None) -> ServerProcess:
    return ServerProcess(
        name=name,
        port=port,
        input_dir=root / name / "input",
        output_dir=root / name / "output",
        temp_dir=root / name / "tempbase",
        db_path=root / name / "comfyui.db",
        log_path=root / f"{name}.log",
        micro_worker_url=micro_worker_url,
    )


def run_single(root: Path) -> tuple[str, bytes]:
    port = find_free_port()
    server = make_server("single", root, port)
    write_demo_input(server.input_dir)

    with server:
        micro_bytes = run_prompt(port, load_demo_workflow("micro_single"), server.output_dir, "5")
        reference_bytes = run_prompt(port, build_reference_workflow("micro_reference"), server.output_dir, "3")

    if micro_bytes != reference_bytes:
        raise RuntimeError(f"single-instance output mismatch: micro={sha256(micro_bytes)} reference={sha256(reference_bytes)}")

    digest = sha256(micro_bytes)
    print(f"single-instance sha256 {digest}")  # noqa: T201
    return digest, reference_bytes


def run_two_instance(root: Path, expected_reference: bytes | None = None) -> tuple[str, bytes]:
    worker_port = find_free_port()
    host_port = find_free_port()
    worker = make_server("worker", root, worker_port)
    host = make_server("host", root, host_port, micro_worker_url=f"http://127.0.0.1:{worker_port}/micro/execute")
    write_demo_input(host.input_dir)

    with contextlib.ExitStack() as stack:
        stack.enter_context(worker)
        stack.enter_context(host)
        micro_bytes = run_prompt(host_port, load_demo_workflow("micro_two_instance"), host.output_dir, "5")
        if expected_reference is None:
            expected_reference = run_prompt(host_port, build_reference_workflow("micro_reference_two"), host.output_dir, "3")

    if micro_bytes != expected_reference:
        raise RuntimeError(f"two-instance output mismatch: micro={sha256(micro_bytes)} reference={sha256(expected_reference)}")

    digest = sha256(micro_bytes)
    print(f"two-instance sha256 {digest}")  # noqa: T201
    return digest, micro_bytes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the Micro Substrate demo workflows end-to-end.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--single-only", action="store_true", help="Run only the single-instance loopback verification.")
    mode.add_argument("--two-only", action="store_true", help="Run only the two-instance verification.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory for temporary server input/output/logs.")
    args = parser.parse_args(argv)

    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="comfy-micro-demo-") as tmp:
            return _main(args, Path(tmp))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    return _main(args, args.work_dir)


def _main(args, root: Path) -> int:
    if args.single_only:
        run_single(root)
        return 0
    if args.two_only:
        run_two_instance(root)
        return 0

    digest, reference = run_single(root)
    two_digest, _ = run_two_instance(root, reference)
    if digest != two_digest:
        raise RuntimeError(f"single and two-instance SHA-256 differ: {digest} != {two_digest}")
    print(f"micro substrate demo verified sha256 {digest}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

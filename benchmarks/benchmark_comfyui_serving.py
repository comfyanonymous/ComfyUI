#!/usr/bin/env python3
"""
ComfyUI model serving benchmark.

Submits prompts concurrently to a running ComfyUI server and reports
latency/throughput metrics. Input images and prompt files are prepared
automatically (and cached for reuse) before the benchmark starts.

On first run the script will:
  1. Download model weights (if --download-models is set).
  2. Download the VBench I2V image dataset (requires: pip install gdown),
     or generate synthetic placeholder images as a fallback.
  3. Write one prompt JSON per input image under benchmarks/prompts/<model>_<task>/.

On subsequent runs all three steps are skipped if the files already exist.
Requests are distributed across prompt files in round-robin order.

Supported models / tasks
------------------------
  wan22 / i2v   — Wan 2.2 Image-to-Video (LightX2V 4-step, 720×720, 81 frames)

Usage
-----
  python3 benchmarks/benchmark_comfyui_serving.py \\
    --model wan22 --task i2v \\
    --num-requests 50 --max-concurrency 4 \\
    --host http://127.0.0.1:8188

  # Also download model weights (run from ComfyUI root):
  python3 benchmarks/benchmark_comfyui_serving.py \\
    --model wan22 --task i2v \\
    --download-models --comfyui-base-dir /path/to/ComfyUI \\
    --num-requests 50 --max-concurrency 4 \\
    --host http://127.0.0.1:8188
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import aiohttp
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark setup helpers
# ──────────────────────────────────────────────────────────────────────────────

# Workflow JSON files live in benchmarks/workflows/<model>_<task>.json.
_WORKFLOWS_DIR = Path(__file__).parent / "workflows"

# Placeholder in workflow JSON files that is replaced with the actual image filename.
_IMAGE_PLACEHOLDER = "__INPUT_IMAGE__"

# Model weight downloads for wan22/i2v.
_WAN22_I2V_MODELS: list[tuple[str, str]] = [
    (
        "models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    ),
    (
        "models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    ),
    (
        "models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
    ),
    (
        "models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
    ),
    (
        "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    (
        "models/vae/wan_2.1_vae.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
    ),
]

# Google Drive file IDs from VBench's vbench2_beta_i2v/download_data.sh
_VBENCH_ORIGIN_ZIP_GDRIVE_ID = "1qhkLCSBkzll0dkKpwlDTwLL0nxdQ4nrY"

# Registry mapping (model, task) → benchmark configuration.
# To add a new model/task: drop a workflow JSON in benchmarks/workflows/ and
# add an entry here.
_MODEL_REGISTRY: dict[tuple[str, str], dict[str, Any]] = {
    ("wan22", "i2v"): {
        "workflow_file": "wan22_i2v.json",
        "model_files": _WAN22_I2V_MODELS,
        "image_source": "vbench_i2v",
    },
}

_VALID_MODELS = sorted({m for m, _ in _MODEL_REGISTRY})
_VALID_TASKS = sorted({t for _, t in _MODEL_REGISTRY})


def _replace_in_graph(obj: Any, placeholder: str, value: str) -> None:
    """Recursively replace every occurrence of *placeholder* with *value* in-place."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v == placeholder:
                obj[k] = value
            else:
                _replace_in_graph(v, placeholder, value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if item == placeholder:
                obj[i] = value
            else:
                _replace_in_graph(item, placeholder, value)


def download_models(base_dir: Path, model: str, task: str) -> None:
    """Download model weights for *model*/*task* into *base_dir* using wget."""
    key = (model, task)
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"No model files registered for {model}/{task}")
    for rel_path, url in _MODEL_REGISTRY[key]["model_files"]:
        dest = base_dir / rel_path
        if dest.exists():
            print(f"[setup] already exists, skipping: {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[setup] downloading {dest.name} ...")
        subprocess.run(["wget", "-O", str(dest), url], check=True)


def _try_download_vbench_i2v(input_dir: Path) -> list[str]:
    """
    Download VBench I2V origin images from Google Drive via gdown (pip install gdown).
    Raises on any failure.
    """
    import gdown  # type: ignore; raises ImportError if not installed

    import zipfile

    zip_path = input_dir / "origin.zip"
    try:
        if not zip_path.exists():
            print("[setup] downloading VBench I2V origin images from Google Drive ...")
            gdown.download(id=_VBENCH_ORIGIN_ZIP_GDRIVE_ID, output=str(zip_path), quiet=False)
        print("[setup] extracting origin.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(input_dir))
        zip_path.unlink()
    except Exception:
        if zip_path.exists():
            zip_path.unlink()
        raise

    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    filenames = sorted(
        p.relative_to(input_dir).as_posix()
        for p in input_dir.rglob("*")
        if p.suffix.lower() in image_exts
    )
    print(f"[setup] prepared {len(filenames)} VBench I2V images in {input_dir}")
    return filenames


def _generate_synthetic_images(input_dir: Path, num_images: int) -> list[str]:
    """Generate synthetic 720×720 white PNG placeholders; returns filenames."""
    try:
        from PIL import Image as PILImage  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Pillow is required for synthetic image generation. "
            "Install it with: pip install Pillow"
        )

    filenames: list[str] = []
    for i in range(num_images):
        fname = f"benchmark_input_{i:04d}.png"
        dest = input_dir / fname
        if not dest.exists():
            PILImage.new("RGB", (720, 720), color=(255, 255, 255)).save(str(dest))
        filenames.append(fname)
    return filenames


def prepare_input_images(
    input_dir: Path,
    num_images: int = 20,
    image_source: str = "vbench_i2v",
) -> list[str]:
    """
    Prepare benchmark input images in *input_dir*.
    For "vbench_i2v", downloads from Google Drive and raises on failure.
    Falls back to synthetic images only when image_source is not "vbench_i2v".
    Returns a list of image paths relative to *input_dir*.
    """
    input_dir.mkdir(parents=True, exist_ok=True)

    if image_source == "vbench_i2v":
        return _try_download_vbench_i2v(input_dir)

    print(f"[setup] generating {num_images} synthetic 720×720 placeholder images ...")
    return _generate_synthetic_images(input_dir, num_images)


def generate_prompt_file(
    output_path: Path,
    workflow_path: Path,
    image_filename: str,
) -> None:
    """
    Write a single ComfyUI prompt JSON to *output_path* from *workflow_path*.

    Replaces every occurrence of the sentinel string "__INPUT_IMAGE__" in the
    workflow graph with *image_filename*.
    """
    graph: dict[str, Any] = json.loads(workflow_path.read_text())
    _replace_in_graph(graph, _IMAGE_PLACEHOLDER, image_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"prompt": graph}, indent=2))


def generate_prompt_files(
    model: str,
    task: str,
    output_dir: Path,
    input_dir: Path,
    num_images: int = 20,
    download_model_weights: bool = False,
    comfyui_base_dir: Path | None = None,
) -> list[Path]:
    """
    Full benchmark setup for a given *model*/*task*:

      1. Optionally download model weights into *comfyui_base_dir*.
      2. Prepare input images in *input_dir* (skipped if images already exist).
      3. Generate one prompt JSON per input image in *output_dir*
         (skipped if prompt files already exist).

    Returns the list of prompt file paths.
    """
    key = (model, task)
    if key not in _MODEL_REGISTRY:
        available = ", ".join(f"{m}/{t}" for m, t in _MODEL_REGISTRY)
        raise ValueError(f"Unknown --model {model!r} --task {task!r}. Available: {available}")

    cfg = _MODEL_REGISTRY[key]

    if download_model_weights:
        if comfyui_base_dir is None:
            raise ValueError("--comfyui-base-dir is required when --download-models is set")
        download_models(comfyui_base_dir, model, task)

    image_filenames = prepare_input_images(
        input_dir,
        num_images=num_images,
        image_source=cfg.get("image_source", "synthetic"),
    )
    if not image_filenames:
        raise RuntimeError(f"No input images available in {input_dir}")

    workflow_path = _WORKFLOWS_DIR / cfg["workflow_file"]
    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for i, image_name in enumerate(image_filenames):
        prompt_path = output_dir / f"{model}_{task}_prompt_{i:04d}.json"
        generate_prompt_file(prompt_path, workflow_path, image_name)
        generated.append(prompt_path)

    print(f"[setup] generated {len(generated)} prompt files in {output_dir}")
    return generated


# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RequestResult:
    request_index: int
    prompt_id: str | None
    ok: bool
    error: str | None
    queued_at: float
    started_at: float
    finished_at: float
    end_to_end_s: float
    execution_ms: float | None
    node_timing_ms: dict[str, dict] | None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    rank = (len(values) - 1) * (pct / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def patch_seed_in_prompt(prompt: dict[str, Any], seed: int, seed_path: str | None) -> dict[str, Any]:
    """
    Patch prompt seed in-place for common sampler nodes.
    seed_path format: "<node_id>.<input_name>".
    """
    if seed_path:
        try:
            node_id, input_name = seed_path.split(".", 1)
            prompt[node_id]["inputs"][input_name] = seed
            return prompt
        except Exception as exc:
            raise ValueError(f"Invalid --seed-path '{seed_path}': {exc}") from exc

    # Best-effort fallback: update any input key named 'seed' or 'noise_seed'
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if "seed" in inputs:
            inputs["seed"] = seed
        if "noise_seed" in inputs:
            inputs["noise_seed"] = seed
    return prompt


def load_prompt_template(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "prompt" in data and isinstance(data["prompt"], dict):
        return data
    if isinstance(data, dict):
        return {"prompt": data}
    raise ValueError("Prompt file must be a JSON object (prompt graph or wrapper with 'prompt').")


async def submit_prompt(
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> str:
    url = f"{base_url}{endpoint}"
    async with session.post(url, json=payload, timeout=timeout_s) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"submit failed [{resp.status}] {text}")
        body = json.loads(text)
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"missing prompt_id in response: {body}")
        return prompt_id


async def wait_for_prompt_done(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt_id: str,
    poll_interval_s: float,
    timeout_s: float,
) -> tuple[float | None, dict | None]:
    """
    Returns (execution_ms, node_timing_ms) from history_item["benchmark"].
    Falls back to (None, None) if unavailable.
    """
    deadline = time.perf_counter() + timeout_s
    history_url = f"{base_url}/history/{prompt_id}"

    while time.perf_counter() < deadline:
        async with session.get(history_url, timeout=timeout_s) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"history failed [{resp.status}] {text}")

            payload = await resp.json()
            if not payload:
                await asyncio.sleep(poll_interval_s)
                continue

            history_item = payload.get(prompt_id)
            if history_item is None:
                await asyncio.sleep(poll_interval_s)
                continue

            status = history_item.get("status", {})
            if status.get("status_str") not in ("success", "error"):
                await asyncio.sleep(poll_interval_s)
                continue

            benchmark = history_item.get("benchmark", {})
            return (
                benchmark.get("execution_ms"),
                benchmark.get("nodes"),
            )

        await asyncio.sleep(poll_interval_s)

    raise TimeoutError(f"timed out waiting for prompt_id={prompt_id}")


def build_arrival_schedule(num_requests: int, request_rate: float, poisson: bool, seed: int) -> list[float]:
    """
    Returns absolute offsets (seconds from benchmark start) for each request.
    """
    if request_rate <= 0:
        return [0.0] * num_requests

    rnd = random.Random(seed)
    offsets: list[float] = []
    t = 0.0
    for _ in range(num_requests):
        if poisson:
            delta = rnd.expovariate(request_rate)
        else:
            delta = 1.0 / request_rate
        t += delta
        offsets.append(t)
    return offsets


async def run_request(
    idx: int,
    start_time: float,
    scheduled_offset_s: float,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    prompt_templates: list[dict[str, Any]],
) -> RequestResult:
    await asyncio.sleep(max(0.0, (start_time + scheduled_offset_s) - time.perf_counter()))
    queued_at = time.perf_counter()

    async with semaphore:
        started_at = time.perf_counter()
        prompt_id = None
        try:
            payload = json.loads(json.dumps(prompt_templates[idx % len(prompt_templates)]))
            payload.setdefault("extra_data", {})
            payload["client_id"] = args.client_id

            seed = args.base_seed + idx
            payload["prompt"] = patch_seed_in_prompt(payload["prompt"], seed, args.seed_path)

            prompt_id = await submit_prompt(
                session=session,
                base_url=args.host,
                endpoint=args.endpoint,
                payload=payload,
                timeout_s=args.request_timeout_s,
            )

            execution_ms, node_timing_ms = await wait_for_prompt_done(
                session=session,
                base_url=args.host,
                prompt_id=prompt_id,
                poll_interval_s=args.poll_interval_s,
                timeout_s=args.request_timeout_s,
            )
            finished_at = time.perf_counter()
            return RequestResult(
                request_index=idx,
                prompt_id=prompt_id,
                ok=True,
                error=None,
                queued_at=queued_at,
                started_at=started_at,
                finished_at=finished_at,
                end_to_end_s=finished_at - queued_at,
                execution_ms=execution_ms,
                node_timing_ms=node_timing_ms,
            )
        except Exception as exc:
            finished_at = time.perf_counter()
            return RequestResult(
                request_index=idx,
                prompt_id=prompt_id,
                ok=False,
                error=repr(exc),
                queued_at=queued_at,
                started_at=started_at,
                finished_at=finished_at,
                end_to_end_s=finished_at - queued_at,
                execution_ms=None,
                node_timing_ms=None,
            )


def print_summary(results: list[RequestResult], wall_s: float) -> None:
    success = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    lat_s = [r.end_to_end_s for r in success]
    exec_ms = [r.execution_ms for r in success if r.execution_ms is not None]

    throughput = (len(success) / wall_s) if wall_s > 0 else 0.0
    print("\n=== ComfyUI Serving Benchmark Summary ===")
    print(f"requests_total:   {len(results)}")
    print(f"requests_success: {len(success)}")
    print(f"requests_failed:  {len(fail)}")
    print(f"wall_time_s:      {wall_s:.3f}")
    print(f"throughput_req_s: {throughput:.3f}")

    if lat_s:
        print(f"latency_p50_s:    {percentile(lat_s, 50):.3f}")
        print(f"latency_p90_s:    {percentile(lat_s, 90):.3f}")
        print(f"latency_p95_s:    {percentile(lat_s, 95):.3f}")
        print(f"latency_p99_s:    {percentile(lat_s, 99):.3f}")
        print(f"latency_mean_s:   {statistics.mean(lat_s):.3f}")
        print(f"latency_max_s:    {max(lat_s):.3f}")

    if exec_ms:
        print(f"execution_mean_ms:  {statistics.mean(exec_ms):.2f}")
        print(f"execution_p95_ms:   {percentile(exec_ms, 95):.2f}")

    # Per-node timing: aggregate execution_ms across all successful results.
    node_totals: dict[str, list[float]] = {}
    for r in success:
        if not r.node_timing_ms:
            continue
        for node_id, info in r.node_timing_ms.items():
            key = f"{info.get('class_type', 'unknown')} ({node_id})"
            node_totals.setdefault(key, []).append(info.get("execution_ms", 0.0))
    if node_totals:
        print("\n--- Per-node execution time (mean ms across successful requests) ---")
        for key, times in sorted(node_totals.items(), key=lambda x: -statistics.mean(x[1])):
            print(f"  {key}: mean={statistics.mean(times):.1f}  p95={percentile(times, 95):.1f}  n={len(times)}")

    if fail:
        print("\nSample failures:")
        for r in fail[:5]:
            print(f"  idx={r.request_index} prompt_id={r.prompt_id} error={r.error}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ComfyUI request serving.")
    p.add_argument("--host", type=str, default="http://127.0.0.1:8188", help="ComfyUI base URL.")
    p.add_argument(
        "--endpoint",
        type=str,
        default="/prompt",
        choices=("/prompt", "/bench/prompt"),
        help="Submission endpoint.",
    )
    p.add_argument(
        "--model",
        choices=_VALID_MODELS,
        required=True,
        help=f"Model to benchmark. Choices: {_VALID_MODELS}.",
    )
    p.add_argument(
        "--task",
        choices=_VALID_TASKS,
        required=True,
        help=f"Task type. Choices: {_VALID_TASKS}.",
    )
    p.add_argument(
        "--prompts-dir",
        type=Path,
        default=None,
        help="Directory where generated prompt JSON files are written (default: benchmarks/prompts/<model>_<task>/).",
    )
    p.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of synthetic images to generate when dataset download is unavailable (default: 20).",
    )
    p.add_argument(
        "--download-models",
        action="store_true",
        help="Download model weights before generating prompts (requires --comfyui-base-dir).",
    )
    p.add_argument(
        "--comfyui-base-dir",
        type=Path,
        default=None,
        help="ComfyUI root directory used as the base for model downloads.",
    )
    p.add_argument("--num-requests", type=int, default=50)
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--request-rate", type=float, default=0.0, help="Requests/sec. 0 = fire immediately.")
    p.add_argument("--poisson", action="store_true", help="Use Poisson inter-arrival when request-rate > 0.")
    p.add_argument("--base-seed", type=int, default=1234)
    p.add_argument(
        "--seed-path",
        type=str,
        default=None,
        help="Optional path to seed field in prompt: <node_id>.<input_name> (e.g. 3.seed).",
    )
    p.add_argument("--client-id", type=str, default=f"bench-{uuid.uuid4().hex[:12]}")
    p.add_argument("--request-timeout-s", type=float, default=600.0)
    p.add_argument("--poll-interval-s", type=float, default=0.2)
    p.add_argument("--output-json", type=Path, default=None, help="Write detailed result JSON.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for schedule generation.")
    return p.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    prompts_dir = args.prompts_dir or Path("benchmarks/prompts") / f"{args.model}_{args.task}"
    prompt_paths = generate_prompt_files(
        model=args.model,
        task=args.task,
        output_dir=prompts_dir,
        input_dir=Path("input"),
        num_images=args.num_images,
        download_model_weights=args.download_models,
        comfyui_base_dir=args.comfyui_base_dir,
    )
    prompt_templates = [load_prompt_template(p) for p in prompt_paths]
    print(f"[bench] loaded {len(prompt_templates)} prompt templates, round-robining over {args.num_requests} requests")

    schedule = build_arrival_schedule(
        num_requests=args.num_requests,
        request_rate=args.request_rate,
        poisson=args.poisson,
        seed=args.seed,
    )
    semaphore = asyncio.Semaphore(args.max_concurrency)
    connector = aiohttp.TCPConnector(limit=max(args.max_concurrency * 2, 32))

    started = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                run_request(
                    idx=i,
                    start_time=started,
                    scheduled_offset_s=schedule[i],
                    semaphore=semaphore,
                    session=session,
                    args=args,
                    prompt_templates=prompt_templates,
                )
            )
            for i in range(args.num_requests)
        ]
        results = []
        with tqdm(total=args.num_requests, unit="req", desc="benchmark") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
                if result.ok:
                    pbar.set_postfix(succeeded=sum(r.ok for r in results))
    wall_s = time.perf_counter() - started

    print_summary(results, wall_s)

    if args.output_json is not None:
        out = {
            "config": vars(args),
            "wall_time_s": wall_s,
            "results": [asdict(r) for r in sorted(results, key=lambda x: x.request_index)],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2))
        print(f"\nWrote results to: {args.output_json}")


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

#!/Users/johnsie/.hermes/hermes-agent/venv/bin/python
"""Controlled four-checkpoint comparison for local ComfyUI.

Runs the same prompt, seed, dimensions, sampler, steps, CFG, and negative prompt
against SDXL Base, Illustrious XL v2, Juggernaut XL v9, and Pony Diffusion V6.
Outputs a per-run manifest and a labeled 2×2 contact sheet.

Usage:
  /Users/johnsie/ComfyUI/scripts/compare-four-checkpoints.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

COMFY_URL = "http://127.0.0.1:8188"
COMFY_OUTPUT = Path("/Users/johnsie/ComfyUI/output")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = COMFY_OUTPUT / "model-compare" / RUN_ID

# Controlled benchmark: only the checkpoint changes.
PROMPT = (
    "editorial lifestyle portrait of a young Taiwanese university student, "
    "front-facing and smiling naturally, holding a blank takeaway coffee cup, "
    "standing in a sunlit modern campus courtyard in Taipei, soft warm daylight, "
    "natural skin texture, light beige casual jacket, green tropical plants and "
    "glass architecture in the background, shallow depth of field, detailed, "
    "no text, no letters, no logos"
)
NEGATIVE_PROMPT = (
    "low quality, blurry, out of focus, distorted face, deformed body, "
    "extra fingers, extra limbs, bad anatomy, text, letters, watermark, signature, logo"
)

# Keep every variable below identical to make the comparison meaningful.
SETTINGS = {
    "seed": 20260801,
    "width": 512,
    "height": 512,
    "steps": 20,
    "cfg": 7.0,
    "sampler_name": "euler",
    "scheduler": "normal",
    "denoise": 1.0,
}

MODELS = [
    ("01_sdxl_base", "SDXL Base 1.0", "sdxl_base_1.0.safetensors"),
    ("02_illustrious_xl_v2", "Illustrious XL v2", "illustrious-xl-v2.safetensors"),
    ("03_juggernaut_xl_v9", "Juggernaut XL v9", "juggernautXL_v9.safetensors"),
    ("04_pony_diffusion_v6", "Pony Diffusion V6", "ponyDiffusionV6XL_v6Start.safetensors"),
]


def workflow(checkpoint: str, prefix: str) -> dict:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": PROMPT, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE_PROMPT, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": SETTINGS["width"],
                "height": SETTINGS["height"],
                "batch_size": 1,
            },
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": SETTINGS["seed"],
                "steps": SETTINGS["steps"],
                "cfg": SETTINGS["cfg"],
                "sampler_name": SETTINGS["sampler_name"],
                "scheduler": SETTINGS["scheduler"],
                "denoise": SETTINGS["denoise"],
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": prefix},
        },
    }


def healthcheck() -> None:
    response = requests.get(f"{COMFY_URL}/system_stats", timeout=15)
    response.raise_for_status()
    payload = response.json()
    devices = payload.get("devices", [])
    device = devices[0].get("name", "unknown") if devices else "unknown"
    print(f"ComfyUI ready: device={device}")


def submit(wf: dict) -> str:
    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": wf},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("node_errors") or payload.get("error"):
        raise RuntimeError(json.dumps(payload, ensure_ascii=False, indent=2))
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id returned: {payload}")
    return prompt_id


def wait_for_image(prompt_id: str, pattern: str, timeout_s: int = 1800) -> Path:
    """Wait for SaveImage output; also surface ComfyUI execution errors."""
    deadline = time.monotonic() + timeout_s
    last_progress = 0
    while time.monotonic() < deadline:
        candidates = sorted(RUN_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]

        try:
            history = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=15).json()
            entry = history.get(prompt_id, {}) if isinstance(history, dict) else {}
            status = entry.get("status", {}) if isinstance(entry, dict) else {}
            if status.get("status_str") == "error":
                messages = status.get("messages", [])
                raise RuntimeError(f"ComfyUI execution failed: {messages}")
        except requests.RequestException:
            # A transient local connection failure should not throw away a long render.
            pass

        elapsed = int(timeout_s - max(0, deadline - time.monotonic()))
        if elapsed - last_progress >= 30:
            print(f"  waiting… {elapsed}s elapsed")
            last_progress = elapsed
        time.sleep(3)

    raise TimeoutError(f"No output file after {timeout_s}s for prompt {prompt_id}")


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#15171b")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_contact_sheet(entries: list[dict]) -> Path | None:
    valid = [entry for entry in entries if entry.get("image")]
    if not valid:
        return None

    cell_w, cell_h, label_h, padding = 512, 512, 56, 20
    sheet = Image.new(
        "RGB",
        (padding + 2 * (cell_w + padding), padding + 2 * (cell_h + label_h + padding)),
        "#0d0f12",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, entry in enumerate(valid):
        col, row = index % 2, index // 2
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + label_h + padding)
        image = Image.open(entry["image"])
        sheet.paste(fit(image, (cell_w, cell_h)), (x, y))
        draw.rectangle((x, y + cell_h, x + cell_w, y + cell_h + label_h), fill="#20242b")
        draw.text((x + 12, y + cell_h + 12), entry["label"], fill="white", font=font)
        draw.text((x + 12, y + cell_h + 30), f"{entry['seconds']:.1f}s", fill="#aab4c3", font=font)

    target = RUN_DIR / "comparison-contact-sheet.png"
    sheet.save(target, "PNG")
    return target


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    healthcheck()
    print(f"Run folder: {RUN_DIR}")
    print(f"Benchmark: {SETTINGS['width']}×{SETTINGS['height']}, {SETTINGS['steps']} steps, seed={SETTINGS['seed']}")

    entries: list[dict] = []
    for slug, label, checkpoint in MODELS:
        prefix = f"model-compare/{RUN_ID}/{slug}"
        print(f"\n[{label}] loading {checkpoint}")
        started = time.monotonic()
        entry: dict[str, object] = {"label": label, "checkpoint": checkpoint}
        try:
            prompt_id = submit(workflow(checkpoint, prefix))
            print(f"  queued: {prompt_id}")
            image = wait_for_image(prompt_id, f"{slug}_*.png")
            entry.update({
                "prompt_id": prompt_id,
                "image": str(image),
                "seconds": round(time.monotonic() - started, 1),
                "status": "success",
            })
            print(f"  done: {image.name} in {entry['seconds']:.1f}s")
        except Exception as exc:
            entry.update({
                "status": "error",
                "error": str(exc),
                "seconds": round(time.monotonic() - started, 1),
            })
            print(f"  FAILED: {exc}", file=sys.stderr)
        entries.append(entry)

    manifest = {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "settings": SETTINGS,
        "entries": entries,
    }
    manifest_path = RUN_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    contact_sheet = make_contact_sheet(entries)
    print("\n=== RESULT ===")
    for entry in entries:
        if entry["status"] == "success":
            print(f"OK   {entry['label']}: {entry['image']} ({entry['seconds']:.1f}s)")
        else:
            print(f"FAIL {entry['label']}: {entry['error']}")
    print(f"Manifest: {manifest_path}")
    if contact_sheet:
        print(f"Contact sheet: {contact_sheet}")

    return 0 if all(entry["status"] == "success" for entry in entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())

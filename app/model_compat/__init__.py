"""Model compatibility: will a model run on this machine, judged without downloading it.

Hugging Face's API lists a repo's files with their sizes. A weight file's size,
read beside the GPU's VRAM and the machine's RAM, says whether ComfyUI can hold
it on the GPU, has to offload part of it to RAM, or cannot load it at all.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time

import aiohttp
import psutil
from aiohttp import web

import folder_paths

GIB = 1024 ** 3
# Room the GPU needs beside the weights: CUDA context and activations at ~1 megapixel.
HEADROOM = 1.5 * GIB
# Share of RAM ComfyUI can reasonably fill with offloaded weights.
OFFLOAD_RAM_SHARE = 0.5
MODEL_EXTENSIONS = (".safetensors", ".sft", ".gguf", ".ckpt", ".pt", ".pth", ".bin")
LOCAL_FOLDERS = ["checkpoints", "diffusion_models", "text_encoders", "vae", "loras", "controlnet", "clip_vision", "upscale_models"]
HF_API = "https://huggingface.co/api"
HF_CACHE_SECONDS = 600
ORG_CONCURRENCY = 6
SHARD = re.compile(r"-\d{5}-of-(\d{5})(?=\.)")
# A repo's main model is its files of the first of these tiers it holds; a tier's roles count together.
MAIN_TIERS = (("diffusion", "checkpoint"), ("model",), ("text_encoder",), ("controlnet",), ("lora",))
LEVEL_RANK = {"fits": 0, "offload": 1, "no": 2}

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def hardware() -> dict:
    import torch
    import comfy.model_management as mm

    device = mm.get_torch_device()
    capability = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
    return {
        # "cuda:0 NVIDIA GeForce RTX 5070 Ti : cudaMallocAsync" -> the card's own name
        "name": re.sub(r"^\w+:\d+\s+|\s+:\s+\w+$", "", mm.get_torch_device_name(device)),
        "type": device.type,
        "vram_total": mm.get_total_memory(device),
        "ram_total": psutil.virtual_memory().total,
        "capability": list(capability) if capability else None,
        "supports": {
            "bf16": bool(capability and capability >= (8, 0)),
            "fp8": bool(capability and capability >= (8, 9)),
            "fp4": bool(capability and capability >= (10, 0)),
        },
    }


def precision(name: str) -> str | None:
    lower = name.lower()
    if lower.endswith(".gguf"):
        flat = re.sub(r"[-.]", "_", lower)
        quant = re.search(r"(?:^|_)(i?q\d(?:_k)?(?:_(?:xxs|xs|nl|s|m|l|0|1))?|bf16|f16|f32)(?=_)", flat)
        return f"GGUF {quant.group(1).upper()}" if quant else "GGUF"
    for pattern, label in (
        (r"nvfp4|fp4|mxfp4", "FP4"),
        (r"fp8|e4m3|e5m2", "FP8"),
        (r"int8", "INT8"),
        (r"bf16", "BF16"),
        (r"fp16|f16|half", "FP16"),
        (r"fp32|f32", "FP32"),
    ):
        if re.search(pattern, lower):
            return label
    return None


def role(path: str) -> str:
    lower = path.lower()
    for pattern, label in (
        # The folder a file sits in says it best; the name is the fallback.
        (r"(^|/)loras?/", "lora"),
        (r"(^|/)vae/", "vae"),
        (r"(^|/)(text_encoders?(_\d)?|clip)/", "text_encoder"),
        (r"(^|/)(diffusion_models|unet|transformer)/", "diffusion"),
        (r"(^|/)controlnet/", "controlnet"),
        (r"(^|/)upscale_models/", "upscale"),
        (r"(^|/)checkpoints/", "checkpoint"),
        (r"(^|/)embeddings/", "embedding"),
        (r"lora", "lora"),
        (r"vae|(^|/)ae\.", "vae"),
        (r"text_encoder|(^|/)clip|t5xxl|umt5", "text_encoder"),
        (r"controlnet", "controlnet"),
        (r"upscale|esrgan", "upscale"),
    ):
        if re.search(pattern, lower):
            return label
    return "model"


def judge(size: int, label: str | None, hw: dict) -> dict:
    """fits: whole on the GPU; offload: runs with part of it in RAM, slower; no: too big."""
    vram, ram = hw["vram_total"], hw["ram_total"]
    if hw["type"] != "cuda":
        level = "offload" if size <= ram * OFFLOAD_RAM_SHARE else "no"
    elif size + HEADROOM <= vram:
        level = "fits"
    elif size <= vram - HEADROOM + ram * OFFLOAD_RAM_SHARE:
        level = "offload"
    else:
        level = "no"

    notes = []
    supports = hw["supports"]
    if label == "FP8" and not supports["fp8"]:
        notes.append("This GPU has no native fp8; ComfyUI upcasts it, with no speedup.")
    if label == "FP4" and not supports["fp4"]:
        notes.append("Fast FP4 needs an RTX 50 series (Blackwell) GPU; it runs slowly here.")
    if label == "BF16" and not supports["bf16"]:
        notes.append("This GPU has no bf16; the weights are cast to fp16/fp32.")
    if level == "offload":
        notes.append("Part of the weights stays in RAM: it runs, but slower.")
    return {"level": level, "notes": notes}


def describe(path: str, size: int, hw: dict) -> dict:
    label = precision(os.path.basename(path))
    return {"path": path, "name": os.path.basename(path), "size": size, "precision": label, "role": role(path),
            **judge(size, label, hw)}


def mark_recommended(files: list[dict]) -> None:
    # Per role, the largest file that still fits whole on the GPU: the best quality this machine runs at full speed.
    best = {}
    for f in files:
        if f["level"] == "fits" and f["role"] not in ("lora", "vae", "embedding") and f["size"] > best.get(f["role"], {}).get("size", -1):
            best[f["role"]] = f
    for f in best.values():
        f["recommended"] = True


def summarize_repo(repo: str, files: list[tuple[str, int]], hw: dict) -> dict:
    """One line per repo: can its main model run here, and with which file."""
    described = [describe(path, size, hw) for path, size in files]
    if not described:
        return {"repo": repo, "level": None, "files": 0}
    tier = next((t for t in MAIN_TIERS if any(f["role"] in t for f in described)), ())
    main = [f for f in described if f["role"] in tier] or described
    # The best level the main model reaches; at "fits" its largest file (best quality at full speed),
    # otherwise its smallest (the closest to fitting).
    best = min(main, key=lambda f: (LEVEL_RANK[f["level"]], -f["size"] if f["level"] == "fits" else f["size"]))
    counts = {level: sum(f["level"] == level for f in described) for level in LEVEL_RANK}
    return {"repo": repo, "level": best["level"], "files": len(described), "counts": counts,
            "best": {k: best[k] for k in ("name", "size", "precision", "role")}}


def parse_repo(text: str) -> str | None:
    text = text.strip()
    match = re.match(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/(?:models/)?([^/\s]+/[^/\s?#]+)", text)
    repo = match.group(1) if match else text
    return repo if re.fullmatch(r"[\w.-]+/[\w.-]+", repo) else None


def group_shards(files: list[tuple[str, int]]) -> list[tuple[str, int]]:
    # Sharded weights (model-00001-of-00004.safetensors) load as one model: sum them.
    grouped: dict[str, tuple[str, int]] = {}
    for path, size in files:
        shard = SHARD.search(path)
        key = SHARD.sub("", path) if shard else path
        label = f"{key} ({int(shard.group(1))} shards)" if shard else path
        grouped[key] = (label, grouped.get(key, (label, 0))[1] + size)
    return list(grouped.values())


class RepoNotFound(Exception):
    pass


class ModelCompatManager:
    WEB_DIR = WEB_DIR

    def __init__(self) -> None:
        self.hf_cache: dict[str, tuple[float, list[tuple[str, int]]]] = {}
        self.org_cache: dict[str, tuple[float, list[tuple[str, list[tuple[str, int]] | None]]]] = {}

    async def hf_files(self, session: aiohttp.ClientSession, repo: str) -> list[tuple[str, int]]:
        cached = self.hf_cache.get(repo)
        if cached and time.time() - cached[0] < HF_CACHE_SECONDS:
            return cached[1]
        url = f"{HF_API}/models/{repo}/tree/main?recursive=true"
        entries = []
        for _ in range(10):  # the tree pages by 1000 entries
            async with session.get(url) as response:
                # Hugging Face answers 401 for a repo that does not exist as for a private one.
                if response.status in (401, 403, 404):
                    raise RepoNotFound(repo)
                response.raise_for_status()
                entries += await response.json()
                url = response.links.get("next", {}).get("url")
            if not url:
                break
        files = group_shards([(e["path"], e.get("lfs", {}).get("size", e.get("size", 0))) for e in entries
                              if e.get("type") == "file" and e["path"].lower().endswith(MODEL_EXTENSIONS)])
        self.hf_cache[repo] = (time.time(), files)
        return files

    def add_routes(self, routes) -> None:
        @routes.get("/model_compat/hardware")
        async def get_hardware(request):
            return web.json_response(hardware())

        @routes.get("/model_compat/hf")
        async def get_hf_repo(request):
            repo = parse_repo(request.query.get("repo", ""))
            if not repo:
                return web.json_response({"error": "Enter a Hugging Face repo id or link, e.g. Comfy-Org/z_image_turbo."}, status=400)
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                    files = await self.hf_files(session, repo)
            except RepoNotFound:
                return web.json_response({"error": f"Repo {repo} was not found or is private."}, status=404)
            except (aiohttp.ClientError, TimeoutError) as error:
                return web.json_response({"error": f"Could not reach Hugging Face: {error}"}, status=502)
            hw = hardware()
            described = sorted((describe(path, size, hw) for path, size in files), key=lambda f: (f["role"], -f["size"]))
            mark_recommended(described)
            return web.json_response({"repo": repo, "hardware": hw, "files": described})

        @routes.get("/model_compat/local")
        async def get_local_models(request):
            hw = hardware()
            files = []
            for folder in LOCAL_FOLDERS:
                for name in folder_paths.get_filename_list(folder):
                    full = folder_paths.get_full_path(folder, name)
                    if full and os.path.isfile(full):
                        files.append({**describe(f"{folder}/{name}", os.path.getsize(full), hw), "folder": folder})
            return web.json_response({"hardware": hw, "files": sorted(files, key=lambda f: (f["folder"], -f["size"]))})

        @routes.get("/model_compat/org")
        async def get_org(request):
            org = request.query.get("name", "Comfy-Org")
            if not re.fullmatch(r"[\w.-]+", org):
                return web.json_response({"error": "Invalid organization name."}, status=400)
            cached = self.org_cache.get(org)
            if not cached or time.time() - cached[0] >= HF_CACHE_SECONDS:
                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                        async with session.get(f"{HF_API}/models?author={org}&limit=1000") as response:
                            response.raise_for_status()
                            repos = sorted((m["id"] for m in await response.json()), key=str.lower)
                        gate = asyncio.Semaphore(ORG_CONCURRENCY)

                        async def files_of(repo):
                            async with gate:
                                try:
                                    return repo, await self.hf_files(session, repo)
                                except (RepoNotFound, aiohttp.ClientError, TimeoutError) as error:
                                    logging.warning("model_compat: could not read %s: %s", repo, error)
                                    return repo, None

                        cached = (time.time(), await asyncio.gather(*(files_of(repo) for repo in repos)))
                except (aiohttp.ClientError, TimeoutError) as error:
                    return web.json_response({"error": f"Could not reach Hugging Face: {error}"}, status=502)
                self.org_cache[org] = cached
            hw = hardware()
            rows = [summarize_repo(repo, files, hw) if files is not None else {"repo": repo, "level": None, "files": 0, "error": True}
                    for repo, files in cached[1]]
            return web.json_response({"org": org, "hardware": hw, "repos": rows})

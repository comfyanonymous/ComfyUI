#!/usr/bin/env python3
"""
Studio server — film/TV production tool built on ComfyUI.
"""

import json
import os
import random
import shutil
import time
import uuid
import mimetypes
import urllib.request
import urllib.error
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ── Paths ──────────────────────────────────────────────────────────────────────

STUDIO_DIR  = Path(__file__).parent
APP_DIR     = STUDIO_DIR / "app"
DATA_DIR    = STUDIO_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
WORKFLOWS_DIR = STUDIO_DIR / "workflows"

COMFYUI_URL  = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
STUDIO_PORT  = int(os.environ.get("STUDIO_PORT", "8189"))

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

# ── Resolution presets ─────────────────────────────────────────────────────────
# All dimensions are multiples of 8 for diffusion model compatibility.

RESOLUTIONS = {
    "2.39:1": {
        "1080p": (1920, 800),
        "2K":    (2560, 1072),
        "4K":    (3840, 1608),
    },
    "1.85:1": {
        "1080p": (2000, 1080),
        "4K":    (3840, 2072),
    },
    "16:9": {
        "720p":  (1280, 720),
        "1080p": (1920, 1080),
        "2K":    (2560, 1440),
        "4K":    (3840, 2160),
    },
    "4:3": {
        "1080p": (1440, 1080),
        "4K":    (2880, 2160),
    },
    "9:16": {
        "720p":  (720, 1280),
        "1080p": (1080, 1920),
    },
    "1:1": {
        "1024px": (1024, 1024),
        "2048px": (2048, 2048),
    },
}

# ── Shot / camera prompt additions ────────────────────────────────────────────

SHOT_PROMPTS = {
    "wide":             "wide shot, establishing shot",
    "medium":           "medium shot",
    "close_up":         "close-up shot",
    "extreme_close_up": "extreme close-up",
    "over_shoulder":    "over-the-shoulder shot",
    "pov":              "point of view shot, first person perspective",
    "aerial":           "aerial shot, drone shot, bird's eye view",
    "two_shot":         "two-shot",
}

CAMERA_PROMPTS = {
    "static":    "",
    "push_in":   "slow push-in, camera moving forward",
    "pull_out":  "pull-out, zoom out",
    "pan":       "camera pan",
    "handheld":  "handheld camera, verité style",
    "crane_up":  "crane shot moving upward",
}

# ── Style presets ──────────────────────────────────────────────────────────────

STYLE_PRESETS = {
    "cinematic": {
        "prompt_prefix": "cinematic, 35mm film, anamorphic lens, dramatic lighting, shallow depth of field, professional cinematography",
        "prompt_suffix": "photorealistic, film grain",
        "negative_prompt": "blurry, low quality, watermark, text, deformed, anime, cartoon, amateur",
    },
    "anime": {
        "prompt_prefix": "anime style, cel animation, vibrant colors, clean linework",
        "prompt_suffix": "high quality anime, studio quality",
        "negative_prompt": "photorealistic, photo, 3d render, blurry, watermark, text, deformed",
    },
    "noir": {
        "prompt_prefix": "film noir, high contrast black and white, dramatic shadows, 1940s atmosphere, chiaroscuro",
        "prompt_suffix": "moody, atmospheric, brooding",
        "negative_prompt": "color, bright, cheerful, blurry, low quality, watermark",
    },
    "illustration": {
        "prompt_prefix": "digital illustration, concept art, detailed painting",
        "prompt_suffix": "high quality illustration, artstation trending",
        "negative_prompt": "photograph, photo, blurry, low quality, watermark, text",
    },
    "documentary": {
        "prompt_prefix": "documentary photography, natural light, candid, reportage style",
        "prompt_suffix": "authentic, realistic, unposed",
        "negative_prompt": "blurry, low quality, watermark, text, studio lighting, artificial",
    },
}

# ── Project helpers ────────────────────────────────────────────────────────────

def list_projects():
    out = []
    for p in sorted(PROJECTS_DIR.iterdir()):
        if p.is_dir():
            cfg = p / "config.json"
            if cfg.exists():
                data = json.loads(cfg.read_text())
                data["id"] = p.name
                out.append(data)
    return out

def get_project(pid):
    cfg = PROJECTS_DIR / pid / "config.json"
    if not cfg.exists():
        return None
    data = json.loads(cfg.read_text())
    data["id"] = pid
    return data

def save_project(pid, data):
    d = PROJECTS_DIR / pid
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {k: v for k, v in data.items() if k != "id"}, indent=2))

def delete_project(pid):
    d = PROJECTS_DIR / pid
    if d.exists():
        shutil.rmtree(d)

# ── Character helpers ──────────────────────────────────────────────────────────

def _list_assets(project_id, kind):
    """Generic lister for characters and locations."""
    base = PROJECTS_DIR / project_id / kind
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if d.is_dir():
            cfg = d / "config.json"
            if cfg.exists():
                data = json.loads(cfg.read_text())
                data["id"] = d.name
                refs_dir = d / "refs"
                data["refs"] = (
                    [f.name for f in sorted(refs_dir.iterdir())
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
                    if refs_dir.exists() else []
                )
                out.append(data)
    return out

def _get_asset(project_id, kind, asset_id):
    cfg = PROJECTS_DIR / project_id / kind / asset_id / "config.json"
    if not cfg.exists():
        return None
    data = json.loads(cfg.read_text())
    data["id"] = asset_id
    refs_dir = PROJECTS_DIR / project_id / kind / asset_id / "refs"
    data["refs"] = (
        [f.name for f in sorted(refs_dir.iterdir())
         if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        if refs_dir.exists() else []
    )
    return data

def _save_asset(project_id, kind, asset_id, data):
    d = PROJECTS_DIR / project_id / kind / asset_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "refs").mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {k: v for k, v in data.items() if k not in ("id", "refs")}, indent=2))

def _delete_asset(project_id, kind, asset_id):
    d = PROJECTS_DIR / project_id / kind / asset_id
    if d.exists():
        shutil.rmtree(d)

def list_characters(pid):  return _list_assets(pid, "characters")
def get_character(pid, cid): return _get_asset(pid, "characters", cid)
def save_character(pid, cid, data): _save_asset(pid, "characters", cid, data)
def delete_character(pid, cid): _delete_asset(pid, "characters", cid)

def list_locations(pid):  return _list_assets(pid, "locations")
def get_location(pid, lid): return _get_asset(pid, "locations", lid)
def save_location(pid, lid, data): _save_asset(pid, "locations", lid, data)
def delete_location(pid, lid): _delete_asset(pid, "locations", lid)

# ── Scene helpers ──────────────────────────────────────────────────────────────

def list_scenes(pid):
    d = PROJECTS_DIR / pid / "scenes"
    if not d.exists():
        return []
    scenes = []
    for f in d.glob("*.json"):
        data = json.loads(f.read_text())
        data["id"] = f.stem
        scenes.append(data)
    # Sort by sequence, then created_at
    scenes.sort(key=lambda s: (s.get("sequence", 9999), s.get("created_at", "")))
    return scenes

def get_scene(pid, sid):
    f = PROJECTS_DIR / pid / "scenes" / f"{sid}.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    data["id"] = sid
    return data

def save_scene(pid, sid, data):
    d = PROJECTS_DIR / pid / "scenes"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}.json").write_text(json.dumps(
        {k: v for k, v in data.items() if k != "id"}, indent=2))

def delete_scene(pid, sid):
    f = PROJECTS_DIR / pid / "scenes" / f"{sid}.json"
    if f.exists():
        f.unlink()

# ── ComfyUI helpers ────────────────────────────────────────────────────────────

def comfy_get(path):
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}{path}", timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return None

def comfy_post(path, body):
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{COMFYUI_URL}{path}", data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def comfy_status():
    return comfy_get("/system_stats") is not None

def comfy_checkpoints():
    """Return list of checkpoint filenames available in ComfyUI."""
    result = comfy_get("/models/checkpoints")
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("checkpoints", "models", "files"):
            if key in result and isinstance(result[key], list):
                return result[key]
    return []

# ── Workflow assembly ──────────────────────────────────────────────────────────

def load_workflow_template(name):
    path = WORKFLOWS_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None

def assemble_workflow(scene, project, characters, locations):
    output_type  = scene.get("output_type", "still")
    char_ids     = scene.get("characters", [])
    loc_id       = scene.get("location_id")
    scene_chars  = [c for c in characters if c["id"] in char_ids]
    scene_loc    = next((l for l in locations if l["id"] == loc_id), None) if loc_id else None

    has_lora        = any(c.get("type") == "lora" for c in scene_chars)
    has_photomaker  = any(c.get("type") == "photomaker" for c in scene_chars)

    if output_type == "video":
        template_name = "video_base"
    elif has_photomaker:
        template_name = "still_photomaker"
    elif has_lora:
        template_name = "still_lora"
    else:
        template_name = "still_base"

    workflow = load_workflow_template(template_name) or load_workflow_template("still_base")
    if workflow is None:
        return None, None, f"No workflow template found"

    # Seed
    seed_override = scene.get("seed")  # None = random, int = locked
    seed = int(seed_override) if seed_override is not None else random.randint(1, 2**32 - 1)

    # Resolution
    aspect  = scene.get("aspect_ratio", "16:9")
    res_key = scene.get("resolution", "1080p")
    dims    = RESOLUTIONS.get(aspect, {}).get(res_key, (1920, 1080))
    width, height = dims

    # Batch size
    batch_size = max(1, int(scene.get("batch_size", 1)))

    # Checkpoint from project style
    checkpoint = project.get("style", {}).get("checkpoint", "")

    # Prompt
    prompt = build_prompt(scene, project, scene_chars, scene_loc)
    negative = project.get("style", {}).get("negative_prompt",
                           "blurry, low quality, watermark, text, deformed")

    # Inject everything
    workflow = inject_text(workflow, "positive", prompt)
    workflow = inject_text(workflow, "negative", negative)
    workflow = inject_resolution(workflow, width, height, batch_size)
    workflow = inject_seed(workflow, seed)
    if checkpoint:
        workflow = inject_checkpoint(workflow, checkpoint)
    if has_lora:
        for char in scene_chars:
            if char.get("type") == "lora" and char.get("lora_file"):
                workflow = inject_lora(workflow, char["lora_file"],
                                       float(char.get("lora_weight", 0.8)))

    return workflow, seed, None

def build_prompt(scene, project, characters, location):
    parts = []

    style = project.get("style", {})
    if style.get("prompt_prefix"):
        parts.append(style["prompt_prefix"])

    # Shot type
    shot = SHOT_PROMPTS.get(scene.get("shot_type", ""), "")
    if shot:
        parts.append(shot)

    # Camera
    cam = CAMERA_PROMPTS.get(scene.get("camera", ""), "")
    if cam:
        parts.append(cam)

    # PhotoMaker tokens
    for char in characters:
        if char.get("type") == "photomaker":
            parts.append(f"{char['name']} img")

    # Scene description
    if scene.get("description"):
        parts.append(scene["description"])

    # Location
    if location:
        loc_text = location.get("name", "")
        if location.get("description"):
            loc_text += ", " + location["description"]
        if location.get("time_of_day"):
            loc_text += ", " + location["time_of_day"]
        if location.get("weather"):
            loc_text += ", " + location["weather"]
        if loc_text:
            parts.append(loc_text)

    if style.get("prompt_suffix"):
        parts.append(style["prompt_suffix"])

    return ", ".join(p.strip() for p in parts if p.strip())

def inject_text(workflow, role, text):
    for node in workflow.values():
        if node.get("class_type") == "CLIPTextEncode":
            if node["inputs"].get("_role") == role:
                node["inputs"]["text"] = text
                return workflow
    # Fallback: first node without _role tag for positive
    if role == "positive":
        for node in workflow.values():
            if node.get("class_type") == "CLIPTextEncode":
                if "_role" not in node["inputs"]:
                    node["inputs"]["text"] = text
                    return workflow
    return workflow

def inject_resolution(workflow, width, height, batch_size):
    for node in workflow.values():
        if node.get("class_type") in (
            "EmptyLatentImage", "EmptyHunyuanLatentVideo",
            "EmptyLTXVLatentVideo", "EmptyMochiLatentVideo", "EmptyCosmosLatentVideo",
        ):
            node["inputs"]["width"]      = width
            node["inputs"]["height"]     = height
            node["inputs"]["batch_size"] = batch_size
    return workflow

def inject_seed(workflow, seed):
    for node in workflow.values():
        if node.get("class_type") == "KSampler":
            node["inputs"]["seed"] = seed
    return workflow

def inject_checkpoint(workflow, name):
    for node in workflow.values():
        if node.get("class_type") in ("CheckpointLoaderSimple",):
            node["inputs"]["ckpt_name"] = name
    return workflow

def inject_lora(workflow, lora_file, weight):
    for node in workflow.values():
        if node.get("class_type") == "LoraLoader":
            node["inputs"]["lora_name"]       = lora_file
            node["inputs"]["strength_model"]  = weight
            node["inputs"]["strength_clip"]   = weight
            break
    return workflow

# ── Multipart parser ───────────────────────────────────────────────────────────

def parse_multipart(rfile, content_type, content_length):
    import cgi, io
    body = rfile.read(content_length)
    env  = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": str(content_length)}
    form = cgi.FieldStorage(fp=io.BytesIO(body), environ=env)
    fields, files = {}, {}
    for key in form.keys():
        item = form[key]
        if item.filename:
            files[key] = (item.filename, item.file.read())
        else:
            fields[key] = item.value
    return fields, files

# ── HTTP handler ───────────────────────────────────────────────────────────────

class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(fmt, *args)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_err(self, msg, status=400):
        self.send_json({"error": msg}, status)

    def send_file(self, path):
        mime, _ = mimetypes.guess_type(str(path))
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── GET ────────────────────────────────────────────────────────────────────

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        parts = [p for p in path.split("/") if p]

        # Static files
        if not path.startswith("/api/"):
            candidate = APP_DIR / (path.lstrip("/") or "index.html")
            if candidate.exists() and candidate.is_file():
                return self.send_file(candidate)
            index = APP_DIR / "index.html"
            if index.exists():
                return self.send_file(index)

        def seg(*expected):
            return parts == list(expected)

        def starts(*expected):
            return parts[:len(expected)] == list(expected)

        # Status
        if seg("api", "status"):
            return self.send_json({"comfyui": comfy_status(), "studio": True})

        # Resolution presets
        if seg("api", "resolutions"):
            return self.send_json(RESOLUTIONS)

        # Style presets
        if seg("api", "style_presets"):
            return self.send_json(STYLE_PRESETS)

        # ComfyUI checkpoints
        if seg("api", "comfy", "checkpoints"):
            return self.send_json(comfy_checkpoints())

        # ComfyUI output proxy
        if starts("api", "comfy", "outputs") and len(parts) == 4:
            try:
                url = f"{COMFYUI_URL}/view?filename={parts[3]}&type=output"
                with urllib.request.urlopen(url, timeout=10) as r:
                    data = r.read()
                    ct   = r.headers.get("Content-Type", "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_err(str(e), 502)
            return

        # ComfyUI queue poll
        if starts("api", "comfy", "queue") and len(parts) == 4:
            prompt_id = parts[3]
            history   = comfy_get(f"/history/{prompt_id}")
            if history and prompt_id in history:
                entry   = history[prompt_id]
                outputs = entry.get("outputs", {})
                files   = []
                for node_out in outputs.values():
                    files += [i["filename"] for i in node_out.get("images", [])]
                    files += [v["filename"] for v in node_out.get("videos", [])]
                return self.send_json({"status": "done", "files": files})
            queue = comfy_get("/queue") or {}
            running = [item[1] for item in queue.get("queue_running", [])]
            pending = [item[1] for item in queue.get("queue_pending", [])]
            if prompt_id in running:
                return self.send_json({"status": "running"})
            if prompt_id in pending:
                return self.send_json({"status": "pending"})
            return self.send_json({"status": "unknown"})

        # Projects
        if seg("api", "projects"):
            return self.send_json(list_projects())
        if starts("api", "projects") and len(parts) == 3:
            p = get_project(parts[2])
            return self.send_json(p) if p else self.send_err("Not found", 404)

        # Characters + Locations (shared pattern)
        for kind in ("characters", "locations"):
            if starts("api", "projects") and len(parts) == 4 and parts[3] == kind:
                pid = parts[2]
                fn = list_characters if kind == "characters" else list_locations
                return self.send_json(fn(pid))
            if starts("api", "projects") and len(parts) == 5 and parts[3] == kind:
                pid, aid = parts[2], parts[4]
                fn = get_character if kind == "characters" else get_location
                a = fn(pid, aid)
                return self.send_json(a) if a else self.send_err("Not found", 404)
            # Ref image
            if (starts("api", "projects") and len(parts) == 7
                    and parts[3] == kind and parts[5] == "refs"):
                pid, aid, fname = parts[2], parts[4], parts[6]
                ref = PROJECTS_DIR / pid / kind / aid / "refs" / fname
                return self.send_file(ref) if ref.exists() else self.send_err("Not found", 404)

        # Scenes
        if starts("api", "projects") and len(parts) == 4 and parts[3] == "scenes":
            return self.send_json(list_scenes(parts[2]))
        if starts("api", "projects") and len(parts) == 5 and parts[3] == "scenes":
            s = get_scene(parts[2], parts[4])
            return self.send_json(s) if s else self.send_err("Not found", 404)

        self.send_err("Not found", 404)

    # ── POST ───────────────────────────────────────────────────────────────────

    def do_POST(self):
        path  = urlparse(self.path).path.rstrip("/")
        parts = [p for p in path.split("/") if p]
        ct    = self.headers.get("Content-Type", "")

        # Projects
        if parts == ["api", "projects"]:
            body = self.read_json()
            if not body.get("name"):
                return self.send_err("name required")
            pid = body.get("id") or str(uuid.uuid4())[:8]
            data = {"name": body["name"], "description": body.get("description", ""),
                    "style": body.get("style", {}),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            save_project(pid, data)
            data["id"] = pid
            return self.send_json(data, 201)

        # Characters / Locations (shared)
        for kind in ("characters", "locations"):
            if (len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == kind):
                pid = parts[2]
                if not get_project(pid):
                    return self.send_err("Project not found", 404)

                if "multipart/form-data" in ct:
                    length = int(self.headers.get("Content-Length", 0))
                    fields, files = parse_multipart(self.rfile, ct, length)
                else:
                    fields = self.read_json()
                    files  = {}

                name = fields.get("name", "")
                if not name:
                    return self.send_err("name required")
                aid = name.lower().replace(" ", "_")

                if kind == "characters":
                    cfg = {"name": name, "type": fields.get("type", "photomaker"),
                           "lora_file": fields.get("lora_file", ""),
                           "lora_weight": float(fields.get("lora_weight", 0.8)),
                           "notes": fields.get("notes", "")}
                else:
                    cfg = {"name": name,
                           "description": fields.get("description", ""),
                           "time_of_day": fields.get("time_of_day", ""),
                           "weather":     fields.get("weather", ""),
                           "notes":       fields.get("notes", "")}

                fn = save_character if kind == "characters" else save_location
                fn(pid, aid, cfg)

                refs_dir = PROJECTS_DIR / pid / kind / aid / "refs"
                refs_dir.mkdir(parents=True, exist_ok=True)
                for field_name, (filename, fbytes) in files.items():
                    if field_name.startswith("ref"):
                        (refs_dir / Path(filename).name).write_bytes(fbytes)

                cfg["id"]   = aid
                cfg["refs"] = [f.name for f in sorted(refs_dir.iterdir())
                               if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
                return self.send_json(cfg, 201)

            # Upload refs to existing asset
            if (len(parts) == 6 and parts[:2] == ["api", "projects"]
                    and parts[3] == kind and parts[5] == "refs"):
                pid, aid = parts[2], parts[4]
                fn_get = get_character if kind == "characters" else get_location
                if not fn_get(pid, aid):
                    return self.send_err("Not found", 404)
                if "multipart/form-data" not in ct:
                    return self.send_err("Expected multipart/form-data")
                length = int(self.headers.get("Content-Length", 0))
                _, files = parse_multipart(self.rfile, ct, length)
                refs_dir = PROJECTS_DIR / pid / kind / aid / "refs"
                refs_dir.mkdir(parents=True, exist_ok=True)
                saved = []
                for _, (filename, fbytes) in files.items():
                    name = Path(filename).name
                    (refs_dir / name).write_bytes(fbytes)
                    saved.append(name)
                return self.send_json({"saved": saved})

        # Scenes
        if (len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "scenes"):
            pid = parts[2]
            project = get_project(pid)
            if not project:
                return self.send_err("Project not found", 404)
            body = self.read_json()
            if not body.get("description"):
                return self.send_err("description required")

            existing = list_scenes(pid)
            sid = str(uuid.uuid4())[:8]
            scene = {
                "title":         body.get("title", ""),
                "description":   body["description"],
                "characters":    body.get("characters", []),
                "location_id":   body.get("location_id"),
                "output_type":   body.get("output_type", "still"),
                "aspect_ratio":  body.get("aspect_ratio", "16:9"),
                "resolution":    body.get("resolution", "1080p"),
                "shot_type":     body.get("shot_type", ""),
                "camera":        body.get("camera", ""),
                "batch_size":    body.get("batch_size", 1),
                "seed":          body.get("seed"),        # None = random
                "review_status": "draft",
                "sequence":      len(existing) + 1,
                "status":        "pending",
                "outputs":       [],
                "last_seed":     None,
                "created_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            save_scene(pid, sid, scene)

            if body.get("generate", False):
                characters = list_characters(pid)
                locations  = list_locations(pid)
                workflow, seed_used, err = assemble_workflow(scene, project, characters, locations)
                if err:
                    scene["status"] = "error"
                    scene["error"]  = err
                else:
                    client_id = str(uuid.uuid4())
                    result    = comfy_post("/prompt", {"prompt": workflow, "client_id": client_id})
                    prompt_id = result.get("prompt_id")
                    if prompt_id:
                        scene["status"]    = "generating"
                        scene["prompt_id"] = prompt_id
                        scene["last_seed"] = seed_used
                    else:
                        scene["status"] = "error"
                        scene["error"]  = result.get("error", "ComfyUI submission failed")
                save_scene(pid, sid, scene)

            scene["id"] = sid
            return self.send_json(scene, 201)

        # Generate existing scene
        if (len(parts) == 6 and parts[:2] == ["api", "projects"]
                and parts[3] == "scenes" and parts[5] == "generate"):
            pid, sid = parts[2], parts[4]
            project  = get_project(pid)
            scene    = get_scene(pid, sid)
            if not project or not scene:
                return self.send_err("Not found", 404)
            characters = list_characters(pid)
            locations  = list_locations(pid)
            workflow, seed_used, err = assemble_workflow(scene, project, characters, locations)
            if err:
                return self.send_err(err)
            client_id = str(uuid.uuid4())
            result    = comfy_post("/prompt", {"prompt": workflow, "client_id": client_id})
            prompt_id = result.get("prompt_id")
            if prompt_id:
                scene["status"]    = "generating"
                scene["prompt_id"] = prompt_id
                scene["last_seed"] = seed_used
            else:
                scene["status"] = "error"
                scene["error"]  = result.get("error", "Failed")
            save_scene(pid, sid, scene)
            scene["id"] = sid
            return self.send_json(scene)

        self.send_err("Not found", 404)

    # ── PUT ────────────────────────────────────────────────────────────────────

    def do_PUT(self):
        path  = urlparse(self.path).path.rstrip("/")
        parts = [p for p in path.split("/") if p]

        # Project
        if len(parts) == 3 and parts[:2] == ["api", "projects"]:
            pid = parts[2]
            existing = get_project(pid)
            if not existing:
                return self.send_err("Not found", 404)
            body = self.read_json()
            existing.update({k: v for k, v in body.items() if k != "id"})
            save_project(pid, existing)
            return self.send_json(existing)

        # Characters / Locations
        for kind in ("characters", "locations"):
            if (len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == kind):
                pid, aid = parts[2], parts[4]
                fn_get  = get_character if kind == "characters" else get_location
                fn_save = save_character if kind == "characters" else save_location
                existing = fn_get(pid, aid)
                if not existing:
                    return self.send_err("Not found", 404)
                body = self.read_json()
                existing.update({k: v for k, v in body.items() if k not in ("id", "refs")})
                fn_save(pid, aid, existing)
                return self.send_json(existing)

        # Scenes
        if (len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == "scenes"):
            pid, sid = parts[2], parts[4]
            existing = get_scene(pid, sid)
            if not existing:
                return self.send_err("Not found", 404)
            body = self.read_json()
            existing.update({k: v for k, v in body.items() if k != "id"})
            save_scene(pid, sid, existing)
            existing["id"] = sid
            return self.send_json(existing)

        # Scene reorder: PUT /api/projects/:id/scenes/reorder  body: [{id, sequence}]
        if (len(parts) == 5 and parts[:2] == ["api", "projects"]
                and parts[3] == "scenes" and parts[4] == "reorder"):
            pid  = parts[2]
            body = self.read_json()  # list of {id, sequence}
            for item in body:
                s = get_scene(pid, item["id"])
                if s:
                    s["sequence"] = item["sequence"]
                    save_scene(pid, item["id"], s)
            return self.send_json({"ok": True})

        self.send_err("Not found", 404)

    # ── DELETE ─────────────────────────────────────────────────────────────────

    def do_DELETE(self):
        path  = urlparse(self.path).path.rstrip("/")
        parts = [p for p in path.split("/") if p]

        if len(parts) == 3 and parts[:2] == ["api", "projects"]:
            delete_project(parts[2])
            return self.send_json({"ok": True})

        for kind in ("characters", "locations"):
            if (len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == kind):
                fn = delete_character if kind == "characters" else delete_location
                fn(parts[2], parts[4])
                return self.send_json({"ok": True})
            if (len(parts) == 7 and parts[:2] == ["api", "projects"]
                    and parts[3] == kind and parts[5] == "refs"):
                ref = PROJECTS_DIR / parts[2] / kind / parts[4] / "refs" / parts[6]
                if ref.exists():
                    ref.unlink()
                return self.send_json({"ok": True})

        if (len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == "scenes"):
            delete_scene(parts[2], parts[4])
            return self.send_json({"ok": True})

        self.send_err("Not found", 404)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Studio  →  http://localhost:{STUDIO_PORT}")
    print(f"ComfyUI →  {COMFYUI_URL}")
    server = HTTPServer(("0.0.0.0", STUDIO_PORT), StudioHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

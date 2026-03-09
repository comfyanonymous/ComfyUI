#!/usr/bin/env python3
"""
Studio server — film/TV production tool built on ComfyUI.
Serves the UI and manages projects, characters, and scenes.
"""

import json
import os
import sys
import uuid
import shutil
import mimetypes
import urllib.request
import urllib.error
import websocket
import threading
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ── Paths ──────────────────────────────────────────────────────────────────────

STUDIO_DIR = Path(__file__).parent
APP_DIR = STUDIO_DIR / "app"
DATA_DIR = STUDIO_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
WORKFLOWS_DIR = STUDIO_DIR / "workflows"

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
STUDIO_PORT = int(os.environ.get("STUDIO_PORT", "8189"))

# Ensure directories exist
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

# ── Project helpers ────────────────────────────────────────────────────────────

def list_projects():
    projects = []
    for p in sorted(PROJECTS_DIR.iterdir()):
        if p.is_dir():
            cfg = p / "config.json"
            if cfg.exists():
                data = json.loads(cfg.read_text())
                data["id"] = p.name
                projects.append(data)
    return projects

def get_project(project_id):
    cfg = PROJECTS_DIR / project_id / "config.json"
    if not cfg.exists():
        return None
    data = json.loads(cfg.read_text())
    data["id"] = project_id
    return data

def save_project(project_id, data):
    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    data_to_save = {k: v for k, v in data.items() if k != "id"}
    (project_dir / "config.json").write_text(json.dumps(data_to_save, indent=2))

def delete_project(project_id):
    project_dir = PROJECTS_DIR / project_id
    if project_dir.exists():
        shutil.rmtree(project_dir)

# ── Character helpers ──────────────────────────────────────────────────────────

def list_characters(project_id):
    chars_dir = PROJECTS_DIR / project_id / "characters"
    if not chars_dir.exists():
        return []
    chars = []
    for c in sorted(chars_dir.iterdir()):
        if c.is_dir():
            cfg = c / "config.json"
            if cfg.exists():
                data = json.loads(cfg.read_text())
                data["id"] = c.name
                data["project_id"] = project_id
                # List reference images
                refs_dir = c / "refs"
                if refs_dir.exists():
                    data["refs"] = [f.name for f in sorted(refs_dir.iterdir())
                                    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
                else:
                    data["refs"] = []
                chars.append(data)
    return chars

def get_character(project_id, char_id):
    cfg = PROJECTS_DIR / project_id / "characters" / char_id / "config.json"
    if not cfg.exists():
        return None
    data = json.loads(cfg.read_text())
    data["id"] = char_id
    data["project_id"] = project_id
    refs_dir = PROJECTS_DIR / project_id / "characters" / char_id / "refs"
    if refs_dir.exists():
        data["refs"] = [f.name for f in sorted(refs_dir.iterdir())
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    else:
        data["refs"] = []
    return data

def save_character(project_id, char_id, data):
    char_dir = PROJECTS_DIR / project_id / "characters" / char_id
    char_dir.mkdir(parents=True, exist_ok=True)
    (char_dir / "refs").mkdir(exist_ok=True)
    data_to_save = {k: v for k, v in data.items() if k not in ("id", "project_id", "refs")}
    (char_dir / "config.json").write_text(json.dumps(data_to_save, indent=2))

def delete_character(project_id, char_id):
    char_dir = PROJECTS_DIR / project_id / "characters" / char_id
    if char_dir.exists():
        shutil.rmtree(char_dir)

# ── Scene helpers ──────────────────────────────────────────────────────────────

def list_scenes(project_id):
    scenes_dir = PROJECTS_DIR / project_id / "scenes"
    if not scenes_dir.exists():
        return []
    scenes = []
    for f in sorted(scenes_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        data = json.loads(f.read_text())
        data["id"] = f.stem
        scenes.append(data)
    return scenes

def get_scene(project_id, scene_id):
    f = PROJECTS_DIR / project_id / "scenes" / f"{scene_id}.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    data["id"] = scene_id
    return data

def save_scene(project_id, scene_id, data):
    scenes_dir = PROJECTS_DIR / project_id / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    data_to_save = {k: v for k, v in data.items() if k != "id"}
    (scenes_dir / f"{scene_id}.json").write_text(json.dumps(data_to_save, indent=2))

def delete_scene(project_id, scene_id):
    f = PROJECTS_DIR / project_id / "scenes" / f"{scene_id}.json"
    if f.exists():
        f.unlink()

# ── ComfyUI proxy helpers ──────────────────────────────────────────────────────

def comfy_get(path):
    try:
        url = f"{COMFYUI_URL}{path}"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return None

def comfy_post(path, body):
    try:
        url = f"{COMFYUI_URL}{path}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def comfy_status():
    result = comfy_get("/system_stats")
    return result is not None

# ── Workflow assembly ──────────────────────────────────────────────────────────

def load_workflow_template(name):
    path = WORKFLOWS_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None

def assemble_workflow(scene, project, characters):
    """
    Build a ComfyUI workflow JSON from scene config, project style, and characters.
    Selects the right template based on output_type and character identity types.
    """
    output_type = scene.get("output_type", "still")
    char_ids = scene.get("characters", [])
    scene_chars = [c for c in characters if c["id"] in char_ids]

    # Determine character identity method
    has_lora = any(c.get("type") == "lora" for c in scene_chars)
    has_photomaker = any(c.get("type") == "photomaker" for c in scene_chars)

    if output_type == "video":
        template_name = "video_base"
    else:
        if has_photomaker:
            template_name = "still_photomaker"
        elif has_lora:
            template_name = "still_lora"
        else:
            template_name = "still_base"

    workflow = load_workflow_template(template_name)
    if workflow is None:
        # Fall back to basic still workflow
        workflow = load_workflow_template("still_base")
    if workflow is None:
        return None, f"Workflow template '{template_name}' not found"

    # Inject prompt
    prompt = build_prompt(scene, project, scene_chars)
    workflow = inject_prompt(workflow, prompt)

    # Inject style (negative prompt / style LoRA)
    style_prompt = project.get("style", {}).get("negative_prompt", "")
    workflow = inject_negative_prompt(workflow, style_prompt)

    # Inject LoRA if needed
    if has_lora:
        for char in scene_chars:
            if char.get("type") == "lora" and char.get("lora_file"):
                workflow = inject_lora(workflow, char["lora_file"], char.get("lora_weight", 0.8))

    return workflow, None

def build_prompt(scene, project, characters):
    parts = []
    # Style prefix from project
    style_prefix = project.get("style", {}).get("prompt_prefix", "")
    if style_prefix:
        parts.append(style_prefix)

    # PhotoMaker character tokens
    for char in characters:
        if char.get("type") == "photomaker":
            parts.append(f"{char['name']} img")

    # Scene description
    parts.append(scene.get("description", ""))

    # Style suffix
    style_suffix = project.get("style", {}).get("prompt_suffix", "")
    if style_suffix:
        parts.append(style_suffix)

    return ", ".join(p.strip() for p in parts if p.strip())

def inject_prompt(workflow, prompt):
    """Find CLIPTextEncode nodes with positive conditioning and inject the prompt."""
    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            inputs = node.get("inputs", {})
            # Heuristic: positive node has a non-empty text or no tag
            if inputs.get("_role") == "positive" or (
                    "text" in inputs and inputs.get("_role") != "negative"):
                inputs["text"] = prompt
                break
    return workflow

def inject_negative_prompt(workflow, negative_prompt):
    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            inputs = node.get("inputs", {})
            if inputs.get("_role") == "negative":
                inputs["text"] = negative_prompt
                break
    return workflow

def inject_lora(workflow, lora_file, weight):
    for node_id, node in workflow.items():
        if node.get("class_type") == "LoraLoader":
            node["inputs"]["lora_name"] = lora_file
            node["inputs"]["strength_model"] = weight
            node["inputs"]["strength_clip"] = weight
            break
    return workflow

# ── Multipart form parser ──────────────────────────────────────────────────────

def parse_multipart(rfile, content_type, content_length):
    """Minimal multipart/form-data parser returning {field: value} and {field: (filename, bytes)}."""
    import cgi
    import io
    body = rfile.read(content_length)
    environ = {
        "REQUEST_METHOD": "POST",
        "CONTENT_TYPE": content_type,
        "CONTENT_LENGTH": str(content_length),
    }
    form = cgi.FieldStorage(fp=io.BytesIO(body), environ=environ)
    fields = {}
    files = {}
    for key in form.keys():
        item = form[key]
        if item.filename:
            files[key] = (item.filename, item.file.read())
        else:
            fields[key] = item.value
    return fields, files

# ── HTTP handler ───────────────────────────────────────────────────────────────

class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quieter logging
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(format, *args)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message, status=400):
        self.send_json({"error": message}, status)

    def send_file(self, path):
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # ── Static files ──
        if path == "/" or path == "/index.html":
            f = APP_DIR / "index.html"
            if f.exists():
                return self.send_file(f)
        if not path.startswith("/api/"):
            # Try to serve from app/
            candidate = APP_DIR / path.lstrip("/")
            if candidate.exists() and candidate.is_file():
                return self.send_file(candidate)

        # ── API ──
        parts = [p for p in path.split("/") if p]
        # parts[0] = "api"

        # GET /api/status
        if parts == ["api", "status"]:
            return self.send_json({"comfyui": comfy_status(), "studio": True})

        # GET /api/projects
        if parts == ["api", "projects"]:
            return self.send_json(list_projects())

        # GET /api/projects/:id
        if len(parts) == 3 and parts[:2] == ["api", "projects"]:
            project = get_project(parts[2])
            if not project:
                return self.send_error_json("Project not found", 404)
            return self.send_json(project)

        # GET /api/projects/:id/characters
        if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "characters":
            return self.send_json(list_characters(parts[2]))

        # GET /api/projects/:id/characters/:char_id
        if len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == "characters":
            char = get_character(parts[2], parts[4])
            if not char:
                return self.send_error_json("Character not found", 404)
            return self.send_json(char)

        # GET /api/projects/:id/characters/:char_id/refs/:filename
        if (len(parts) == 7 and parts[:2] == ["api", "projects"]
                and parts[3] == "characters" and parts[5] == "refs"):
            ref_path = PROJECTS_DIR / parts[2] / "characters" / parts[4] / "refs" / parts[6]
            if not ref_path.exists():
                return self.send_error_json("File not found", 404)
            return self.send_file(ref_path)

        # GET /api/projects/:id/scenes
        if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "scenes":
            return self.send_json(list_scenes(parts[2]))

        # GET /api/projects/:id/scenes/:scene_id
        if len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == "scenes":
            scene = get_scene(parts[2], parts[4])
            if not scene:
                return self.send_error_json("Scene not found", 404)
            return self.send_json(scene)

        # GET /api/comfy/outputs/:filename  — proxy output images/videos from ComfyUI
        if len(parts) == 4 and parts[:3] == ["api", "comfy", "outputs"]:
            filename = parts[3]
            try:
                url = f"{COMFYUI_URL}/view?filename={filename}&type=output"
                with urllib.request.urlopen(url, timeout=10) as r:
                    data = r.read()
                    ct = r.headers.get("Content-Type", "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                return self.send_error_json(str(e), 502)
            return

        # GET /api/comfy/queue/:prompt_id — poll job status
        if len(parts) == 4 and parts[:3] == ["api", "comfy", "queue"]:
            prompt_id = parts[3]
            history = comfy_get(f"/history/{prompt_id}")
            if history and prompt_id in history:
                entry = history[prompt_id]
                outputs = entry.get("outputs", {})
                files = []
                for node_id, node_out in outputs.items():
                    for img in node_out.get("images", []):
                        files.append(img["filename"])
                    for vid in node_out.get("videos", []):
                        files.append(vid["filename"])
                return self.send_json({"status": "done", "files": files})
            # Check if still queued
            queue = comfy_get("/queue")
            if queue:
                running = [item[1] for item in queue.get("queue_running", [])]
                pending = [item[1] for item in queue.get("queue_pending", [])]
                if prompt_id in running:
                    return self.send_json({"status": "running"})
                if prompt_id in pending:
                    return self.send_json({"status": "pending"})
            return self.send_json({"status": "unknown"})

        self.send_error_json("Not found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = [p for p in path.split("/") if p]
        ct = self.headers.get("Content-Type", "")

        # POST /api/projects
        if parts == ["api", "projects"]:
            body = self.read_json_body()
            project_id = body.get("id") or str(uuid.uuid4())[:8]
            if not body.get("name"):
                return self.send_error_json("name is required")
            data = {
                "name": body["name"],
                "description": body.get("description", ""),
                "style": body.get("style", {"prompt_prefix": "", "prompt_suffix": "", "negative_prompt": ""}),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            save_project(project_id, data)
            data["id"] = project_id
            return self.send_json(data, 201)

        # POST /api/projects/:id/characters
        if (len(parts) == 4 and parts[:2] == ["api", "projects"]
                and parts[3] == "characters"):
            project_id = parts[2]
            if not get_project(project_id):
                return self.send_error_json("Project not found", 404)

            if "multipart/form-data" in ct:
                length = int(self.headers.get("Content-Length", 0))
                fields, files = parse_multipart(self.rfile, ct, length)
                name = fields.get("name", "")
                char_type = fields.get("type", "photomaker")
                lora_file = fields.get("lora_file", "")
                lora_weight = float(fields.get("lora_weight", "0.8"))
            else:
                body = self.read_json_body()
                name = body.get("name", "")
                char_type = body.get("type", "photomaker")
                lora_file = body.get("lora_file", "")
                lora_weight = float(body.get("lora_weight", 0.8))
                files = {}

            if not name:
                return self.send_error_json("name is required")

            char_id = name.lower().replace(" ", "_")
            data = {
                "name": name,
                "type": char_type,
                "lora_file": lora_file,
                "lora_weight": lora_weight,
                "notes": fields.get("notes", "") if "multipart/form-data" in ct else "",
            }
            save_character(project_id, char_id, data)

            # Save any uploaded ref images
            refs_dir = PROJECTS_DIR / project_id / "characters" / char_id / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            for field_name, (filename, file_bytes) in files.items():
                if field_name.startswith("ref"):
                    safe_name = Path(filename).name
                    (refs_dir / safe_name).write_bytes(file_bytes)

            data["id"] = char_id
            data["refs"] = [f.name for f in sorted(refs_dir.iterdir())
                            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
            return self.send_json(data, 201)

        # POST /api/projects/:id/characters/:char_id/refs  — upload reference photos
        if (len(parts) == 6 and parts[:2] == ["api", "projects"]
                and parts[3] == "characters" and parts[5] == "refs"):
            project_id, char_id = parts[2], parts[4]
            if not get_character(project_id, char_id):
                return self.send_error_json("Character not found", 404)
            if "multipart/form-data" not in ct:
                return self.send_error_json("Expected multipart/form-data")
            length = int(self.headers.get("Content-Length", 0))
            _, files = parse_multipart(self.rfile, ct, length)
            refs_dir = PROJECTS_DIR / project_id / "characters" / char_id / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for field_name, (filename, file_bytes) in files.items():
                safe_name = Path(filename).name
                (refs_dir / safe_name).write_bytes(file_bytes)
                saved.append(safe_name)
            return self.send_json({"saved": saved})

        # POST /api/projects/:id/scenes
        if (len(parts) == 4 and parts[:2] == ["api", "projects"]
                and parts[3] == "scenes"):
            project_id = parts[2]
            project = get_project(project_id)
            if not project:
                return self.send_error_json("Project not found", 404)
            body = self.read_json_body()
            if not body.get("description"):
                return self.send_error_json("description is required")
            scene_id = str(uuid.uuid4())[:8]
            scene_data = {
                "title": body.get("title", ""),
                "description": body["description"],
                "characters": body.get("characters", []),
                "output_type": body.get("output_type", "still"),
                "status": "pending",
                "outputs": [],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            save_scene(project_id, scene_id, scene_data)

            # Submit to ComfyUI if requested
            if body.get("generate", False):
                characters = list_characters(project_id)
                workflow, err = assemble_workflow(scene_data, project, characters)
                if err:
                    scene_data["status"] = "error"
                    scene_data["error"] = err
                    save_scene(project_id, scene_id, scene_data)
                    scene_data["id"] = scene_id
                    return self.send_json(scene_data, 201)

                client_id = str(uuid.uuid4())
                result = comfy_post("/prompt", {"prompt": workflow, "client_id": client_id})
                prompt_id = result.get("prompt_id")
                if prompt_id:
                    scene_data["status"] = "generating"
                    scene_data["prompt_id"] = prompt_id
                else:
                    scene_data["status"] = "error"
                    scene_data["error"] = result.get("error", "ComfyUI submission failed")
                save_scene(project_id, scene_id, scene_data)

            scene_data["id"] = scene_id
            return self.send_json(scene_data, 201)

        # POST /api/projects/:id/scenes/:scene_id/generate  — trigger generation for existing scene
        if (len(parts) == 6 and parts[:2] == ["api", "projects"]
                and parts[3] == "scenes" and parts[5] == "generate"):
            project_id, scene_id = parts[2], parts[4]
            project = get_project(project_id)
            scene = get_scene(project_id, scene_id)
            if not project or not scene:
                return self.send_error_json("Project or scene not found", 404)
            characters = list_characters(project_id)
            workflow, err = assemble_workflow(scene, project, characters)
            if err:
                return self.send_error_json(err)
            client_id = str(uuid.uuid4())
            result = comfy_post("/prompt", {"prompt": workflow, "client_id": client_id})
            prompt_id = result.get("prompt_id")
            if prompt_id:
                scene["status"] = "generating"
                scene["prompt_id"] = prompt_id
            else:
                scene["status"] = "error"
                scene["error"] = result.get("error", "ComfyUI submission failed")
            save_scene(project_id, scene_id, scene)
            return self.send_json(scene)

        self.send_error_json("Not found", 404)

    def do_PUT(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = [p for p in path.split("/") if p]

        # PUT /api/projects/:id
        if len(parts) == 3 and parts[:2] == ["api", "projects"]:
            project_id = parts[2]
            existing = get_project(project_id)
            if not existing:
                return self.send_error_json("Project not found", 404)
            body = self.read_json_body()
            existing.update({k: v for k, v in body.items() if k != "id"})
            save_project(project_id, existing)
            return self.send_json(existing)

        # PUT /api/projects/:id/characters/:char_id
        if (len(parts) == 5 and parts[:2] == ["api", "projects"]
                and parts[3] == "characters"):
            project_id, char_id = parts[2], parts[4]
            existing = get_character(project_id, char_id)
            if not existing:
                return self.send_error_json("Character not found", 404)
            body = self.read_json_body()
            existing.update({k: v for k, v in body.items() if k not in ("id", "project_id", "refs")})
            save_character(project_id, char_id, existing)
            return self.send_json(existing)

        # PUT /api/projects/:id/scenes/:scene_id
        if (len(parts) == 5 and parts[:2] == ["api", "projects"]
                and parts[3] == "scenes"):
            project_id, scene_id = parts[2], parts[4]
            existing = get_scene(project_id, scene_id)
            if not existing:
                return self.send_error_json("Scene not found", 404)
            body = self.read_json_body()
            existing.update({k: v for k, v in body.items() if k != "id"})
            save_scene(project_id, scene_id, existing)
            return self.send_json(existing)

        self.send_error_json("Not found", 404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = [p for p in path.split("/") if p]

        if len(parts) == 3 and parts[:2] == ["api", "projects"]:
            delete_project(parts[2])
            return self.send_json({"ok": True})

        if (len(parts) == 5 and parts[:2] == ["api", "projects"]
                and parts[3] == "characters"):
            delete_character(parts[2], parts[4])
            return self.send_json({"ok": True})

        if (len(parts) == 5 and parts[:2] == ["api", "projects"]
                and parts[3] == "scenes"):
            delete_scene(parts[2], parts[4])
            return self.send_json({"ok": True})

        # DELETE /api/projects/:id/characters/:char_id/refs/:filename
        if (len(parts) == 7 and parts[:2] == ["api", "projects"]
                and parts[3] == "characters" and parts[5] == "refs"):
            ref_path = PROJECTS_DIR / parts[2] / "characters" / parts[4] / "refs" / parts[6]
            if ref_path.exists():
                ref_path.unlink()
            return self.send_json({"ok": True})

        self.send_error_json("Not found", 404)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Studio server starting on http://localhost:{STUDIO_PORT}")
    print(f"ComfyUI expected at {COMFYUI_URL}")
    server = HTTPServer(("0.0.0.0", STUDIO_PORT), StudioHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

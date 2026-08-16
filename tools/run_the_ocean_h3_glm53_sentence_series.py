#!/usr/bin/env python3
"""Render THE OCEAN from sentence-audited Z.AI GLM-5.3 H3 prompts."""

import copy
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path("/home/op/ai/ComfyUI")
SERVER = "http://127.0.0.1:8188"
SINGLE_SOURCE = ROOT / "workflows/generated/zoe-missed-call-zero-10step-992x544-native-v1.json"
MULTI_SOURCE = ROOT / "workflows/adapted/minimax-h3-fl2va-w4a8-qwen32vl-int8-gpu1-optimized-multishot-30s-api.json"
PROMPT_MANIFEST = ROOT / "logs/the-ocean-glm53-sentence-prompts-20260816/manifest.json"
FRAME_ROOT = ROOT / "input/the-ocean-20260810"
WORKFLOW_ROOT = ROOT / "workflows/generated/the-ocean-h3-glm53-sentence-20step-960x544-20260816"
OUTPUT_ROOT = ROOT / "output/video/minimax-h3"
SUMMARY = ROOT / "logs/the-ocean-h3-glm53-sentence-20step-20260816-summary.json"
PRIOR_UNIT = "harry-piedra-faso-qwen38-resume.service"
UPSCALER_PYTHON = Path("/home/op/ai/h3-upscaler-eval/conda/bin/python")
UPSCALER = Path("/home/op/ai/h3-upscaler-eval/upscale_nvidia_vfx_ultra.py")
UPSCALER_METRICS = Path("/home/op/ai/h3-upscaler-eval/results/the-ocean-h3-glm53-sentence-20step-vfx-ultra-20260816.json")
NATIVE_MASTER = OUTPUT_ROOT / "the-ocean-h3-glm53-sentence-20step-native-960x544-4m18s-20260816.mp4"
FINAL_MASTER = OUTPUT_ROOT / "the-ocean-h3-glm53-sentence-20step-rtx-vfx-ultra-1920x1080-4m18s-20260816.mp4"


def request_json(path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        SERVER + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def prior_active():
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", PRIOR_UNIT]
    ).returncode == 0


def queue_idle():
    queue = request_json("/queue")
    return not queue.get("queue_running") and not queue.get("queue_pending")


def gpu_utilization():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in raw.splitlines():
        if line.strip():
            index, util, used, free = [int(value.strip()) for value in line.split(",")]
            rows.append({"index": index, "utilization": util, "memory_used": used, "memory_free": free})
    return rows


def wait_for_safe_idle(require_gpu0_headroom=False):
    while True:
        rows = gpu_utilization()
        enough = not require_gpu0_headroom or next(row for row in rows if row["index"] == 0)["memory_free"] >= 8000
        if not prior_active() and queue_idle() and all(row["utilization"] < 15 for row in rows) and enough:
            return rows
        time.sleep(15)


def ffprobe(path):
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=codec_name,codec_type,width,height,nb_frames,r_frame_rate,sample_rate,channels",
            "-of", "json", str(path),
        ],
        text=True,
    )
    data = json.loads(raw)
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"], check=True)
    return data


def assert_video(
    data, width, height, frames=None, duration=None, tolerance=0.08, *, silent=False
):
    video = next((stream for stream in data["streams"] if stream.get("codec_type") == "video"), None)
    if not video or video.get("width") != width or video.get("height") != height:
        raise RuntimeError(f"expected {width}x{height} video; got {video}")
    if frames is not None and int(video.get("nb_frames", -1)) != frames:
        raise RuntimeError(f"expected {frames} frames; got {video.get('nb_frames')}")
    if duration is not None and abs(float(data["format"]["duration"]) - duration) > tolerance:
        raise RuntimeError(f"expected duration {duration}; got {data['format']['duration']}")
    if silent and any(stream.get("codec_type") == "audio" for stream in data["streams"]):
        raise RuntimeError("expected a silent video with zero audio streams")


def load_state():
    if SUMMARY.exists():
        return json.loads(SUMMARY.read_text())
    return {"status": "waiting", "scenes": [], "created_epoch": time.time()}


def write_state(state):
    temporary = SUMMARY.with_suffix(SUMMARY.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(SUMMARY)


def frame_name(scene):
    if scene == 1:
        return "the-ocean-20260810/vladykeri.png"
    return f"the-ocean-20260810/scene-{scene:02d}-first-frame-chatgpt.png"


def build_graph(item, prompt, workflow_path):
    scene = item["scene"]
    seed = 1786355000 + scene
    prefix = f"video/minimax-h3/the-ocean-scene-{scene:02d}-glm53-sentence-20step-960x544-20260816"
    if item["shots"] == 3:
        graph = json.loads(MULTI_SOURCE.read_text())
        zoe = json.loads(SINGLE_SOURCE.read_text())
        # H3's parser treats any script beginning with '[' as JSON. The
        # formatter's validated shot markers intentionally begin with
        # '[Shot N]', so add a plain-text preamble before posting the graph;
        # the parser still splits the three narratives on standalone '---'.
        if prompt.lstrip().startswith("["):
            prompt = "Multishot plain-text script:\n\n" + prompt
        for node in ("1", "2", "3", "4"):
            graph[node]["inputs"] = copy.deepcopy(zoe[node]["inputs"])
            if "_meta" in zoe[node]:
                graph[node]["_meta"] = copy.deepcopy(zoe[node]["_meta"])
        graph["5"]["inputs"].update(
            {
                "script": prompt,
                "shot_count": 0,
                "width": 960,
                "height": 544,
                "frames_per_shot": 243,
                "seed": seed,
                "steps": 20,
                "seed_per_shot": True,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "start_image": ["18", 0],
                "reference_images": ["18", 0],
                "reference_image_size": "max",
            }
        )
        graph["7"]["inputs"]["filename_prefix"] = prefix
        graph["18"] = {"class_type": "LoadImage", "inputs": {"image": frame_name(scene)}}
        expected_frames = 727
        mode = "multishot-3x243"
    else:
        graph = json.loads(SINGLE_SOURCE.read_text())
        graph["5"]["inputs"].update(
            {
                "width": 960,
                "height": 544,
                "length": 362,
                "prompt": prompt,
                "ref_image_size": "max",
            }
        )
        graph["9"]["inputs"]["noise_seed"] = seed
        graph["11"]["inputs"]["steps"] = 20
        graph["16"]["inputs"]["filename_prefix"] = prefix
        graph["18"]["inputs"]["image"] = frame_name(scene)
        expected_frames = 362
        mode = "zoe-single-362"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(json.dumps(graph, indent=2) + "\n")
    return graph, prefix, seed, expected_frames, mode


def wait_history(prompt_id, timeout=10800):
    started = time.time()
    deadline = started + timeout
    while time.time() < deadline:
        history = request_json("/history/" + prompt_id)
        if prompt_id in history:
            item = history[prompt_id]
            status = item.get("status", {})
            status_text = str(status.get("status_str", "")).lower()
            messages = status.get("messages", [])
            execution_errors = [
                message for message in messages
                if isinstance(message, (list, tuple))
                and message
                and message[0] in {"execution_error", "execution_interrupted"}
            ]
            if status_text in {"error", "failed"} or execution_errors:
                raise RuntimeError(
                    f"render {prompt_id} failed: status={status_text or 'unknown'} "
                    f"errors={execution_errors!r}"
                )
            if status.get("completed"):
                videos = []
                for output in item.get("outputs", {}).values():
                    # ComfyUI's SaveVideo reports an MP4 under ``images``
                    # (with ``animated: true``) on this server.  Older
                    # versions exposed the same descriptor under ``videos``.
                    # Accept both, then let resolve_video reject actual still
                    # images by extension.
                    videos.extend(output.get("videos", []))
                    videos.extend(output.get("images", []))
                return item, videos, time.time() - started
        time.sleep(5)
    raise RuntimeError(f"render {prompt_id} timed out")


def resolve_video(videos):
    video_extensions = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    videos = [video for video in videos if Path(video.get("filename", "")).suffix.lower() in video_extensions]
    if not videos:
        raise RuntimeError("ComfyUI history completed without a video descriptor")
    video = videos[-1]
    subfolder = video.get("subfolder", "")
    return ROOT / "output" / subfolder / video["filename"]


def valid_completed(existing, expected_frames):
    try:
        path = Path(existing["native"])
        data = ffprobe(path)
        assert_video(data, 960, 544, frames=expected_frames)
        return True
    except (KeyError, OSError, RuntimeError, subprocess.SubprocessError, ValueError):
        return False


def render_scenes(state, manifest):
    completed = {item["scene"]: item for item in state.get("scenes", []) if item.get("status") == "complete"}
    results = []
    for item in manifest:
        scene = item["scene"]
        expected_frames = 727 if item["shots"] == 3 else 362
        if scene in completed and valid_completed(completed[scene], expected_frames):
            results.append(completed[scene])
            continue
        wait_snapshot = wait_for_safe_idle()
        prompt = Path(item["prompt"]).read_text().strip()
        workflow = WORKFLOW_ROOT / f"scene-{scene:02d}-api.json"
        graph, prefix, seed, expected_frames, mode = build_graph(item, prompt, workflow)
        queued_epoch = time.time()
        response = request_json("/prompt", {"prompt": graph})
        prompt_id = response["prompt_id"]
        history, videos, execution_seconds = wait_history(prompt_id)
        native = resolve_video(videos)
        if native.stat().st_mtime < queued_epoch:
            raise RuntimeError(f"scene {scene:02d} resolved to stale output {native}")
        probe = ffprobe(native)
        assert_video(probe, 960, 544, frames=expected_frames)
        scene_result = {
            "scene": scene,
            "status": "complete",
            "mode": mode,
            "prompt": item["prompt"],
            "first_frame": str(ROOT / "input" / frame_name(scene)),
            "workflow": str(workflow),
            "source_template": str(SINGLE_SOURCE),
            "gpu_distribution": {
                "diffusion": "cuda:0 donor cuda:1",
                "qwen3_vl_32b_int8": "cuda:2 donor cuda:1",
                "video_vae": "cuda:1",
                "audio_vae": "cuda:1",
            },
            "steps": 20,
            "seed": seed,
            "prompt_id": prompt_id,
            "queued_epoch": queued_epoch,
            "execution_seconds": execution_seconds,
            "render_output_seconds": expected_frames / 24,
            "seconds_per_output_second": execution_seconds / (expected_frames / 24),
            "gpu_before": wait_snapshot,
            "native": str(native),
            "native_probe": probe,
            "history_status": history.get("status", {}),
        }
        results.append(scene_result)
        state.update({"status": "rendering", "current_scene": scene, "scenes": results})
        write_state(state)
    return results


def assemble_master(results, manifest):
    inputs = []
    filters = []
    streams = []
    for index, (result, item) in enumerate(zip(results, manifest)):
        inputs += ["-i", result["native"]]
        duration = item["nominal_seconds"]
        filters.append(
            f"[{index}:v]trim=start=0:duration={duration},setpts=PTS-STARTPTS,fps=24,"
            f"scale=960:544:flags=lanczos,setsar=1,format=yuv420p[v{index}]"
        )
        streams.append(f"[v{index}]")
    filters.append("".join(streams) + f"concat=n={len(results)}:v=1:a=0[outv]")
    command = [
        "ffmpeg", "-y", "-v", "error", *inputs,
        "-filter_complex", ";".join(filters), "-map", "[outv]",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-pix_fmt", "yuv420p", "-r", "24", "-movflags", "+faststart", str(NATIVE_MASTER),
    ]
    subprocess.run(command, check=True)
    probe = ffprobe(NATIVE_MASTER)
    assert_video(probe, 960, 544, frames=6192, duration=258.0, silent=True)
    return probe


def upscale_master():
    wait_for_safe_idle(require_gpu0_headroom=True)
    UPSCALER_METRICS.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(UPSCALER_PYTHON), str(UPSCALER), str(NATIVE_MASTER), str(FINAL_MASTER),
            "--metrics", str(UPSCALER_METRICS), "--device", "0",
        ],
        check=True,
    )
    probe = ffprobe(FINAL_MASTER)
    assert_video(probe, 1920, 1080, frames=6192, duration=258.0, silent=True)
    return probe


def validate_inputs(manifest):
    if len(manifest) != 12 or [item["scene"] for item in manifest] != list(range(1, 13)):
        raise RuntimeError("prompt manifest must contain scenes 1 through 12")
    for item in manifest:
        prompt = Path(item["prompt"])
        frame = ROOT / "input" / frame_name(item["scene"])
        if not prompt.is_file() or not frame.is_file():
            raise RuntimeError(f"scene {item['scene']:02d} is missing prompt or first frame")
        if item["shots"] == 3:
            parts = [part.strip() for part in prompt.read_text().split("\n---\n") if part.strip()]
            if len(parts) != 3 or any(part.count("[Shot ") != 1 for part in parts):
                raise RuntimeError(f"scene {item['scene']:02d} multishot prompt failed validation")


def main():
    prompt_state = json.loads(PROMPT_MANIFEST.read_text())
    if prompt_state.get("status") != "DONE":
        raise RuntimeError("GLM-5.3 sentence prompt manifest is not complete")
    formatter = prompt_state.get("formatter", {})
    if formatter.get("model") != "glm-5.3":
        raise RuntimeError("prompt manifest is not pinned to Z.AI GLM-5.3")
    if prompt_state.get("sentence_count") != 58:
        raise RuntimeError(
            "prompt manifest must contain the 58 independently formatted grammatical source sentences"
        )
    if formatter.get("granularity") != (
        "one independent call per grammatical source sentence"
    ):
        raise RuntimeError("prompt manifest is not strict sentence-authoritative output")
    if len(prompt_state.get("records", [])) != 58:
        raise RuntimeError("prompt manifest does not contain 58 sentence-level records")
    if any(record.get("status") != "PASS" for record in prompt_state["records"]):
        raise RuntimeError("one or more GLM-5.3 sentence formatter records failed validation")
    for record in prompt_state["records"]:
        provenance = record.get("response_provenance", {})
        cache_record = Path(record.get("cache_record", ""))
        if (
            provenance.get("provider_model") != "glm-5.3"
            or not provenance.get("provider_response_id")
            or not cache_record.is_file()
            or hashlib.sha256(cache_record.read_bytes()).hexdigest()
            != record.get("cache_record_sha256")
        ):
            raise RuntimeError(
                f"sentence {record.get('unit')} lacks provider-auditable GLM-5.3 provenance"
            )
    manifest = prompt_state["scenes"]
    validate_inputs(manifest)
    state = load_state()
    state.update(
        {
            "status": "waiting_for_prior_series",
            "source_template": str(SINGLE_SOURCE),
            "canvas": [960, 544],
            "steps": 20,
            "timeline_seconds": 258,
            "formatter_model": "glm-5.3",
            "formatter_granularity": "one call per grammatical source sentence",
            "upscaler": "NVIDIA VFX VideoSuperRes ULTRA 2x",
        }
    )
    write_state(state)
    results = render_scenes(state, manifest)
    state.update({"status": "assembling", "current_scene": None, "scenes": results})
    write_state(state)
    native_probe = assemble_master(results, manifest)
    state.update({"status": "upscaling", "native_master": str(NATIVE_MASTER), "native_master_probe": native_probe})
    write_state(state)
    final_probe = upscale_master()
    state.update(
        {
            "status": "complete",
            "completed_epoch": time.time(),
            "final_master": str(FINAL_MASTER),
            "final_master_probe": final_probe,
            "upscale_metrics": str(UPSCALER_METRICS),
        }
    )
    write_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        state = load_state()
        state.update({"status": "error", "error": str(error), "failed_epoch": time.time()})
        write_state(state)
        raise

import asyncio
import base64
from io import BytesIO
import ipaddress
import json
import os
from pathlib import Path
import urllib.parse
import uuid

from typing_extensions import override

import aiohttp
import numpy as np
from PIL import Image
import torch

from comfy_api.latest import ComfyExtension, InputImpl, io

DEFAULT_VLLM_BASE_URL = "http://localhost:8000"
VLLM_TIMEOUT = 3600
COSMOS3_WIDTH = 1280
COSMOS3_HEIGHT = 720
COSMOS3_NUM_FRAMES = 189
COSMOS3_FPS = 24
COSMOS3_ACTION_IMAGE_SIZE = 480
Cosmos3GenerationSettings = io.Custom("COSMOS3_GENERATION_SETTINGS")


def _load_prompt(prompt_path, input_name):
    path = Path(prompt_path.strip()).expanduser()
    if not path.is_file():
        raise ValueError(f"{input_name} file does not exist: {path}")
    prompt = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(prompt, ensure_ascii=True, separators=(",", ":")), prompt


def _prompt_generation_settings(prompt):
    if not isinstance(prompt, dict):
        return None

    settings = {}
    resolution = prompt.get("resolution")
    if resolution is not None:
        if (
            not isinstance(resolution, dict)
            or not isinstance(resolution.get("W"), int)
            or not isinstance(resolution.get("H"), int)
            or resolution["W"] <= 0
            or resolution["H"] <= 0
        ):
            raise ValueError("prompt resolution must contain positive integer W and H values")
        settings["width"] = resolution["W"]
        settings["height"] = resolution["H"]

    fps = prompt.get("fps")
    if fps is not None:
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError("prompt fps must be a positive integer")
        settings["fps"] = fps

    num_frames = prompt.get("num_frames")
    if num_frames is not None:
        if not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError("prompt num_frames must be a positive integer")
        settings["num_frames"] = num_frames

    return settings or None


def _apply_generation_settings(generation_settings, width, height, num_frames=None, fps=None):
    if generation_settings is None:
        return width, height, num_frames, fps
    if not isinstance(generation_settings, dict):
        raise ValueError("generation_settings must come from Cosmos3 Prompt (API)")
    return (
        generation_settings.get("width", width),
        generation_settings.get("height", height),
        generation_settings.get("num_frames", num_frames),
        generation_settings.get("fps", fps),
    )


def _vllm_api_url(base_url, endpoint):
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("base_url must be an HTTP URL")

    try:
        is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        is_loopback = parsed.hostname == "localhost"
    if not is_loopback:
        raise ValueError("Cosmos3 vLLM-Omni nodes only connect to a loopback server")

    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return urllib.parse.urlunparse(parsed._replace(path=f"{path}/{endpoint}", params="", query="", fragment=""))


def _vllm_headers(accept):
    headers = {"Accept": accept}
    api_key = os.environ.get("COSMOS3_VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _vllm_post(url, body, headers):
    try:
        timeout = aiohttp.ClientTimeout(total=VLLM_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=body, headers=headers) as response:
                content = await response.read()
                if response.status >= 400:
                    details = content.decode("utf-8", errors="replace")
                    raise RuntimeError(f"vLLM-Omni request failed ({response.status}): {details[:1000]}")
                return content
    except aiohttp.ClientError as error:
        raise RuntimeError(f"Could not reach the vLLM-Omni server: {error}") from error


async def _vllm_get(url, accept):
    try:
        timeout = aiohttp.ClientTimeout(total=VLLM_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=_vllm_headers(accept)) as response:
                content = await response.read()
                if response.status >= 400:
                    details = content.decode("utf-8", errors="replace")
                    raise RuntimeError(f"vLLM-Omni request failed ({response.status}): {details[:1000]}")
                return content
    except aiohttp.ClientError as error:
        raise RuntimeError(f"Could not reach the vLLM-Omni server: {error}") from error


async def _vllm_post_json(url, payload):
    body = json.dumps(payload).encode("utf-8")
    headers = _vllm_headers("application/json")
    headers["Content-Type"] = "application/json"
    response = await _vllm_post(url, body, headers)
    return json.loads(response)


def _vllm_multipart(fields, file=None):
    boundary = uuid.uuid4().hex
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode())
        body.extend(b"\r\n")

    if file is not None:
        name, filename, content_type, data = file
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode())
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(data)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), boundary


async def _vllm_post_video(url, fields, file=None):
    body, boundary = _vllm_multipart(fields, file)
    headers = _vllm_headers("application/json")
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    response = await _vllm_post(url, body, headers)
    return json.loads(response)


async def _vllm_generate_video(base_url, fields, file=None):
    job = await _vllm_post_video(_vllm_api_url(base_url, "videos"), fields, file)
    video_id = job.get("id") if isinstance(job, dict) else None
    if not video_id:
        raise RuntimeError("vLLM-Omni returned an invalid video job response")

    endpoint = f"videos/{urllib.parse.quote(str(video_id), safe='')}"
    deadline = asyncio.get_running_loop().time() + VLLM_TIMEOUT
    while True:
        status_response = json.loads(await _vllm_get(_vllm_api_url(base_url, endpoint), "application/json"))
        status = status_response.get("status") if isinstance(status_response, dict) else None
        if status == "completed":
            return await _vllm_get(_vllm_api_url(base_url, f"{endpoint}/content"), "video/mp4")
        if status in ("failed", "cancelled", "expired"):
            details = status_response.get("error") or status_response.get("failure_reason") or status
            raise RuntimeError(f"vLLM-Omni video generation {status}: {details}")
        if status not in ("queued", "in_progress"):
            raise RuntimeError("vLLM-Omni returned an invalid video job status")

        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RuntimeError(f"vLLM-Omni video generation did not finish within {VLLM_TIMEOUT} seconds")
        await asyncio.sleep(min(2.0, remaining))


def _vllm_decode_image(response):
    try:
        encoded = response["data"][0]["b64_json"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("vLLM-Omni returned an invalid image response") from error
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _vllm_encode_image(image):
    array = image[0].detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    output = BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


def _vllm_video_fields(prompt, negative_prompt, width, height, num_frames, fps, steps, guidance_scale, flow_shift, seed, generate_sound, guardrails):
    fields = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "size": f"{width}x{height}",
        "num_frames": num_frames,
        "fps": fps,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "flow_shift": flow_shift,
        "seed": seed,
        "extra_params": json.dumps({
            "use_resolution_template": False,
            "use_duration_template": False,
            "guardrails": guardrails,
        }),
        "generate_sound": generate_sound,
        "sound_duration": f"{num_frames / fps:.2f}",
    }
    return fields


def _vllm_generation_inputs(include_image=False, include_video_options=False, include_sound_option=True):
    inputs = []
    if include_image:
        inputs.append(io.Image.Input("image"))
    inputs.extend([
        io.String.Input("prompt", force_input=True),
        io.String.Input("negative_prompt", default="", force_input=True, optional=True),
        io.String.Input("base_url", default=DEFAULT_VLLM_BASE_URL),
        io.Int.Input("width", default=COSMOS3_WIDTH, min=64, max=4096, step=16),
        io.Int.Input("height", default=COSMOS3_HEIGHT, min=64, max=4096, step=16),
    ])
    if include_video_options:
        inputs.extend([
            io.Int.Input("num_frames", default=COSMOS3_NUM_FRAMES, min=1, max=1000),
            io.Int.Input("fps", default=COSMOS3_FPS, min=1, max=120),
        ])
    inputs.extend([
        io.Int.Input("steps", default=35, min=1, max=1000),
        io.Float.Input("guidance_scale", default=6.0, min=0.0, max=100.0, step=0.1),
        io.Float.Input("flow_shift", default=10.0, min=0.0, max=100.0, step=0.1),
        io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
    ])
    if include_video_options and include_sound_option:
        inputs.append(io.Boolean.Input("generate_sound", default=False))
    inputs.append(io.Boolean.Input("guardrails", default=True))
    inputs.append(Cosmos3GenerationSettings.Input(
        "generation_settings",
        optional=True,
        tooltip="Optional settings from Cosmos3 Prompt (API) that override the matching inputs.",
    ))
    return inputs


class Cosmos3PromptAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Cosmos3PromptAPI",
            display_name="Cosmos3 Prompt (API)",
            category="partner/text/Cosmos3",
            description="Loads raw or JSON Cosmos3 prompts for a local vLLM-Omni server.",
            inputs=[
                io.Combo.Input(
                    "prompt_input_mode",
                    options=["manual_text", "file_path"],
                    default="manual_text",
                    tooltip="Use manual text, or load each non-empty prompt input from a JSON file.",
                ),
                io.String.Input("prompt", multiline=True, default=""),
                io.String.Input("negative_prompt", multiline=True, default=""),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="negative_prompt"),
                Cosmos3GenerationSettings.Output(display_name="generation_settings"),
            ],
        )

    @classmethod
    def execute(cls, prompt_input_mode, prompt, negative_prompt):
        generation_settings = None
        if prompt_input_mode == "file_path":
            prompt, prompt_data = _load_prompt(prompt, "prompt")
            generation_settings = _prompt_generation_settings(prompt_data)
            if negative_prompt:
                negative_prompt, _ = _load_prompt(negative_prompt, "negative_prompt")
        return io.NodeOutput(
            prompt,
            negative_prompt,
            generation_settings,
        )


class Cosmos3TextToImageAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Cosmos3TextToImageAPI",
            display_name="Cosmos3 Text to Image (API)",
            category="partner/image/Cosmos",
            description="Generates an image with a local Cosmos3 vLLM-Omni server.",
            inputs=_vllm_generation_inputs(),
            outputs=[io.Image.Output()],
        )

    @classmethod
    async def execute(cls, prompt, base_url, width, height, steps, guidance_scale, flow_shift, seed, guardrails, negative_prompt="", generation_settings=None):
        width, height, _, _ = _apply_generation_settings(generation_settings, width, height)
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "size": f"{width}x{height}",
            "n": 1,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "seed": seed,
            "extra_args": {
                "use_resolution_template": False,
                "guardrails": guardrails,
            },
        }
        response = await _vllm_post_json(_vllm_api_url(base_url, "images/generations"), payload)
        return io.NodeOutput(_vllm_decode_image(response))


class Cosmos3TextToVideoAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Cosmos3TextToVideoAPI",
            display_name="Cosmos3 Text to Video (API)",
            category="partner/video/Cosmos",
            description="Generates a video with a local Cosmos3 vLLM-Omni server.",
            inputs=_vllm_generation_inputs(include_video_options=True),
            outputs=[io.Video.Output()],
        )

    @classmethod
    async def execute(cls, prompt, base_url, width, height, num_frames, fps, steps, guidance_scale, flow_shift, seed, generate_sound, guardrails, negative_prompt="", generation_settings=None):
        width, height, num_frames, fps = _apply_generation_settings(generation_settings, width, height, num_frames, fps)
        fields = _vllm_video_fields(prompt, negative_prompt, width, height, num_frames, fps, steps, guidance_scale, flow_shift, seed, generate_sound, guardrails)
        response = await _vllm_generate_video(base_url, fields)
        return io.NodeOutput(InputImpl.VideoFromFile(BytesIO(response)))


class Cosmos3TextToVideoWithSoundAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Cosmos3TextToVideoWithSoundAPI",
            display_name="Cosmos3 Text to Video with Sound (API)",
            category="partner/video/Cosmos",
            description="Generates a video with sound using a local Cosmos3 vLLM-Omni server.",
            inputs=_vllm_generation_inputs(include_video_options=True, include_sound_option=False),
            outputs=[io.Video.Output()],
        )

    @classmethod
    async def execute(cls, prompt, base_url, width, height, num_frames, fps, steps, guidance_scale, flow_shift, seed, guardrails, negative_prompt="", generation_settings=None):
        width, height, num_frames, fps = _apply_generation_settings(generation_settings, width, height, num_frames, fps)
        fields = _vllm_video_fields(prompt, negative_prompt, width, height, num_frames, fps, steps, guidance_scale, flow_shift, seed, True, guardrails)
        response = await _vllm_generate_video(base_url, fields)
        return io.NodeOutput(InputImpl.VideoFromFile(BytesIO(response)))


class Cosmos3ImageToVideoAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Cosmos3ImageToVideoAPI",
            display_name="Cosmos3 Image to Video (API)",
            category="partner/video/Cosmos",
            description="Generates a video from an image with a local Cosmos3 vLLM-Omni server.",
            inputs=_vllm_generation_inputs(include_image=True, include_video_options=True),
            outputs=[io.Video.Output()],
        )

    @classmethod
    async def execute(cls, image, prompt, base_url, width, height, num_frames, fps, steps, guidance_scale, flow_shift, seed, generate_sound, guardrails, negative_prompt="", generation_settings=None):
        width, height, num_frames, fps = _apply_generation_settings(generation_settings, width, height, num_frames, fps)
        fields = _vllm_video_fields(prompt, negative_prompt, width, height, num_frames, fps, steps, guidance_scale, flow_shift, seed, generate_sound, guardrails)
        file = ("input_reference", "input.png", "image/png", _vllm_encode_image(image))
        response = await _vllm_generate_video(base_url, fields, file)
        return io.NodeOutput(InputImpl.VideoFromFile(BytesIO(response)))


class Cosmos3ActionToVideoAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Cosmos3ActionToVideoAPI",
            display_name="Cosmos3 Action to Video (API)",
            category="partner/video/Cosmos",
            description="Generates future observations from a start image and action trajectory.",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("action_path", tooltip="Path to a JSON action trajectory."),
                io.String.Input("prompt", default="", force_input=True, optional=True),
                io.String.Input("negative_prompt", default="", force_input=True, optional=True),
                io.String.Input("base_url", default=DEFAULT_VLLM_BASE_URL),
                io.String.Input("domain_name", default="droid_lerobot"),
                io.String.Input("view_point", default="ego_view", advanced=True),
                io.Int.Input("image_size", default=COSMOS3_ACTION_IMAGE_SIZE, min=16, max=4096, step=16, advanced=True),
                io.Int.Input("fps", default=15, min=1, max=120),
                io.Int.Input("steps", default=30, min=1, max=1000),
                io.Float.Input("guidance_scale", default=1.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("flow_shift", default=10.0, min=0.0, max=100.0, step=0.1),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Boolean.Input("guardrails", default=False),
            ],
            outputs=[io.Video.Output()],
        )

    @classmethod
    async def execute(cls, image, action_path, base_url, domain_name, view_point, image_size, fps, steps, guidance_scale, flow_shift, seed, guardrails, prompt="", negative_prompt=""):
        trajectory = json.loads(Path(action_path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError("action trajectory must be a non-empty JSON array")
        if any(not isinstance(step, list) or not step for step in trajectory):
            raise ValueError("action trajectory must contain non-empty action arrays")
        action_dim = len(trajectory[0])
        if any(len(step) != action_dim for step in trajectory):
            raise ValueError("action trajectory rows must have the same length")
        action_chunk_size = len(trajectory)
        height, width = image.shape[1:3]
        fields = {
            "prompt": prompt.strip() or " ",
            "negative_prompt": negative_prompt,
            "num_frames": action_chunk_size + 1,
            "fps": fps,
            "size": f"{width}x{height}",
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "seed": seed,
            "extra_params": json.dumps({
                "action_mode": "forward_dynamics",
                "domain_name": domain_name,
                "action_chunk_size": action_chunk_size,
                "image_size": image_size,
                "view_point": view_point,
                "action": trajectory,
                "guardrails": guardrails,
            }),
        }
        file = ("input_reference", "input.png", "image/png", _vllm_encode_image(image))
        response = await _vllm_generate_video(base_url, fields, file)
        return io.NodeOutput(InputImpl.VideoFromFile(BytesIO(response)))


class Cosmos3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Cosmos3PromptAPI,
            Cosmos3TextToImageAPI,
            Cosmos3TextToVideoAPI,
            Cosmos3TextToVideoWithSoundAPI,
            Cosmos3ImageToVideoAPI,
            Cosmos3ActionToVideoAPI,
        ]

async def comfy_entrypoint() -> ComfyExtension:
    return Cosmos3Extension()

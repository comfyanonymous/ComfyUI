import asyncio
import json

import pytest
import torch

from comfy_api_nodes import nodes_cosmos3 as cosmos3


@pytest.mark.parametrize(
    "base_url, endpoint, expected",
    [
        ("http://localhost:8000", "images/generations", "http://localhost:8000/v1/images/generations"),
        ("https://127.0.0.1:8000/v1/", "videos/job id", "https://127.0.0.1:8000/v1/videos/job id"),
        ("http://[::1]:8000/api", "videos", "http://[::1]:8000/api/v1/videos"),
    ],
)
def test_vllm_api_url_accepts_loopback_servers(base_url, endpoint, expected):
    assert cosmos3._vllm_api_url(base_url, endpoint) == expected


@pytest.mark.parametrize("base_url", ["http://example.com", "http://192.168.1.10:8000"])
def test_vllm_api_url_rejects_non_loopback_servers(base_url):
    with pytest.raises(ValueError, match="loopback"):
        cosmos3._vllm_api_url(base_url, "videos")


def test_vllm_api_url_rejects_non_http_url():
    with pytest.raises(ValueError, match="HTTP URL"):
        cosmos3._vllm_api_url("ftp://localhost", "videos")


def test_prompt_file_settings_override_generation_inputs(tmp_path):
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text(json.dumps({"text": "a robot", "resolution": {"W": 640, "H": 480}, "fps": 12, "num_frames": 25}))

    output = cosmos3.Cosmos3PromptAPI.execute("file_path", str(prompt_file), "")

    assert json.loads(output.args[0]) == {"text": "a robot", "resolution": {"W": 640, "H": 480}, "fps": 12, "num_frames": 25}
    assert output.args[1] == ""
    assert output.args[2] == {"width": 640, "height": 480, "fps": 12, "num_frames": 25}
    assert cosmos3._apply_generation_settings(output.args[2], 1280, 720, 189, 24) == (640, 480, 25, 12)


def test_text_to_image_sends_expected_request(monkeypatch):
    request = {}

    async def post_json(url, payload):
        request["url"] = url
        request["payload"] = payload
        return {"data": [{"b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"}]}

    monkeypatch.setattr(cosmos3, "_vllm_post_json", post_json)

    output = asyncio.run(cosmos3.Cosmos3TextToImageAPI.execute(
        "a sunny beach", "http://localhost:8000", 1280, 720, 35, 6.0, 10.0, 42, True,
        negative_prompt="rain", generation_settings={"width": 640, "height": 480},
    ))

    assert request == {
        "url": "http://localhost:8000/v1/images/generations",
        "payload": {
            "prompt": "a sunny beach", "negative_prompt": "rain", "size": "640x480", "n": 1,
            "num_inference_steps": 35, "guidance_scale": 6.0, "flow_shift": 10.0, "seed": 42,
            "extra_args": {"use_resolution_template": False, "guardrails": True},
        },
    }
    assert output.args[0].shape == (1, 1, 1, 3)


def test_generate_video_polls_until_completion(monkeypatch):
    requests = []
    statuses = iter([b'{"status":"queued"}', b'{"status":"in_progress"}', b'{"status":"completed"}'])

    async def post_video(url, fields, file):
        requests.append(("post", url, fields, file))
        return {"id": "job/id"}

    async def get(url, accept):
        requests.append(("get", url, accept))
        if url.endswith("/content"):
            return b"video"
        return next(statuses)

    async def sleep(_):
        return None

    monkeypatch.setattr(cosmos3, "_vllm_post_video", post_video)
    monkeypatch.setattr(cosmos3, "_vllm_get", get)
    monkeypatch.setattr(cosmos3.asyncio, "sleep", sleep)

    assert asyncio.run(cosmos3._vllm_generate_video("http://localhost:8000", {"prompt": "test"})) == b"video"
    assert requests == [
        ("post", "http://localhost:8000/v1/videos", {"prompt": "test"}, None),
        ("get", "http://localhost:8000/v1/videos/job%2Fid", "application/json"),
        ("get", "http://localhost:8000/v1/videos/job%2Fid", "application/json"),
        ("get", "http://localhost:8000/v1/videos/job%2Fid", "application/json"),
        ("get", "http://localhost:8000/v1/videos/job%2Fid/content", "video/mp4"),
    ]


def test_action_to_video_uses_image_dimensions_and_trajectory(tmp_path, monkeypatch):
    action_file = tmp_path / "actions.json"
    action_file.write_text("[[1, 2], [3, 4]]")
    request = {}

    async def generate_video(base_url, fields, file):
        request["base_url"] = base_url
        request["fields"] = fields
        request["file"] = file
        return b"video"

    monkeypatch.setattr(cosmos3, "_vllm_generate_video", generate_video)

    output = asyncio.run(cosmos3.Cosmos3ActionToVideoAPI.execute(
        torch.zeros((1, 24, 32, 3)), str(action_file), "http://localhost:8000", "droid_lerobot", "ego_view",
        480, 15, 30, 1.0, 10.0, 7, False, prompt="", negative_prompt="none",
    ))

    assert request["base_url"] == "http://localhost:8000"
    assert request["fields"]["size"] == "32x24"
    assert request["fields"]["num_frames"] == 3
    assert json.loads(request["fields"]["extra_params"])["action"] == [[1, 2], [3, 4]]
    assert request["file"][:3] == ("input_reference", "input.png", "image/png")
    assert output.args[0].get_stream_source().read() == b"video"

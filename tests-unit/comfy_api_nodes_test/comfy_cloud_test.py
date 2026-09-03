import asyncio
from io import BytesIO
from typing import get_args
from unittest.mock import AsyncMock

import aiohttp
import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_api_nodes.apis.comfy_cloud import (
    ComfyCloudGenerateRequest,
    ComfyCloudGenerateResponse,
    ComfyCloudStatusResponse,
    ComfyCloudWorkflow,
    ComfyCloudWorkflowInputs,
)
from comfy_api_nodes import nodes_comfy_cloud
from comfy_api_nodes.util import download_helpers



def _execute_with_defaults(node, prompt, **overrides):
    """Call a node with its own schema defaults, so a test does not have to track
    each workflow's argument list."""
    kwargs = {}
    for spec in node.define_schema().inputs:
        default = getattr(spec, "default", None)
        if default is not None:
            kwargs[spec.id] = default
        elif getattr(spec, "options", None):
            kwargs[spec.id] = spec.options[0]
    kwargs["prompt"] = prompt
    kwargs.update(overrides)
    return asyncio.run(node.execute(**kwargs))

@pytest.mark.parametrize(
    ("node", "workflow", "returns_video", "requires_image", "controls"),
    [
        (
            nodes_comfy_cloud.ComfyCloudZImageTurboNode, "z-image-turbo/text-to-image", False, False,
            {"width": 1024, "height": 1024, "model": "z_image_turbo_bf16", "steps": 8, "shift": 3.0},
        ),
        (
            nodes_comfy_cloud.ComfyCloudMiniMaxH3TextSoundNode, "minimax-h3/text-to-video", True, False,
            {"aspect_ratio": "16:9", "duration_seconds": 5.0, "resolution": "480p", "steps": 20},
        ),
    ],
)
def test_workflow_submission_polling_and_download(
    monkeypatch, node, workflow, returns_video, requires_image, controls
):
    sync = AsyncMock(
        return_value=ComfyCloudGenerateResponse(
            task_id="task-1",
            status="queued",
            polling_url="/proxy/comfy-cloud/workflow/tasks/task-1",
            cancel_url="/proxy/comfy-cloud/workflow/tasks/task-1/cancel",
        )
    )
    poll = AsyncMock(
        return_value=ComfyCloudStatusResponse(
            task_id="task-1",
            status="completed",
            progress=100,
            output_url="/proxy/comfy-cloud/results/task-1/output",
        )
    )
    upload = AsyncMock(return_value="https://example.com/input.png")
    image_download = AsyncMock(return_value="image-output")
    video_download = AsyncMock(return_value="video-output")
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)
    monkeypatch.setattr(nodes_comfy_cloud, "poll_op", poll)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_image_tensor", image_download)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_video_output", video_download)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)

    output = _execute_with_defaults(node, "A tiny fennec fox")

    endpoint = sync.call_args.args[1]
    request = sync.call_args.kwargs["data"]
    assert endpoint.path == "/proxy/comfy-cloud/workflow/generate"
    assert endpoint.method == "POST"
    assert request == ComfyCloudGenerateRequest(
        workflow=workflow,
        inputs=ComfyCloudWorkflowInputs(
            prompt="A tiny fennec fox",
            image_url="https://example.com/input.png" if requires_image else None,
            seed=42,
            **controls,
        ),
    )
    assert upload.await_count == int(requires_image)

    poll_endpoint = poll.call_args.args[1]
    cancel_endpoint = poll.call_args.kwargs["cancel_endpoint"]
    assert poll_endpoint.path == "/proxy/comfy-cloud/workflow/tasks/task-1"
    assert cancel_endpoint.path == "/proxy/comfy-cloud/workflow/tasks/task-1/cancel"
    assert cancel_endpoint.method == "POST"
    assert output[0] == ("video-output" if returns_video else "image-output")


def test_contract_omits_optional_status_fields():
    request = ComfyCloudGenerateRequest(
        workflow="z-image-turbo/text-to-image",
        inputs=ComfyCloudWorkflowInputs(prompt="A lighthouse"),
    )
    status = ComfyCloudStatusResponse(task_id="task-1", status="queued")

    assert request.model_dump(exclude_none=True) == {
        "workflow": "z-image-turbo/text-to-image",
        "inputs": {"prompt": "A lighthouse"},
    }
    assert status.model_dump(exclude_none=True) == {"task_id": "task-1", "status": "queued"}


def test_status_progress_is_clamped_for_display():
    assert nodes_comfy_cloud._progress(ComfyCloudStatusResponse(task_id="task-1", status="running", progress=100.5)) == 100
    assert nodes_comfy_cloud._progress(ComfyCloudStatusResponse(task_id="task-1", status="running", progress=-1)) == 0


def test_poll_failure_cancels_submitted_task(monkeypatch):
    poll = AsyncMock(side_effect=ValueError("invalid status response"))
    cancel = AsyncMock(return_value={"status": "cancellation_requested"})
    monkeypatch.setattr(nodes_comfy_cloud, "poll_op", poll)
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op_raw", cancel)

    with pytest.raises(ValueError, match="invalid status response"):
        asyncio.run(nodes_comfy_cloud._poll_task(nodes_comfy_cloud.ComfyCloudZImageTurboNode, "task/1"))

    cancel.assert_awaited_once()
    assert cancel.call_args.args[1].path == "/proxy/comfy-cloud/workflow/tasks/task%2F1/cancel"
    assert cancel.call_args.kwargs["max_retries"] == 0


@pytest.mark.parametrize("response_model", [ComfyCloudGenerateResponse, ComfyCloudStatusResponse])
@pytest.mark.parametrize("task_id", ["", "   "])
def test_contract_rejects_empty_task_ids(response_model, task_id):
    values = {"task_id": task_id, "status": "queued"}
    if response_model is ComfyCloudGenerateResponse:
        values.update(polling_url="/poll", cancel_url="/cancel")

    with pytest.raises(ValueError, match="task_id"):
        response_model(**values)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/output.png",
        "http://127.0.0.1/output.png",
        "//169.254.169.254/latest/meta-data",
        "/unrelated/path/output.png",
        "https://user@example.com/output.png",
        "/proxy/comfy-cloud/../../v1/users/me",
        "/proxy/comfy-cloud/%2e%2e/%2e%2e/v1/users/me",
        "https://127.0.0.1/output.png",
        "https://169.254.169.254/latest/meta-data",
        "https://attacker.example/output.png",
    ],
)
def test_cloud_workflows_reject_untrusted_output_urls(monkeypatch, url):
    sync = AsyncMock(
        return_value=ComfyCloudGenerateResponse(
            task_id="task-1",
            status="queued",
            polling_url="/poll",
            cancel_url="/cancel",
        )
    )
    poll = AsyncMock(
        return_value=ComfyCloudStatusResponse(task_id="task-1", status="completed", output_url=url)
    )
    download = AsyncMock()
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)
    monkeypatch.setattr(nodes_comfy_cloud, "poll_op", poll)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_image_tensor", download)

    with pytest.raises(RuntimeError, match="invalid output URL"):
        asyncio.run(nodes_comfy_cloud.ComfyCloudZImageTurboNode.execute("prompt"))
    download.assert_not_awaited()


def test_cloud_workflows_accept_signed_https_output_urls(monkeypatch):
    sync = AsyncMock(
        return_value=ComfyCloudGenerateResponse(
            task_id="task-1",
            status="queued",
            polling_url="/poll",
            cancel_url="/cancel",
        )
    )
    poll = AsyncMock(
        return_value=ComfyCloudStatusResponse(
            task_id="task-1",
            status="completed",
            output_url="https://storage.googleapis.com/comfy-cloud-assets/output.png?signature=example",
        )
    )
    download = AsyncMock(return_value="image-output")
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)
    monkeypatch.setattr(nodes_comfy_cloud, "poll_op", poll)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_image_tensor", download)

    output = asyncio.run(nodes_comfy_cloud.ComfyCloudZImageTurboNode.execute("prompt"))

    assert output[0] == "image-output"
    download.assert_awaited_once_with(
        "https://storage.googleapis.com/comfy-cloud-assets/output.png?signature=example",
        timeout=nodes_comfy_cloud._OUTPUT_DOWNLOAD_TIMEOUT,
        cls=nodes_comfy_cloud.ComfyCloudZImageTurboNode,
        allow_redirects=False,
    )


@pytest.mark.parametrize(
    "node",
    [
        nodes_comfy_cloud.ComfyCloudZImageTurboNode,
        nodes_comfy_cloud.ComfyCloudFlux2TextToImageNode,
        nodes_comfy_cloud.ComfyCloudMiniMaxH3TextSoundNode,
    ],
)
def test_capability_nodes_reject_oversized_prompts(monkeypatch, node):
    sync = AsyncMock()
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)

    with pytest.raises(Exception, match="4096"):
        asyncio.run(node.execute("x" * 4097, object()))
    sync.assert_not_awaited()


def test_capability_nodes_strip_prompts_before_submission(monkeypatch):
    values = nodes_comfy_cloud._validate_node_inputs(
        nodes_comfy_cloud.ComfyCloudZImageTurboNode, {"prompt": "  prompt  "}
    )

    assert values["prompt"] == "prompt"


def test_task_routes_ignore_response_urls_and_errors_hide_task_token(monkeypatch):
    sync = AsyncMock(
        return_value=ComfyCloudGenerateResponse(
            task_id="secret/task-token",
            status="queued",
            polling_url="https://attacker.example/poll",
            cancel_url="https://attacker.example/cancel",
        )
    )
    poll = AsyncMock(
        return_value=ComfyCloudStatusResponse(
            task_id="secret/task-token",
            status="completed",
            error="provider details with secret/task-token",
        )
    )
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)
    monkeypatch.setattr(nodes_comfy_cloud, "poll_op", poll)

    with pytest.raises(RuntimeError) as error:
        _execute_with_defaults(nodes_comfy_cloud.ComfyCloudMiniMaxH3TextSoundNode, "A prompt")

    assert poll.call_args.args[1].path == "/proxy/comfy-cloud/workflow/tasks/secret%2Ftask-token"
    assert poll.call_args.kwargs["cancel_endpoint"].path == "/proxy/comfy-cloud/workflow/tasks/secret%2Ftask-token/cancel"
    assert "task-token" not in str(error.value)
    assert "provider details" not in str(error.value)


CONTROLLED_IMAGE_NODES = [
    (
        nodes_comfy_cloud.ComfyCloudZImageTurboNode,
        "z-image-turbo/text-to-image",
        ["prompt", "aspect_ratio", "seed", "model", "steps", "shift"],
        {
            "prompt": "A glass forest", "aspect_ratio": "16:9", "seed": 9, "model": "z_image_turbo_nvfp4",
            "steps": 10, "shift": 2.5,
        },
        {
            "prompt": "A glass forest", "width": 1344, "height": 768, "seed": 9, "model": "z_image_turbo_nvfp4",
            "steps": 10, "shift": 2.5,
        },
    ),
    (
        nodes_comfy_cloud.ComfyCloudFlux2TextToImageNode,
        "flux-2/text-to-image",
        [
            "prompt", "aspect_ratio", "turbo", "seed", "model", "lora", "steps", "turbo_steps",
            "turbo_strength", "guidance",
        ],
        {
            "prompt": "A glass forest", "aspect_ratio": "1:1", "turbo": False, "seed": 9,
            "model": "flux2-dev", "lora": "flux2-lenovo_ultrareal", "steps": 24, "turbo_steps": 6,
            "turbo_strength": 0.8, "guidance": 4.5,
        },
        {
            "prompt": "A glass forest", "width": 1024, "height": 1024, "turbo": False, "seed": 9,
            "model": "flux2-dev", "lora": "flux2-lenovo_ultrareal", "steps": 24, "turbo_steps": 6,
            "turbo_strength": 0.8, "guidance": 4.5,
        },
    ),
]


@pytest.mark.parametrize(("node", "workflow", "input_names", "arguments", "expected_inputs"), CONTROLLED_IMAGE_NODES)
def test_controlled_image_node_schema_and_request_mapping(
    monkeypatch, node, workflow, input_names, arguments, expected_inputs
):
    sync = AsyncMock(
        return_value=ComfyCloudGenerateResponse(
            task_id="task-2",
            status="queued",
            polling_url="/tasks/task-2",
            cancel_url="/tasks/task-2/cancel",
        )
    )
    poll = AsyncMock(
        return_value=ComfyCloudStatusResponse(
            task_id="task-2",
            status="completed",
            output_url="/proxy/comfy-cloud/results/task-2/image.png",
        )
    )
    upload = AsyncMock(return_value="/uploads/input.png")
    download = AsyncMock(return_value="image-output")
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)
    monkeypatch.setattr(nodes_comfy_cloud, "poll_op", poll)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_image_tensor", download)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)

    schema = node.define_schema()
    assert schema.node_id == node.node_id
    assert schema.display_name == node.display_name
    assert schema.is_api_node is True
    assert [input.id for input in schema.inputs] == input_names
    assert len(schema.outputs) == 1
    assert schema.outputs[0].get_io_type() == "IMAGE"

    output = asyncio.run(node.execute(**arguments))
    request = sync.call_args.kwargs["data"]

    assert request.workflow == workflow
    assert request.inputs.model_dump(exclude_none=True) == expected_inputs
    assert "asset_id" not in request.model_dump_json()
    assert '"id"' not in request.model_dump_json()
    assert upload.await_count == int("image" in arguments)
    if "image" in arguments:
        assert upload.call_args.kwargs == {"total_pixels": None}
    download.assert_awaited_once_with(
        "/proxy/comfy-cloud/results/task-2/image.png",
        timeout=30 * 60,
        cls=node,
        allow_redirects=False,
    )
    assert output[0] == "image-output"


def test_controlled_image_nodes_are_declared_and_registered():
    workflows = {workflow for _, workflow, _, _, _ in CONTROLLED_IMAGE_NODES}
    registered = set(asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list()))

    assert workflows <= set(get_args(ComfyCloudWorkflow))
    assert {node for node, _, _, _, _ in CONTROLLED_IMAGE_NODES} <= registered


# The controls a first-time user is shown before expanding anything. Everything
# outside this set has to be advanced, or the node opens on a wall of dials.
PLAIN_CONTROLS = {
    "prompt", "instruction", "image", "audio", "first_frame", "last_frame",
    "aspect_ratio", "duration_seconds", "seed", "scale", "quality_mode",
    "prompt_enhance", "enhance_prompt", "turbo", "rendering_speed",
}


def test_tuning_controls_are_hidden_behind_the_advanced_flag():
    for node in asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list()):
        plain = [
            input_spec.id
            for input_spec in node.define_schema().inputs
            if not getattr(input_spec, "advanced", None)
        ]
        assert set(plain) <= PLAIN_CONTROLS, f"{node.__name__} shows {sorted(set(plain) - PLAIN_CONTROLS)}"
        assert len(plain) <= 6, f"{node.__name__} opens with {len(plain)} controls"


def test_pointer_node_pickers_name_the_trade_off_rather_than_the_model():
    """A default/* id is a pointer: the model behind it is re-pointed over time and
    the id does not change. A saved graph stores the KEY, so a key naming a model
    would stop resolving the day the pointer moved."""
    tiers = {"fast", "balanced", "quality"}
    for node in asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list()):
        if not getattr(node, "workflow", "").startswith("default/"):
            continue
        for options in (node.model_options, node.lora_options):
            assert set(options) <= tiers, f"{node.__name__}: {sorted(set(options) - tiers)}"


def test_every_seed_starts_at_42_and_advances_after_a_run():
    for node in asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list()):
        seed = next((i for i in node.define_schema().inputs if i.id == "seed"), None)
        assert seed is not None, f"{node.__name__} has no seed"
        assert seed.default == 42, f"{node.__name__} seeds at {seed.default}"
        assert seed.control_after_generate is True


def test_aspect_ratios_resolve_to_sizes_the_latents_accept():
    """The graphs that take a width and a height get them from this table, so every
    side has to sit on the 16-pixel grid at both render scales, not only at 1x."""
    for ratio in nodes_comfy_cloud._ASPECT_RATIOS:
        for scale in (1.0, 1.25):
            width, height = nodes_comfy_cloud._dimensions(ratio, scale)
            assert width % 16 == 0 and height % 16 == 0, (ratio, scale)
            assert 256 <= width <= 2048 and 256 <= height <= 2048, (ratio, scale)
    assert nodes_comfy_cloud._dimensions("1:1") == (1024, 1024)
    assert nodes_comfy_cloud._dimensions("1:1", 1.25) == (1280, 1280)


def test_cloud_workflow_controls_have_connection_sockets():
    nodes = asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list())

    for node in nodes:
        for input_spec in node.define_schema().inputs:
            if isinstance(input_spec, nodes_comfy_cloud.IO.WidgetInput):
                assert input_spec.socketless is False, f"{node.__name__}.{input_spec.id}"


def test_cloud_workflow_schemas_have_descriptions():
    nodes = asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list())

    for node in nodes:
        assert node.define_schema().description.strip(), node.__name__


def test_cloud_workflow_schemas_share_exact_estimated_rate_metadata():
    nodes = asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list())

    assert nodes_comfy_cloud.COMFY_CLOUD_GPU_SECOND_USD == 0.00185
    assert nodes_comfy_cloud.COMFY_CLOUD_CREDITS_PER_USD == 211
    assert nodes_comfy_cloud.COMFY_CLOUD_GPU_SECOND_CREDITS == pytest.approx(0.39035)
    for node in nodes:
        schema = node.define_schema()
        badge = schema.price_badge.as_dict(schema.inputs)
        assert badge["expr"] == (
            '{"type":"usd","usd":0.001850,"format":{"suffix":"/GPU-second","approximate":true}}'
        )
        assert schema.description.endswith(
            "Runs on a Comfy Cloud GPU, billed by how long it runs at "
            "$0.00185/GPU-second (0.39 credits). Paid in credits, no Cloud "
            "subscription required."
        ), node.__name__


def test_every_node_describes_what_it_actually_does():
    """These descriptions used to be three generic bodies shared across sixteen
    nodes, which left the tooltip saying nothing about which model runs or why
    you would pick this node over the one beside it."""
    nodes = asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list())
    summaries = {}

    for node in nodes:
        schema = node.define_schema()
        summary = schema.description.split(" Runs on a Comfy Cloud GPU")[0]
        assert len(summary) > 60, f"{node.__name__} says too little: {summary!r}"
        summaries.setdefault(summary, []).append(node.__name__)

    shared = {s: n for s, n in summaries.items() if len(n) > 1}
    # The four capability nodes deliberately share wording per output type.
    assert all(len(n) <= 2 for n in shared.values()), shared


def test_all_linkable_widget_constraints_are_validated():
    nodes = asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list())

    for node in nodes:
        for input_spec in node.define_schema().inputs:
            if not isinstance(input_spec, nodes_comfy_cloud.IO.WidgetInput):
                continue
            io_type = input_spec.get_io_type()
            if io_type == "COMBO":
                invalid = "not-an-option"
            elif io_type == "BOOLEAN":
                invalid = "true"
            elif io_type == "INT":
                invalid = (input_spec.min - 1) if input_spec.min is not None else 1.5
            elif io_type == "FLOAT":
                invalid = float("nan")
            elif io_type == "STRING" and nodes_comfy_cloud._TEXT_LIMITS.get(input_spec.id, (0,))[0]:
                invalid = "   "
            else:
                continue
            with pytest.raises((ValueError, Exception), match=input_spec.id):
                nodes_comfy_cloud._validate_node_inputs(node, {input_spec.id: invalid})


def test_text_inputs_name_the_field_when_a_linked_value_is_not_a_string():
    """_with_input_sockets clears socketless on every widget, so a graph can link an INT
    output into a text input. That must name the field, not raise AttributeError."""
    with pytest.raises(ValueError, match="prompt"):
        nodes_comfy_cloud._validate_node_inputs(
            nodes_comfy_cloud.ComfyCloudZImageTurboNode, {"prompt": 7}
        )


def test_upload_inputs_have_decoded_resource_limits():
    oversized_image = torch.empty((1, 8193, 1, 3), device="meta")
    # The only case that reaches the pixel-count branch rather than the dimension one.
    oversized_pixels = torch.empty((1, 8000, 5000, 3), device="meta")
    oversized_audio = {
        "waveform": torch.empty((1, 2, nodes_comfy_cloud._MAX_DECODED_AUDIO_BYTES // 8 + 1), device="meta"),
        "sample_rate": 48000,
    }

    with pytest.raises(ValueError, match="32-megapixel"):
        nodes_comfy_cloud._validate_image_upload(oversized_image)
    with pytest.raises(ValueError, match="32-megapixel"):
        nodes_comfy_cloud._validate_image_upload(oversized_pixels)
    with pytest.raises(ValueError, match="256 MiB"):
        nodes_comfy_cloud._validate_audio_upload(oversized_audio)


def _stub_download_session(monkeypatch, responses):
    """Point download_url_to_bytesio at a scripted list of responses, one per attempt.

    Returns the list that records the `allow_redirects` each attempt asked aiohttp for.
    """
    seen_allow_redirects = []

    class Session:
        def __init__(self, timeout):
            pass

        async def get(self, url, headers, allow_redirects=True):
            seen_allow_redirects.append(allow_redirects)
            return responses.pop(0)

        async def close(self):
            pass

    monkeypatch.setattr(download_helpers.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(download_helpers, "sleep_with_interrupt", AsyncMock())
    return seen_allow_redirects


class _StubContent:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.finished = False

    async def read(self, size):
        chunk = next(self.chunks)
        if isinstance(chunk, Exception):
            raise chunk
        if not chunk:
            self.finished = True
        return chunk

    def at_eof(self):
        return self.finished


class _StubResponse:
    headers = {}
    content_length = None

    def __init__(self, chunks, status=200):
        self.content = _StubContent(chunks)
        self.status = status

    async def json(self):
        return {}

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def test_in_memory_download_empties_its_sink_before_retrying(monkeypatch):
    """A retry re-reads the body from byte zero. Without the reset the caller gets
    the partial first attempt concatenated with the whole second one."""
    _stub_download_session(
        monkeypatch,
        [
            _StubResponse([b"partial", aiohttp.ClientPayloadError("retry")]),
            _StubResponse([b"final", b""]),
        ],
    )
    destination = BytesIO()

    asyncio.run(download_helpers.download_url_to_bytesio("https://example.com/result", destination))

    assert destination.read() == b"final"


def test_redirects_are_refused_rather_than_followed_when_disallowed(monkeypatch):
    """Comfy Cloud vets the output URL's bucket before downloading, so a redirect has to
    fail rather than quietly fetch the body from somewhere that was never vetted."""
    seen = _stub_download_session(monkeypatch, [_StubResponse([], status=302)])

    with pytest.raises(Exception, match="HTTP 302"):
        asyncio.run(
            download_helpers.download_url_to_bytesio(
                "https://example.com/result", BytesIO(), allow_redirects=False, max_retries=0
            )
        )

    assert seen == [False]


def test_extension_registers_exactly_the_shipped_set():
    """The shipped surface is deliberate: four capability nodes whose model the backend
    chooses, plus the named workflows that expose control a generic shape cannot."""
    capability_nodes = {
        nodes_comfy_cloud.ComfyCloudZImageTurboNode,
        nodes_comfy_cloud.ComfyCloudFlux2TextToImageNode,
        nodes_comfy_cloud.ComfyCloudMiniMaxH3TextSoundNode,
    }
    named_nodes = {
        nodes_comfy_cloud.ComfyCloudMiniMaxH3TextSoundNode,
        nodes_comfy_cloud.ComfyCloudZImageTurboNode,
        nodes_comfy_cloud.ComfyCloudFlux2TextToImageNode,
    }
    registered = set(asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list()))

    assert registered == capability_nodes | named_nodes


def test_every_node_is_named_for_its_provider():
    """A user searching "flux" sees our paid cloud node beside the free local one.
    The provider prefix is the only thing on the name that tells them apart."""
    for node in asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list()):
        name = node.define_schema().display_name
        assert name.startswith("Comfy Cloud "), f"{node.__name__} is named {name!r}"


@pytest.mark.parametrize(
    "bucket",
    ["comfy-cloud", "comfy-cloud-assets-evil", "example", "comfy-cloud-assets.evil.com",
     "partner-nodes-assets-evil", "partner-nodes",
     # Advertises an allowed bucket in the first segment and resolves to another one.
     "comfy-cloud-assets/../other-bucket", "comfy-cloud-assets/%2e%2e/other-bucket"],
)
def test_output_urls_outside_the_backend_bucket_allowlist_are_rejected(bucket):
    """The output URL is backend-supplied, so this is defence in depth -- but the host check
    alone would accept any bucket on storage.googleapis.com. Pinned to the same set the Go
    side enforces (comfyCloudOutputBuckets)."""
    with pytest.raises(RuntimeError):
        nodes_comfy_cloud._validated_output_url(
            f"https://storage.googleapis.com/{bucket}/output.png?signature=example"
        )


@pytest.mark.parametrize(
    "bucket", ["comfy-cloud-assets", "comfy-cloud-assets-stg", "comfy-cloud-assets-test"]
)
def test_output_urls_in_the_backend_bucket_allowlist_are_accepted(bucket):
    url = f"https://storage.googleapis.com/{bucket}/output.png?signature=example"
    assert nodes_comfy_cloud._validated_output_url(url) == url


def test_poll_budget_covers_the_platform_run_ceiling():
    """The poll budget must sit just above the platform's own max_runtime, so a job that is
    still legitimately running is ended by its own timeout rather than abandoned by the node."""
    budget = nodes_comfy_cloud._POLL_MAX_ATTEMPTS * nodes_comfy_cloud._POLL_INTERVAL_SECONDS
    assert budget > nodes_comfy_cloud._RUN_TIMEOUT_SECONDS
    assert budget - nodes_comfy_cloud._RUN_TIMEOUT_SECONDS <= 300


def test_output_buckets_match_the_server_allowlist():
    """Pins the bucket set so a change here is deliberate and paired.

    The same list lives in cloud at services/comfy-api/config/config.go
    (comfyCloudOutputBuckets). Widening one side only is not a benign skew: the
    job runs on the GPU, the caller is billed per GPU-second, and the output is
    then rejected on the user's machine. The node half also ships on the ComfyUI
    release train, so a mismatch persists until the next release.
    """
    assert nodes_comfy_cloud._OUTPUT_BUCKETS == frozenset(
        {
            "comfy-cloud-assets",
            "comfy-cloud-assets-stg",
            "comfy-cloud-assets-test",
            "partner-nodes-assets",
            "partner-nodes-assets-staging",
        }
    )


def test_disabled_provider_message_reaches_the_user_verbatim():
    """comfy-api's error envelope is flat, and the message is what a user reads.

    Before this, _friendly_http_message only unwrapped a NESTED error object, so
    a flat {"error": code, "message": text} body fell through to the raw-JSON
    branch and the user was shown the whole payload.
    """
    from comfy_api_nodes.util import client

    body = {
        "error": "comfy_cloud_provider_disabled",
        "message": "Comfy Cloud is currently unavailable. Please try again later.",
    }
    assert client._friendly_http_message(503, body) == body["message"]


def test_disabled_provider_is_not_retried():
    """A switched-off provider is a terminal answer, not a transient blip.

    503 is in _RETRY_STATUS, so without this the user waits through three
    backoff rounds only to be told the same thing.
    """
    from comfy_api_nodes.util import client

    assert client._is_terminal_service_refusal(
        {"error": "comfy_cloud_provider_disabled", "message": "x"}
    )
    # A genuine transient 503 carries no such code and must still be retried.
    assert not client._is_terminal_service_refusal({"error": "upstream_timeout"})
    assert not client._is_terminal_service_refusal({})
    assert not client._is_terminal_service_refusal("bad gateway")


@pytest.mark.parametrize(
    ("kind", "io_type"),
    [("image", "IMAGE"), ("video", "VIDEO"), ("audio", "AUDIO"), ("model3d", "FILE_3D_GLB")],
)
def test_every_output_kind_declares_its_io_type(kind, io_type):
    output, _ = nodes_comfy_cloud._OUTPUT_KINDS[kind]
    assert output().get_io_type() == io_type


@pytest.mark.parametrize("kind", ["image", "video", "audio", "model3d"])
def test_every_output_kind_refuses_a_redirect_off_the_vetted_url(monkeypatch, kind):
    """The bucket pin decides where the bytes may come from. Following a redirect
    would hand that decision back to the server we just checked."""
    seen = {}

    async def fake(url, *args, **kwargs):
        seen.update(kwargs)
        return "output"

    async def fake_bytesio(url, dest, *args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_image_tensor", fake)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_video_output", fake)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_file_3d", fake)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_bytesio", fake_bytesio)
    monkeypatch.setattr(nodes_comfy_cloud, "audio_bytes_to_audio_input", lambda data: "audio")
    monkeypatch.setattr(
        nodes_comfy_cloud, "_submit_workflow",
        AsyncMock(return_value="https://storage.googleapis.com/comfy-cloud-assets/o.bin?X-Goog-Signature=x"),
    )

    _, run = nodes_comfy_cloud._OUTPUT_KINDS[kind]
    asyncio.run(run(nodes_comfy_cloud.ComfyCloudZImageTurboNode, "z-image-turbo/text-to-image", None))

    assert seen["allow_redirects"] is False

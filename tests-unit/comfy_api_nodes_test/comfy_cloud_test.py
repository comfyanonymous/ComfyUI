import asyncio
from io import BytesIO
from typing import get_args
from unittest.mock import AsyncMock, Mock

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
from comfy_api_nodes.util import conversions, download_helpers


@pytest.mark.parametrize(
    ("node", "workflow", "returns_video", "requires_image"),
    [
        (nodes_comfy_cloud.ComfyCloudTextToImageNode, "text-to-image", False, False),
        (nodes_comfy_cloud.ComfyCloudTextToVideoNode, "text-to-video", True, False),
        (nodes_comfy_cloud.ComfyCloudImageToVideoNode, "image-to-video", True, True),
        (nodes_comfy_cloud.ComfyCloudImageEditNode, "image-edit", False, True),
    ],
)
def test_workflow_submission_polling_and_download(monkeypatch, node, workflow, returns_video, requires_image):
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

    image = object() if requires_image else None
    output = asyncio.run(node.execute("A tiny fennec fox", image))

    endpoint = sync.call_args.args[1]
    request = sync.call_args.kwargs["data"]
    assert endpoint.path == "/proxy/comfy-cloud/workflow/generate"
    assert endpoint.method == "POST"
    assert request == ComfyCloudGenerateRequest(
        workflow=workflow,
        inputs=ComfyCloudWorkflowInputs(
            prompt="A tiny fennec fox",
            image_url="https://example.com/input.png" if requires_image else None,
        ),
    )
    assert upload.await_count == int(requires_image)

    poll_endpoint = poll.call_args.args[1]
    cancel_endpoint = poll.call_args.kwargs["cancel_endpoint"]
    assert poll_endpoint.path == "/proxy/comfy-cloud/workflow/tasks/task-1"
    assert cancel_endpoint.path == "/proxy/comfy-cloud/workflow/tasks/task-1/cancel"
    assert cancel_endpoint.method == "POST"
    assert output[0] == ("video-output" if returns_video else "image-output")


@pytest.mark.parametrize(
    "node",
    [nodes_comfy_cloud.ComfyCloudImageToVideoNode, nodes_comfy_cloud.ComfyCloudImageEditNode],
)
def test_image_workflows_reject_batches(monkeypatch, node):
    upload = AsyncMock()
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 2)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", upload)

    with pytest.raises(ValueError, match="Exactly one input image"):
        asyncio.run(node.execute("Animate this", object()))
    upload.assert_not_awaited()


def test_contract_omits_optional_status_fields():
    request = ComfyCloudGenerateRequest(
        workflow="text-to-image",
        inputs=ComfyCloudWorkflowInputs(prompt="A lighthouse"),
    )
    status = ComfyCloudStatusResponse(task_id="task-1", status="queued")

    assert request.model_dump(exclude_none=True) == {
        "workflow": "text-to-image",
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
        asyncio.run(nodes_comfy_cloud._poll_task(nodes_comfy_cloud.ComfyCloudTextToImageNode, "task/1"))

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
        asyncio.run(nodes_comfy_cloud.ComfyCloudTextToImageNode.execute("prompt"))
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
            output_url="https://storage.googleapis.com/comfy-cloud/output.png?signature=example",
        )
    )
    download = AsyncMock(return_value="image-output")
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)
    monkeypatch.setattr(nodes_comfy_cloud, "poll_op", poll)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_image_tensor", download)

    output = asyncio.run(nodes_comfy_cloud.ComfyCloudTextToImageNode.execute("prompt"))

    assert output[0] == "image-output"
    download.assert_awaited_once_with(
        "https://storage.googleapis.com/comfy-cloud/output.png?signature=example",
        timeout=nodes_comfy_cloud._OUTPUT_DOWNLOAD_TIMEOUT,
        cls=nodes_comfy_cloud.ComfyCloudTextToImageNode,
        allow_redirects=False,
    )


@pytest.mark.parametrize(
    "node",
    [
        nodes_comfy_cloud.ComfyCloudTextToImageNode,
        nodes_comfy_cloud.ComfyCloudTextToVideoNode,
        nodes_comfy_cloud.ComfyCloudImageToVideoNode,
        nodes_comfy_cloud.ComfyCloudImageEditNode,
    ],
)
def test_legacy_nodes_reject_oversized_prompts(monkeypatch, node):
    sync = AsyncMock()
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)

    with pytest.raises(Exception, match="4096"):
        asyncio.run(node.execute("x" * 4097, object()))
    sync.assert_not_awaited()


def test_legacy_nodes_strip_prompts_before_submission(monkeypatch):
    run = AsyncMock(return_value=("output",))
    monkeypatch.setattr(nodes_comfy_cloud.ComfyCloudTextToImageNode, "_run", run)

    asyncio.run(nodes_comfy_cloud.ComfyCloudTextToImageNode.execute("  prompt  "))

    assert run.call_args.args[0].prompt == "prompt"


def test_poc_nodes_strip_prompt_fields_before_submission(monkeypatch):
    run = AsyncMock(return_value=("output",))
    monkeypatch.setattr(nodes_comfy_cloud.ComfyCloudMageFlowImageNode, "_run", run)

    asyncio.run(
        nodes_comfy_cloud.ComfyCloudMageFlowImageNode.execute(
            "  prompt  ", "  avoid this  ", "1:1", 0
        )
    )

    inputs = run.call_args.args[0]
    assert inputs.prompt == "prompt"
    assert inputs.negative_prompt == "avoid this"


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
        asyncio.run(nodes_comfy_cloud.ComfyCloudTextToVideoNode.execute("A prompt"))

    assert poll.call_args.args[1].path == "/proxy/comfy-cloud/workflow/tasks/secret%2Ftask-token"
    assert poll.call_args.kwargs["cancel_endpoint"].path == "/proxy/comfy-cloud/workflow/tasks/secret%2Ftask-token/cancel"
    assert "task-token" not in str(error.value)
    assert "provider details" not in str(error.value)


IMAGE_POC_NODES = [
    (
        nodes_comfy_cloud.ComfyCloudIdeogram4DesignNode,
        "image.ideogram-4-design.v1",
        ["prompt", "aspect_ratio", "quality_mode", "seed"],
        {
            "prompt": "A geometric fox logo",
            "aspect_ratio": "21:9",
            "quality_mode": "fast",
            "seed": 11,
        },
    ),
    (
        nodes_comfy_cloud.ComfyCloudKrea2CreativeImageNode,
        "image.krea-2-creative-image.v1",
        ["prompt", "prompt_enhance", "aspect_ratio", "seed"],
        {"prompt": "A glass forest", "prompt_enhance": False, "aspect_ratio": "16:9", "seed": 12},
    ),
    (
        nodes_comfy_cloud.ComfyCloudMageFlowImageNode,
        "image.mage-flow-image.v1",
        ["prompt", "negative_prompt", "aspect_ratio", "seed"],
        {"prompt": "A moonlit lake", "negative_prompt": "fog", "aspect_ratio": "3:2", "seed": 13},
    ),
    (
        nodes_comfy_cloud.ComfyCloudFlux2ReferenceEditNode,
        "image.flux-2-reference-edit.v1",
        ["image", "instruction", "guidance", "quality_mode", "seed"],
        {"image": object(), "instruction": "Make it winter", "guidance": 5.5, "quality_mode": "fast", "seed": 14},
    ),
    (
        nodes_comfy_cloud.ComfyCloudQwenImageEdit2511Node,
        "image.qwen-image-edit-2511.v1",
        ["image", "instruction", "quality_mode", "seed"],
        {"image": object(), "instruction": "Remove the sign", "quality_mode": "fast", "seed": 15},
    ),
    (
        nodes_comfy_cloud.ComfyCloudSeedVR2ImageUpscaleNode,
        "image.seedvr2-image-upscale.v1",
        ["image", "scale"],
        {"image": object(), "scale": "2x"},
    ),
]


@pytest.mark.parametrize(("node", "workflow", "input_names", "arguments"), IMAGE_POC_NODES)
def test_image_poc_node_schema_and_request_mapping(monkeypatch, node, workflow, input_names, arguments):
    sync = AsyncMock(
        return_value=ComfyCloudGenerateResponse(
            task_id="task-poc",
            status="queued",
            polling_url="/tasks/task-poc",
            cancel_url="/tasks/task-poc/cancel",
        )
    )
    poll = AsyncMock(
        return_value=ComfyCloudStatusResponse(
            task_id="task-poc",
            status="completed",
            output_url="/proxy/comfy-cloud/results/task-poc/image.png",
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
    expected_inputs = {key: value for key, value in arguments.items() if key != "image"}
    if "image" in arguments:
        expected_inputs["assets"] = {"image": {"type": "IMAGE", "url": "/uploads/input.png"}}

    assert request.workflow == workflow
    assert request.inputs.model_dump(exclude_none=True) == expected_inputs
    assert "asset_id" not in request.model_dump_json()
    assert '"id"' not in request.model_dump_json()
    assert upload.await_count == int("image" in arguments)
    if "image" in arguments:
        assert upload.call_args.kwargs == {"total_pixels": None}
    download.assert_awaited_once_with(
        "/proxy/comfy-cloud/results/task-poc/image.png",
        timeout=30 * 60,
        cls=node,
        allow_redirects=False,
    )
    assert output[0] == "image-output"


def test_image_poc_schema_defaults_ranges_and_enums():
    schemas = {
        node.workflow: {input.id: input for input in node.define_schema().inputs}
        for node, _, _, _ in IMAGE_POC_NODES
    }
    aspect_ratios = ["1:1", "3:4", "2:3", "3:2", "4:3", "16:9", "9:16", "21:9"]

    for workflow in [
        "image.ideogram-4-design.v1",
        "image.krea-2-creative-image.v1",
        "image.mage-flow-image.v1",
    ]:
        assert schemas[workflow]["aspect_ratio"].options == aspect_ratios
        assert schemas[workflow]["aspect_ratio"].default == "1:1"
        seed = schemas[workflow]["seed"]
        assert (seed.default, seed.min, seed.max) == (0, 0, 0xFFFFFFFFFFFFFFFF)

    assert schemas["image.ideogram-4-design.v1"]["quality_mode"].options == ["quality", "balanced", "fast"]
    assert schemas["image.ideogram-4-design.v1"]["quality_mode"].default == "balanced"
    assert schemas["image.krea-2-creative-image.v1"]["prompt_enhance"].default is True
    assert schemas["image.mage-flow-image.v1"]["negative_prompt"].default == ""

    guidance = schemas["image.flux-2-reference-edit.v1"]["guidance"]
    assert (guidance.default, guidance.min, guidance.max, guidance.step) == (4.0, 1.0, 10.0, 0.1)
    for workflow in ["image.flux-2-reference-edit.v1", "image.qwen-image-edit-2511.v1"]:
        assert schemas[workflow]["quality_mode"].options == ["quality", "fast"]
        assert schemas[workflow]["quality_mode"].default == "quality"
        seed = schemas[workflow]["seed"]
        assert (seed.default, seed.min, seed.max) == (0, 0, 0xFFFFFFFFFFFFFFFF)

    scale = schemas["image.seedvr2-image-upscale.v1"]["scale"]
    assert scale.options == ["2x", "4x"]
    assert scale.default == "4x"


def test_image_poc_api_declarations_and_extension_registration():
    workflows = {workflow for _, workflow, _, _ in IMAGE_POC_NODES}
    registered = set(asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list()))

    assert workflows <= set(get_args(ComfyCloudWorkflow))
    assert {node for node, _, _, _ in IMAGE_POC_NODES} <= registered


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

    assert nodes_comfy_cloud.COMFY_CLOUD_GPU_SECOND_USD == 0.001295
    assert nodes_comfy_cloud.COMFY_CLOUD_CREDITS_PER_USD == 211
    assert nodes_comfy_cloud.COMFY_CLOUD_GPU_SECOND_CREDITS == pytest.approx(0.273245)
    assert nodes_comfy_cloud.COMFY_CLOUD_GPU_HOUR_USD == pytest.approx(4.662)
    assert nodes_comfy_cloud.COMFY_CLOUD_GPU_HOUR_CREDITS == pytest.approx(983.682)
    for node in nodes:
        schema = node.define_schema()
        badge = schema.price_badge.as_dict(schema.inputs)
        assert badge["expr"] == (
            '{"type":"usd","usd":0.001295,"format":{"suffix":"/GPU-second","approximate":true}}'
        )
        assert "Estimated compute rate" in schema.description
        assert "Actual final cost depends on GPU runtime" in schema.description


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


def test_linked_values_are_validated_before_upload(monkeypatch):
    upload = AsyncMock()
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)

    with pytest.raises(ValueError, match="guidance"):
        asyncio.run(
            nodes_comfy_cloud.ComfyCloudFlux2ReferenceEditNode.execute(
                object(), "instruction", float("nan"), "quality", 0
            )
        )
    upload.assert_not_awaited()


def test_upload_inputs_have_decoded_resource_limits():
    oversized_image = torch.empty((1, 8193, 1, 3), device="meta")
    oversized_audio = {
        "waveform": torch.empty((1, 2, nodes_comfy_cloud._MAX_DECODED_AUDIO_BYTES // 8 + 1), device="meta"),
        "sample_rate": 48000,
    }

    with pytest.raises(ValueError, match="32-megapixel"):
        nodes_comfy_cloud._validate_image_upload(oversized_image)
    with pytest.raises(ValueError, match="256 MiB"):
        nodes_comfy_cloud._validate_audio_upload(oversized_audio)


@pytest.mark.parametrize(
    ("node", "input_names"),
    [
        (nodes_comfy_cloud.ComfyCloudMiniMaxH3TextSoundNode, ["prompt", "aspect_ratio", "duration_seconds", "seed"]),
        (nodes_comfy_cloud.ComfyCloudMiniMaxH3ImageSoundNode, ["image", "prompt", "aspect_ratio", "duration_seconds", "seed"]),
        (nodes_comfy_cloud.ComfyCloudLTX23ImageAudioPerformanceNode, ["image", "audio", "prompt", "enhance_prompt", "duration_seconds", "seed"]),
        (nodes_comfy_cloud.ComfyCloudLTX23FirstLastFrameNode, ["first_frame", "last_frame", "prompt", "duration_seconds", "seed"]),
        (nodes_comfy_cloud.ComfyCloudWan22FirstLastFrameNode, ["first_frame", "last_frame", "prompt", "negative_prompt", "duration_seconds", "seed"]),
        (nodes_comfy_cloud.ComfyCloudSCAIL2CharacterReplacementNode, ["reference_character", "driving_video", "scene_prompt", "driving_subject", "reference_subject", "seed"]),
    ],
)
def test_video_node_schemas_expose_only_manifest_inputs(node, input_names):
    schema = node.define_schema()
    assert schema.is_api_node
    assert [input.id for input in schema.inputs] == input_names
    assert len(schema.outputs) == 1
    assert schema.outputs[0].get_io_type() == "VIDEO"


def test_ltx_performance_stages_image_and_audio(monkeypatch):
    run = AsyncMock(return_value=("video-output",))
    image_upload = AsyncMock(return_value="https://example.com/image.png")
    audio_upload = AsyncMock(return_value="https://example.com/audio.mp4")
    monkeypatch.setattr(nodes_comfy_cloud, "_run_video_workflow", run)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", image_upload)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_audio_to_comfyapi", audio_upload)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)
    audio = {"waveform": torch.zeros(1, 1, 480000), "sample_rate": 48000}

    asyncio.run(nodes_comfy_cloud.ComfyCloudLTX23ImageAudioPerformanceNode.execute(object(), audio, "sing", True, 9, 7))

    inputs = run.call_args.args[2]
    assert inputs.image_url == "https://example.com/image.png"
    assert inputs.audio_url == "https://example.com/audio.mp4"
    assert inputs.duration_seconds == 9


def test_scail_stages_reference_image_and_driving_video(monkeypatch):
    run = AsyncMock(return_value=("video-output",))
    image_upload = AsyncMock(return_value="https://example.com/character.png")
    video_upload = AsyncMock(return_value="https://example.com/driving.mp4")
    video = Mock()
    video.get_frame_count.return_value = 100
    monkeypatch.setattr(nodes_comfy_cloud, "_run_video_workflow", run)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", image_upload)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_video_to_comfyapi", video_upload)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)

    asyncio.run(nodes_comfy_cloud.ComfyCloudSCAIL2CharacterReplacementNode.execute(object(), video, "park", "woman", "human", 1))

    inputs = run.call_args.args[2]
    assert inputs.reference_character_url == "https://example.com/character.png"
    assert inputs.driving_video_url == "https://example.com/driving.mp4"
    video.get_frame_count.assert_called_once()


def test_scail_defaults_and_frame_count_fail_closed_before_upload(monkeypatch):
    schema = {input.id: input for input in nodes_comfy_cloud.ComfyCloudSCAIL2CharacterReplacementNode.define_schema().inputs}
    image_upload = AsyncMock()
    video_upload = AsyncMock()
    video = Mock()
    video.get_frame_count.side_effect = RuntimeError("decode failed")
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", image_upload)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_video_to_comfyapi", video_upload)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)

    assert schema["driving_subject"].default == "human"
    with pytest.raises(ValueError, match="Unable to determine video frame count"):
        asyncio.run(nodes_comfy_cloud.ComfyCloudSCAIL2CharacterReplacementNode.execute(object(), video, "park", "human", "human", 1))
    image_upload.assert_not_awaited()
    video_upload.assert_not_awaited()


def test_scail_rejects_missing_frame_count_metadata_before_upload(monkeypatch):
    upload = AsyncMock()
    video = Mock()
    video.get_frame_count.return_value = None
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_video_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)

    with pytest.raises(ValueError, match="Unable to determine video frame count"):
        asyncio.run(
            nodes_comfy_cloud.ComfyCloudSCAIL2CharacterReplacementNode.execute(
                object(), video, "park", "human", "human", 1
            )
        )
    upload.assert_not_awaited()


@pytest.mark.parametrize("frame_count", [80, 158])
def test_scail_rejects_out_of_range_frames_before_upload(monkeypatch, frame_count):
    upload = AsyncMock()
    video = Mock()
    video.get_frame_count.return_value = frame_count
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_video_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)

    with pytest.raises(ValueError, match="frame count"):
        asyncio.run(nodes_comfy_cloud.ComfyCloudSCAIL2CharacterReplacementNode.execute(object(), video, "park", "human", "human", 1))
    upload.assert_not_awaited()


def test_in_memory_download_resets_retry_and_enforces_stream_limit(monkeypatch):
    class Content:
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

    class Response:
        status = 200
        headers = {}

        def __init__(self, chunks, content_length=None):
            self.content = Content(chunks)
            self.content_length = content_length

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    responses = [Response([b"partial", aiohttp.ClientPayloadError("retry")]), Response([b"final", b""])]

    class Session:
        def __init__(self, timeout):
            pass

        async def get(self, url, headers, allow_redirects=True):
            return responses.pop(0)

        async def close(self):
            pass

    monkeypatch.setattr(download_helpers.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(download_helpers, "sleep_with_interrupt", AsyncMock())
    destination = BytesIO()

    asyncio.run(download_helpers.download_url_to_bytesio("https://example.com/result", destination))
    assert destination.read() == b"final"

    monkeypatch.setattr(download_helpers, "_MAX_IN_MEMORY_DOWNLOAD_BYTES", 4)
    responses.append(Response([b"12345", b""]))
    with pytest.raises(ValueError, match="in-memory limit"):
        asyncio.run(download_helpers.download_url_to_bytesio("https://example.com/result", BytesIO()))

    responses.append(Response([], content_length=5))
    with pytest.raises(ValueError, match="in-memory limit"):
        asyncio.run(download_helpers.download_url_to_bytesio("https://example.com/result", BytesIO()))


def test_file_object_download_resets_retry_and_enforces_stream_limit(monkeypatch, tmp_path):
    destination = (tmp_path / "result.bin").open("w+b")
    destination.write(b"stale")

    class Content:
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

    class Response:
        status = 200
        headers = {}
        content_length = None

        def __init__(self, chunks):
            self.content = Content(chunks)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    responses = [Response([b"partial", aiohttp.ClientPayloadError("retry")]), Response([b"done", b""])]

    class Session:
        def __init__(self, timeout):
            pass

        async def get(self, url, headers, allow_redirects=True):
            return responses.pop(0)

        async def close(self):
            pass

    monkeypatch.setattr(download_helpers.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(download_helpers, "sleep_with_interrupt", AsyncMock())
    monkeypatch.setattr(download_helpers, "_MAX_IN_MEMORY_DOWNLOAD_BYTES", 10)

    asyncio.run(download_helpers.download_url_to_bytesio("https://example.com/result", destination))
    assert destination.read() == b"done"

    monkeypatch.setattr(download_helpers, "_MAX_IN_MEMORY_DOWNLOAD_BYTES", 4)
    responses.append(Response([b"12345", b""]))
    with pytest.raises(ValueError, match="in-memory limit"):
        asyncio.run(download_helpers.download_url_to_bytesio("https://example.com/result", destination))
    destination.close()


def test_download_cloud_audio_url_to_audio_input(monkeypatch):
    node = nodes_comfy_cloud.ComfyCloudTextToImageNode
    downloaded = b"encoded audio"
    expected = {"waveform": torch.ones(1, 2, 3), "sample_rate": 48000}
    download_call = Mock()

    async def download(url, dest, **kwargs):
        download_call(url=url, dest=dest, **kwargs)
        dest.write(downloaded)
        dest.seek(0)

    audio_decode = Mock(return_value=expected)
    monkeypatch.setattr(download_helpers, "download_url_to_bytesio", download)
    monkeypatch.setattr(download_helpers, "audio_bytes_to_audio_input", audio_decode)

    output = asyncio.run(
        download_helpers.download_url_to_audio_input(
            "/proxy/comfy-cloud/results/task-1/audio.flac",
            timeout=30,
            max_retries=2,
            cls=node,
        )
    )

    assert output is expected
    download_call.assert_called_once()
    assert download_call.call_args.kwargs["url"] == "/proxy/comfy-cloud/results/task-1/audio.flac"
    assert isinstance(download_call.call_args.kwargs["dest"], BytesIO)
    assert download_call.call_args.kwargs["timeout"] == 30
    assert download_call.call_args.kwargs["max_retries"] == 2
    audio_decode.assert_called_once()
    assert isinstance(audio_decode.call_args.args[0], BytesIO)
    assert audio_decode.call_args.kwargs == {}
    assert download_call.call_args.kwargs["cls"] is node
    assert download_call.call_args.kwargs["allow_redirects"] is True


def test_audio_decode_stops_before_exceeding_budget(monkeypatch):
    class Frame:
        def to_ndarray(self):
            return torch.ones(2, 8).numpy()

    stream = type(
        "Stream",
        (),
        {"codec_context": type("Codec", (), {"sample_rate": 48000})(), "channels": 2, "index": 0},
    )()

    class AudioFile:
        streams = type("Streams", (), {"audio": [stream]})()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def decode(self, streams):
            yield Frame()
            yield Frame()

    monkeypatch.setattr(conversions.av, "open", lambda source: AudioFile())
    monkeypatch.setattr(conversions, "_MAX_DECODED_AUDIO_BYTES", 64)

    with pytest.raises(ValueError, match="Decoded audio exceeds"):
        conversions.audio_bytes_to_audio_input(BytesIO(b"encoded"))


AUDIO_POC_NODES = [
    (nodes_comfy_cloud.ComfyCloudACEStep15XLTurboNode, "audio.ace-step-1-5-xl-turbo.v1", ["style_prompt", "lyrics", "duration_seconds", "seed", "bpm", "time_signature", "language", "key"]),
    (nodes_comfy_cloud.ComfyCloudStableAudio3MediumNode, "audio.stable-audio-3-medium.v1", ["prompt", "duration_seconds", "seed", "expand_prompt", "category"]),
    (nodes_comfy_cloud.ComfyCloudChatterboxMultilingualVoiceCloneNode, "audio.chatterbox-multilingual-voice-clone.v1", ["text", "voice_reference", "language", "exaggeration", "cfg_weight", "temperature", "seed"]),
    (nodes_comfy_cloud.ComfyCloudChatterboxDialogueNode, "audio.chatterbox-dialogue.v1", ["script", "speaker_a_reference", "speaker_b_reference", "exaggeration", "cfg_weight", "temperature", "seed"]),
    (nodes_comfy_cloud.ComfyCloudChatterboxVoiceConversionNode, "audio.chatterbox-voice-conversion.v1", ["source_audio", "target_voice_reference", "seed"]),
    (nodes_comfy_cloud.ComfyCloudMelBandRoFormerStemSeparationNode, "audio.melbandroformer-stem-separation.v1", ["audio"]),
]


@pytest.mark.parametrize(("node", "workflow", "input_names"), AUDIO_POC_NODES)
def test_audio_poc_schemas_and_registration(node, workflow, input_names):
    schema = node.define_schema()
    registered = asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list())

    assert schema.is_api_node
    assert schema.category == "partner/audio/Comfy Cloud"
    assert [input.id for input in schema.inputs] == input_names
    assert all(output.get_io_type() == "AUDIO" for output in schema.outputs)
    assert workflow in get_args(ComfyCloudWorkflow)
    assert node in registered


def test_audio_poc_schema_defaults_ranges_and_enums():
    schemas = {workflow: {input.id: input for input in node.define_schema().inputs} for node, workflow, _ in AUDIO_POC_NODES}
    ace = schemas["audio.ace-step-1-5-xl-turbo.v1"]
    assert (ace["duration_seconds"].default, ace["duration_seconds"].min, ace["duration_seconds"].max, ace["duration_seconds"].step) == (120, 10, 300, 0.1)
    assert (ace["bpm"].default, ace["bpm"].min, ace["bpm"].max) == (120, 10, 300)
    assert ace["time_signature"].options == ["2", "3", "4", "6"]
    assert ace["language"].default == "en"
    assert ace["key"].default == "E minor"

    stable = schemas["audio.stable-audio-3-medium.v1"]
    assert stable["category"].options == ["Music", "Instrument", "SFX", "One-shot"]
    assert stable["expand_prompt"].default is True
    for workflow in [
        "audio.chatterbox-multilingual-voice-clone.v1",
        "audio.chatterbox-dialogue.v1",
        "audio.chatterbox-voice-conversion.v1",
    ]:
        assert schemas[workflow]["seed"].max == 0xFFFFFFFF

    mel_schema = nodes_comfy_cloud.ComfyCloudMelBandRoFormerStemSeparationNode.define_schema()
    assert [output.id for output in mel_schema.outputs] == ["vocals", "instruments"]


def test_audio_poc_request_mapping_and_named_result_decoding(monkeypatch):
    sync = AsyncMock(return_value=ComfyCloudGenerateResponse(task_id="task-audio", status="queued", polling_url="/tasks/task-audio", cancel_url="/tasks/task-audio/cancel"))
    poll = AsyncMock(return_value=ComfyCloudStatusResponse(task_id="task-audio", status="completed", output_urls={"vocals": "/proxy/comfy-cloud/results/vocals.mp3", "instruments": "/proxy/comfy-cloud/results/instruments.mp3"}))
    upload = AsyncMock(return_value="/uploads/song.m4a")
    download = AsyncMock(side_effect=["vocals-audio", "instruments-audio"])
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)
    monkeypatch.setattr(nodes_comfy_cloud, "poll_op", poll)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_audio_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_audio_input", download)
    audio = {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}

    output = asyncio.run(nodes_comfy_cloud.ComfyCloudMelBandRoFormerStemSeparationNode.execute(audio))

    request = sync.call_args.kwargs["data"]
    assert request.workflow == "audio.melbandroformer-stem-separation.v1"
    assert request.inputs.model_dump(exclude_none=True) == {"assets": {"audio": {"type": "AUDIO", "url": "/uploads/song.m4a"}}}
    assert [call.args[0] for call in download.await_args_list] == [
        "/proxy/comfy-cloud/results/vocals.mp3",
        "/proxy/comfy-cloud/results/instruments.mp3",
    ]
    assert all(call.kwargs["timeout"] == 30 * 60 for call in download.await_args_list)
    assert all(call.kwargs["allow_redirects"] is False for call in download.await_args_list)
    assert poll.call_args.kwargs["cancel_endpoint"].path == "/proxy/comfy-cloud/workflow/tasks/task-audio/cancel"
    assert tuple(output) == ("vocals-audio", "instruments-audio")


def test_chatterbox_audio_inputs_use_named_staged_assets(monkeypatch):
    run = AsyncMock(return_value=("audio-output",))
    upload = AsyncMock(side_effect=["/uploads/source.m4a", "/uploads/target.m4a"])
    monkeypatch.setattr(nodes_comfy_cloud, "_run_audio_workflow", run)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_audio_to_comfyapi", upload)
    source = {"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000}
    target = {"waveform": torch.zeros(1, 1, 96000), "sample_rate": 48000}

    asyncio.run(nodes_comfy_cloud.ComfyCloudChatterboxVoiceConversionNode.execute(source, target, 7))

    inputs = run.call_args.args[2]
    assert inputs.model_dump(exclude_none=True) == {
        "assets": {
            "source_audio": {"type": "AUDIO", "url": "/uploads/source.m4a"},
            "target_voice_reference": {"type": "AUDIO", "url": "/uploads/target.m4a"},
        },
        "seed": 7,
    }
    assert "audio_url" not in inputs.model_dump(exclude_none=True)


def test_chatterbox_dialogue_rejects_invalid_speaker_labels(monkeypatch):
    upload = AsyncMock()
    monkeypatch.setattr(nodes_comfy_cloud, "upload_audio_to_comfyapi", upload)
    audio = {"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000}

    with pytest.raises(ValueError, match="only speakers A and B"):
        asyncio.run(nodes_comfy_cloud.ComfyCloudChatterboxDialogueNode.execute("NARRATOR: Hello", audio, audio, 0.5, 0.5, 0.8, 0))
    upload.assert_not_awaited()


def test_chatterbox_dialogue_normalizes_labels_and_continuations(monkeypatch):
    run = AsyncMock(return_value=("audio-output",))
    monkeypatch.setattr(nodes_comfy_cloud, "_run_audio_workflow", run)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_audio_to_comfyapi", AsyncMock(side_effect=["/a", "/b"]))
    audio = {"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000}

    asyncio.run(nodes_comfy_cloud.ComfyCloudChatterboxDialogueNode.execute("a: Hello\ncontinued\n\nSpeaker B: Hi", audio, audio, 0.5, 0.5, 0.8, 0))

    assert run.call_args.args[2].script == "SPEAKER A: Hello continued\nSPEAKER B: Hi"


def test_chatterbox_dialogue_accepts_colons_and_label_only_lines():
    assert nodes_comfy_cloud._normalize_dialogue("SPEAKER A :\nMeet at 10:30\nhttps://example.com\nB: Done") == (
        "SPEAKER A: Meet at 10:30 https://example.com\nSPEAKER B: Done"
    )


@pytest.mark.parametrize("script", ["SPEAKER A:", "just a continuation", "   "])
def test_chatterbox_dialogue_rejects_blank_or_unattributed_text(monkeypatch, script):
    upload = AsyncMock()
    monkeypatch.setattr(nodes_comfy_cloud, "upload_audio_to_comfyapi", upload)
    audio = {"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000}

    with pytest.raises(Exception):
        asyncio.run(nodes_comfy_cloud.ComfyCloudChatterboxDialogueNode.execute(script, audio, audio, 0.5, 0.5, 0.8, 0))
    upload.assert_not_awaited()


def test_audio_duration_tolerates_one_sample_and_rejects_invalid_sample_rate():
    one_sample_over = {"waveform": torch.zeros(1, 1, 48001), "sample_rate": 48000}
    nodes_comfy_cloud._validate_audio_duration("Audio", one_sample_over, 0.5, 1)

    with pytest.raises(ValueError, match="sample rate"):
        nodes_comfy_cloud._validate_audio_duration("Audio", {"waveform": torch.zeros(1, 1, 1), "sample_rate": 0}, 0.5, 1)

    with pytest.raises(ValueError, match="between"):
        nodes_comfy_cloud._validate_audio_duration("Audio", {"waveform": torch.zeros(1, 1, 31), "sample_rate": 1}, 1, 30)


@pytest.mark.parametrize(("file_format", "expected_format"), [(".GLB", "glb"), ("SPZ", "spz")])
def test_download_cloud_3d_url_to_file_3d(monkeypatch, file_format, expected_format):
    node = nodes_comfy_cloud.ComfyCloudTextToImageNode
    downloaded = b"3d result"
    calls = []

    async def download(url, dest, **kwargs):
        calls.append((url, dest, kwargs))
        dest.write(downloaded)
        dest.seek(0)

    monkeypatch.setattr(download_helpers, "download_url_to_bytesio", download)

    output = asyncio.run(
        download_helpers.download_url_to_file_3d(
            f"/proxy/comfy-cloud/results/task-1/model.{expected_format}",
            file_format,
            timeout=45,
            max_retries=3,
            cls=node,
        )
    )

    assert output.format == expected_format
    assert output.get_bytes() == downloaded
    assert calls[0][0] == f"/proxy/comfy-cloud/results/task-1/model.{expected_format}"
    assert isinstance(calls[0][1], BytesIO)
    assert calls[0][2] == {"timeout": 45, "max_retries": 3, "cls": node, "allow_redirects": True}


THREE_D_POC_NODES = [
    (nodes_comfy_cloud.ComfyCloudTripoSplatImageToGaussianSplatNode, "3d.triposplat-image-to-gaussian-splat.v1", ["image", "remove_background", "seed", "gaussian_count"], {"image": object(), "remove_background": False, "seed": 7, "gaussian_count": 32768}, "FILE_3D_SPZ", "spz"),
    (nodes_comfy_cloud.ComfyCloudHunyuan3D21ImageTo3DNode, "3d.hunyuan3d-2-1-image-to-3d.v1", ["image", "seed"], {"image": object(), "seed": 8}, "FILE_3D_GLB", "glb"),
    (nodes_comfy_cloud.ComfyCloudHunyuan3DMultiViewTo3DNode, "3d.hunyuan3d-multiview-to-3d.v1", ["front_image", "back_image", "seed"], {"front_image": object(), "back_image": object(), "seed": 9}, "FILE_3D_GLB", "glb"),
    (nodes_comfy_cloud.ComfyCloudMoGe2PhotoToTexturedMeshNode, "3d.moge-2-photo-to-textured-mesh.v1", ["image", "fov_degrees", "detail", "mesh_decimation", "gap_threshold", "texture"], {"image": object(), "fov_degrees": 45.5, "detail": 8, "mesh_decimation": 2, "gap_threshold": 0.05, "texture": False}, "FILE_3D_GLB", "glb"),
    (nodes_comfy_cloud.ComfyCloudMoGe2PanoramaTo3DSceneNode, "3d.moge-2-panorama-to-3d-scene.v1", ["panorama", "detail", "split_resolution", "merge_resolution", "mesh_decimation", "gap_threshold", "texture"], {"panorama": object(), "detail": 6, "split_resolution": 768, "merge_resolution": 2048, "mesh_decimation": 3, "gap_threshold": 0.06, "texture": False}, "FILE_3D_GLB", "glb"),
]


@pytest.mark.parametrize(("node", "workflow", "input_names", "arguments", "output_type", "file_format"), THREE_D_POC_NODES)
def test_3d_poc_node_schema_request_mapping_and_registration(monkeypatch, node, workflow, input_names, arguments, output_type, file_format):
    run = AsyncMock(return_value=("3d-output",))
    upload = AsyncMock(side_effect=["/uploads/front.png", "/uploads/back.png"])
    monkeypatch.setattr(nodes_comfy_cloud, "_run_3d_workflow", run)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)

    schema = node.define_schema()
    assert schema.is_api_node
    assert schema.category == "partner/3d/Comfy Cloud"
    assert [input.id for input in schema.inputs] == input_names
    assert schema.outputs[0].get_io_type() == output_type
    assert workflow in get_args(ComfyCloudWorkflow)
    assert node in asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list())

    output = asyncio.run(node.execute(**arguments))
    assert output[0] == "3d-output"
    assert run.call_args.args[1] == workflow
    assert run.call_args.args[3] == file_format
    request_inputs = run.call_args.args[2].model_dump(exclude_none=True)
    image_names = [name for name in ("image", "front_image", "back_image", "panorama") if name in arguments]
    expected_inputs = {name: value for name, value in arguments.items() if name not in image_names}
    expected_inputs["assets"] = {
        name: {"type": "IMAGE", "url": f"/uploads/{'front' if index == 0 else 'back'}.png"}
        for index, name in enumerate(image_names)
    }
    assert request_inputs == expected_inputs
    assert '"id"' not in run.call_args.args[2].model_dump_json()


def test_3d_poc_schema_defaults_and_ranges():
    schemas = {workflow: {input.id: input for input in node.define_schema().inputs} for node, workflow, _, _, _, _ in THREE_D_POC_NODES}
    tripo = schemas["3d.triposplat-image-to-gaussian-splat.v1"]
    assert tripo["remove_background"].default is True
    assert (tripo["seed"].default, tripo["seed"].min, tripo["seed"].max) == (46, 0, 0xFFFFFFFFFFFFFFFF)
    assert (tripo["gaussian_count"].default, tripo["gaussian_count"].min, tripo["gaussian_count"].max) == (262144, 32768, 262144)
    assert "application/octet-stream" in nodes_comfy_cloud.ComfyCloudTripoSplatImageToGaussianSplatNode.define_schema().outputs[0].tooltip
    assert schemas["3d.hunyuan3d-2-1-image-to-3d.v1"]["seed"].default == 952805179515179
    assert schemas["3d.hunyuan3d-multiview-to-3d.v1"]["seed"].default == 502126049100058
    photo = schemas["3d.moge-2-photo-to-textured-mesh.v1"]
    assert (photo["fov_degrees"].default, photo["fov_degrees"].min, photo["fov_degrees"].max, photo["fov_degrees"].step) == (0, 0, 170, 0.1)
    assert (photo["detail"].default, photo["detail"].min, photo["detail"].max) == (9, 0, 9)
    panorama = schemas["3d.moge-2-panorama-to-3d-scene.v1"]
    assert (panorama["split_resolution"].default, panorama["split_resolution"].min, panorama["split_resolution"].max) == (512, 256, 1024)
    assert (panorama["merge_resolution"].default, panorama["merge_resolution"].min, panorama["merge_resolution"].max) == (1024, 256, 8192)


def test_hunyuan_multiview_validates_both_images_before_upload(monkeypatch):
    upload = AsyncMock()
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: image)

    with pytest.raises(ValueError, match="Exactly one front image and one back image"):
        asyncio.run(nodes_comfy_cloud.ComfyCloudHunyuan3DMultiViewTo3DNode.execute(1, 2, 9))
    upload.assert_not_awaited()


def test_extension_preserves_all_23_poc_node_registrations():
    legacy_nodes = {
        nodes_comfy_cloud.ComfyCloudTextToImageNode,
        nodes_comfy_cloud.ComfyCloudTextToVideoNode,
        nodes_comfy_cloud.ComfyCloudImageToVideoNode,
        nodes_comfy_cloud.ComfyCloudImageEditNode,
    }
    registered = set(asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list()))

    assert len(registered - legacy_nodes) == 23


def test_3d_workflow_submission_polling_cancel_and_download(monkeypatch):
    sync = AsyncMock(return_value=ComfyCloudGenerateResponse(task_id="task-3d", status="queued", polling_url="/tasks/task-3d", cancel_url="/tasks/task-3d/cancel"))
    poll = AsyncMock(return_value=ComfyCloudStatusResponse(task_id="task-3d", status="completed", output_url="/proxy/comfy-cloud/results/model.spz"))
    download = AsyncMock(return_value="spz-output")
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)
    monkeypatch.setattr(nodes_comfy_cloud, "poll_op", poll)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_file_3d", download)

    output = asyncio.run(nodes_comfy_cloud._run_3d_workflow(nodes_comfy_cloud.ComfyCloudTripoSplatImageToGaussianSplatNode, "3d.triposplat-image-to-gaussian-splat.v1", ComfyCloudWorkflowInputs(seed=46), "spz"))

    request = sync.call_args.kwargs["data"]
    assert request.workflow == "3d.triposplat-image-to-gaussian-splat.v1"
    assert request.inputs.model_dump(exclude_none=True) == {"seed": 46}
    assert poll.call_args.args[1].path == "/proxy/comfy-cloud/workflow/tasks/task-3d"
    assert poll.call_args.kwargs["cancel_endpoint"].path == "/proxy/comfy-cloud/workflow/tasks/task-3d/cancel"
    assert poll.call_args.kwargs["cancel_endpoint"].method == "POST"
    download.assert_awaited_once_with(
        "/proxy/comfy-cloud/results/model.spz",
        "spz",
        timeout=30 * 60,
        cls=nodes_comfy_cloud.ComfyCloudTripoSplatImageToGaussianSplatNode,
        allow_redirects=False,
    )
    assert output[0] == "spz-output"

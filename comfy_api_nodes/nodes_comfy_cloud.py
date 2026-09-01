import math
import posixpath
from typing import ClassVar
from urllib.parse import quote, unquote, urlsplit

import torch

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.comfy_cloud import (
    ComfyCloudAssetInput,
    ComfyCloudGenerateRequest,
    ComfyCloudGenerateResponse,
    ComfyCloudStatusResponse,
    ComfyCloudWorkflow,
    ComfyCloudWorkflowInputs,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    sync_op_raw,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    validate_string,
)


# Mirrors comfyCloudOutputBuckets on the backend (services/comfy-api/server/middleware/comfy_cloud.go).
# Buckets a Comfy Cloud output may be served from. MUST stay in step with
# comfyCloudOutputBuckets in cloud's services/comfy-api/config/config.go: widen
# one without the other and every run completes on the GPU and then has its
# output rejected here. The partner-nodes-* entries are the buckets staging and
# prod already point the shared partner-node asset key at.
_OUTPUT_BUCKETS = frozenset(
    {
        "comfy-cloud-assets",
        "comfy-cloud-assets-stg",
        "comfy-cloud-assets-test",
        "partner-nodes-assets",
        "partner-nodes-assets-staging",
    }
)
_GENERATE_ENDPOINT = ApiEndpoint(path="/proxy/comfy-cloud/workflow/generate", method="POST")
# Three separate budgets that are easy to confuse and must not drift apart:
#   _RUN_TIMEOUT_SECONDS      the platform's own ceiling on a Cloud job (ingest stamps
#                             max_runtime; 2100s is the default tier value). We do not
#                             enforce it, we poll under it.
#   _POLL_* below             how long this node waits for that job. Deliberately a little
#                             ABOVE the platform ceiling so a job that is still legitimately
#                             running is never abandoned early, and the job's own timeout is
#                             what ends it.
#   _OUTPUT_DOWNLOAD_TIMEOUT  a separate budget for fetching the finished artifact, which
#                             happens only after the job has already succeeded.
_RUN_TIMEOUT_SECONDS = 2100
_POLL_INTERVAL_SECONDS = 5.0
_POLL_MAX_ATTEMPTS = int(_RUN_TIMEOUT_SECONDS / _POLL_INTERVAL_SECONDS) + 24  # +2 min of slack
_OUTPUT_DOWNLOAD_TIMEOUT = 30 * 60
_MAX_UPLOAD_IMAGE_PIXELS = 32_000_000
_MAX_UPLOAD_IMAGE_DIMENSION = 8192
_MAX_DECODED_AUDIO_BYTES = 256 * 1024 * 1024
# The rate Metronome actually charges for a Comfy Cloud run: the "GPU Hour Usage"
# billable metric sums gpu_seconds and prices it per gpu_type, and the rate for
# rtx_pro_6000 is 0.185 cents per GPU-second. Confirmed against the live rate
# card, not derived here.
#
# This is a HARDCODED MIRROR of that rate card and nothing links the two, so it
# will drift the moment pricing changes. It previously read 0.001295, which
# under-quoted every run by ~43% against the user. If the rate card moves, this
# has to move with it, and it ships on the ComfyUI release train.
COMFY_CLOUD_GPU_SECOND_USD = 0.00185
COMFY_CLOUD_CREDITS_PER_USD = 211
COMFY_CLOUD_GPU_SECOND_CREDITS = COMFY_CLOUD_GPU_SECOND_USD * COMFY_CLOUD_CREDITS_PER_USD
_COMFY_CLOUD_PRICE_BADGE = IO.PriceBadge(
    expr=(
        f'{{"type":"usd","usd":{COMFY_CLOUD_GPU_SECOND_USD:.6f},'
        '"format":{"suffix":"/GPU-second","approximate":true}}'
    )
)
_COMFY_CLOUD_RATE_DESCRIPTION = (
    f" Runs on a Comfy Cloud GPU, billed by how long it runs at "
    f"${COMFY_CLOUD_GPU_SECOND_USD:.5f}/GPU-second "
    f"({COMFY_CLOUD_GPU_SECOND_CREDITS:.2f} credits). Paid in credits, no Cloud "
    "subscription required."
)


def _comfy_cloud_description(summary: str) -> str:
    """Node descriptions carry the rate because the price badge only renders on
    Nodes 2.0, and a plain local install still defaults to the classic canvas."""
    return summary + _COMFY_CLOUD_RATE_DESCRIPTION
_TEXT_LIMITS = {
    "prompt": (1, 4096),
    "instruction": (1, 4096),
    "negative_prompt": (0, 2048),
    "scene_prompt": (1, 4096),
    "driving_subject": (1, 256),
    "reference_subject": (1, 256),
    "style_prompt": (1, 4096),
    "lyrics": (0, 20000),
    "text": (1, 5000),
    "script": (1, 10000),
}


def _task_endpoints(task_id: str) -> tuple[ApiEndpoint, ApiEndpoint]:
    if not task_id.strip():
        raise ValueError("Comfy Cloud returned an empty task ID.")
    task_path = f"/proxy/comfy-cloud/workflow/tasks/{quote(task_id, safe='')}"
    return ApiEndpoint(path=task_path), ApiEndpoint(path=f"{task_path}/cancel", method="POST")


def _with_input_sockets(inputs: list[IO.Input]) -> list[IO.Input]:
    for input_spec in inputs:
        if isinstance(input_spec, IO.WidgetInput):
            input_spec.socketless = False
    return inputs


def _validated_output_url(url: str) -> str:
    parsed = urlsplit(url)
    decoded_path = unquote(parsed.path)
    is_proxy_path = (
        not parsed.scheme
        and not parsed.netloc
        and decoded_path.startswith("/proxy/comfy-cloud/")
        and posixpath.normpath(decoded_path) == decoded_path
    )
    # The bucket is the first path segment of a signed GCS URL. Pin it to the same set the
    # backend enforces (comfyCloudOutputBuckets) rather than trusting any bucket on the host:
    # the URL is backend-supplied, so this is defence in depth, but the asymmetry with the Go
    # side is free to close and an unpinned host check is the kind of thing that quietly stops
    # being true.
    bucket = decoded_path.lstrip("/").split("/", 1)[0]
    is_signed_https_url = (
        parsed.scheme == "https"
        and parsed.hostname == "storage.googleapis.com"
        and parsed.port is None
        and parsed.username is None
        and parsed.password is None
        and bucket in _OUTPUT_BUCKETS
    )
    if not is_proxy_path and not is_signed_https_url:
        raise RuntimeError("Comfy Cloud returned an invalid output URL.")
    return url


def _validate_image_upload(image: Input.Image) -> None:
    if not isinstance(image, torch.Tensor):
        return
    if image.ndim not in (3, 4):
        raise ValueError("Invalid input image shape.")
    height, width = image.shape[-3:-1]
    if max(height, width) > _MAX_UPLOAD_IMAGE_DIMENSION or height * width > _MAX_UPLOAD_IMAGE_PIXELS:
        raise ValueError("Input image exceeds the 8192px or 32-megapixel Comfy Cloud limit.")


def _validate_audio_upload(audio: Input.Audio) -> None:
    waveform = audio["waveform"]
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2):
        raise ValueError("Audio must contain one mono or stereo waveform.")
    if waveform.numel() * waveform.element_size() > _MAX_DECODED_AUDIO_BYTES:
        raise ValueError("Decoded audio exceeds the 256 MiB Comfy Cloud limit.")


def _progress(response: ComfyCloudStatusResponse) -> float | None:
    if response.progress is None or not math.isfinite(response.progress):
        return None
    return min(100.0, max(0.0, response.progress))


async def _poll_task(cls: type[IO.ComfyNode], task_id: str) -> ComfyCloudStatusResponse:
    polling_endpoint, cancel_endpoint = _task_endpoints(task_id)
    try:
        return await poll_op(
            cls,
            polling_endpoint,
            response_model=ComfyCloudStatusResponse,
            status_extractor=lambda response: response.status,
            progress_extractor=_progress,
            cancel_endpoint=cancel_endpoint,
            poll_interval=_POLL_INTERVAL_SECONDS,
            max_poll_attempts=_POLL_MAX_ATTEMPTS,
        )
    except Exception:
        try:
            await sync_op_raw(cls, cancel_endpoint, max_retries=0)
        except Exception:
            pass
        raise


def _validate_node_inputs(cls: type[IO.ComfyNode], values: dict) -> dict:
    validated = dict(values)
    for input_spec in cls.define_schema().inputs:
        if not isinstance(input_spec, IO.WidgetInput) or input_spec.id not in values:
            continue
        value = values[input_spec.id]
        io_type = input_spec.get_io_type()
        if io_type == "STRING":
            value = value.strip()
            minimum, maximum = _TEXT_LIMITS.get(input_spec.id, (0, None))
            validate_string(
                value,
                min_length=minimum,
                max_length=maximum,
                field_name=input_spec.id,
            )
            validated[input_spec.id] = value
        elif io_type == "COMBO" and value not in input_spec.options:
            raise ValueError(f"Invalid {input_spec.id}: {value!r}.")
        elif io_type == "BOOLEAN" and not isinstance(value, bool):
            raise ValueError(f"{input_spec.id} must be a boolean.")
        elif io_type == "INT":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{input_spec.id} must be an integer.")
            if input_spec.min is not None and value < input_spec.min:
                raise ValueError(f"{input_spec.id} must be at least {input_spec.min}.")
            if input_spec.max is not None and value > input_spec.max:
                raise ValueError(f"{input_spec.id} must be at most {input_spec.max}.")
        elif io_type == "FLOAT":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{input_spec.id} must be a finite number.")
            if input_spec.min is not None and value < input_spec.min:
                raise ValueError(f"{input_spec.id} must be at least {input_spec.min}.")
            if input_spec.max is not None and value > input_spec.max:
                raise ValueError(f"{input_spec.id} must be at most {input_spec.max}.")
            if input_spec.step:
                origin = input_spec.min or 0
                steps = (value - origin) / input_spec.step
                if not math.isclose(steps, round(steps), abs_tol=1e-7):
                    raise ValueError(f"{input_spec.id} must use increments of {input_spec.step}.")
    return validated


class _ComfyCloudWorkflowNode(IO.ComfyNode):
    workflow: ClassVar[ComfyCloudWorkflow]
    node_id: ClassVar[str]
    display_name: ClassVar[str]
    summary: ClassVar[str]
    category: ClassVar[str]
    requires_image: ClassVar[bool]
    returns_video: ClassVar[bool]

    @classmethod
    def define_schema(cls) -> IO.Schema:
        inputs = [
            IO.String.Input(
                "prompt",
                multiline=True,
                default="",
                tooltip="Describe the content to generate or the edit to apply.",
            )
        ]
        if cls.requires_image:
            inputs.append(IO.Image.Input("image"))

        output = IO.Video.Output() if cls.returns_video else IO.Image.Output()
        return IO.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            description=_comfy_cloud_description(cls.summary),
            inputs=_with_input_sockets(inputs),
            outputs=[output],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_COMFY_CLOUD_PRICE_BADGE,
        )

    @classmethod
    async def execute(cls, prompt: str, image: Input.Image | None = None) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]

        image_url = None
        if cls.requires_image:
            image_url = await cls._upload_image(image)

        return await cls._run(ComfyCloudWorkflowInputs(prompt=prompt, image_url=image_url))

    @classmethod
    async def _upload_image(cls, image: Input.Image, total_pixels: int | None = 2048 * 2048) -> str:
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        _validate_image_upload(image)
        return await upload_image_to_comfyapi(cls, image, total_pixels=total_pixels)

    @classmethod
    async def _run(cls, inputs: ComfyCloudWorkflowInputs) -> IO.NodeOutput:
        task = await sync_op(
            cls,
            _GENERATE_ENDPOINT,
            response_model=ComfyCloudGenerateResponse,
            data=ComfyCloudGenerateRequest(
                workflow=cls.workflow,
                inputs=inputs,
            ),
        )
        result = await _poll_task(cls, task.task_id)
        if not result.output_url:
            raise RuntimeError("Comfy Cloud task completed without an output URL.")

        if cls.returns_video:
            output = await download_url_to_video_output(
                _validated_output_url(result.output_url),
                timeout=_OUTPUT_DOWNLOAD_TIMEOUT,
                cls=cls,
                allow_redirects=False,
            )
        else:
            output = await download_url_to_image_tensor(
                _validated_output_url(result.output_url),
                timeout=_OUTPUT_DOWNLOAD_TIMEOUT,
                cls=cls,
                allow_redirects=False,
            )
        return IO.NodeOutput(output)


class ComfyCloudTextToImageNode(_ComfyCloudWorkflowNode):
    workflow = "text-to-image"
    node_id = "ComfyCloudTextToImageNode"
    display_name = "Comfy Cloud Text to Image"
    summary = (
        "Generates an image from a text prompt. Comfy Cloud chooses the model and moves it to "
        "a better one over time, so the graph keeps improving without you editing it."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False


class ComfyCloudTextToVideoNode(_ComfyCloudWorkflowNode):
    workflow = "text-to-video"
    node_id = "ComfyCloudTextToVideoNode"
    display_name = "Comfy Cloud Text to Video"
    summary = (
        "Generates a video from a text prompt. Comfy Cloud chooses the model and moves it to "
        "a better one over time, so the graph keeps improving without you editing it."
    )
    category = "partner/video/Comfy Cloud"
    requires_image = False
    returns_video = True


class ComfyCloudImageToVideoNode(_ComfyCloudWorkflowNode):
    workflow = "image-to-video"
    node_id = "ComfyCloudImageToVideoNode"
    display_name = "Comfy Cloud Image to Video"
    summary = (
        "Animates a still image into a video. Comfy Cloud chooses the model and moves it to a "
        "better one over time, so the graph keeps improving without you editing it."
    )
    category = "partner/video/Comfy Cloud"
    requires_image = True
    returns_video = True


class ComfyCloudImageEditNode(_ComfyCloudWorkflowNode):
    workflow = "image-edit"
    node_id = "ComfyCloudImageEditNode"
    display_name = "Comfy Cloud Image Edit"
    summary = (
        "Edits an image from a written instruction. Comfy Cloud chooses the model and moves "
        "it to a better one over time, so the graph keeps improving without you editing it."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = True
    returns_video = False


_ASPECT_RATIOS = ["1:1", "3:4", "2:3", "3:2", "4:3", "16:9", "9:16", "21:9"]
_UINT64_MAX = 0xFFFFFFFFFFFFFFFF


def _prompt_input(name: str = "prompt") -> IO.String.Input:
    return IO.String.Input(name, multiline=True, default="")


def _aspect_ratio_input() -> IO.Combo.Input:
    return IO.Combo.Input("aspect_ratio", options=_ASPECT_RATIOS, default="1:1")


def _seed_input() -> IO.Int.Input:
    return IO.Int.Input("seed", default=0, min=0, max=_UINT64_MAX, control_after_generate=True)


def _image_schema(node_id: str, display_name: str, summary: str, inputs: list[IO.Input]) -> IO.Schema:
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        category="partner/image/Comfy Cloud",
        description=_comfy_cloud_description(summary),
        inputs=_with_input_sockets(inputs),
        outputs=[IO.Image.Output()],
        hidden=[
            IO.Hidden.auth_token_comfy_org,
            IO.Hidden.api_key_comfy_org,
            IO.Hidden.unique_id,
        ],
        is_api_node=True,
        price_badge=_COMFY_CLOUD_PRICE_BADGE,
    )


class ComfyCloudCapybaraTextToImageNode(_ComfyCloudWorkflowNode):
    workflow = "image.capybara-0-1-text-to-image.v1"
    node_id = "ComfyCloudCapybaraTextToImageNode"
    display_name = "Comfy Cloud Capybara 0.1"
    summary = (
        "Generates an image from a text prompt with Capybara 0.1, rendered at 1280x1280 "
        "rather than upscaled from a smaller canvas."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            [
                _prompt_input(),
                _seed_input(),
            ],
        )

    @classmethod
    # pylint: disable=arguments-renamed
    async def execute(cls, prompt: str, seed: int = 0) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        return await cls._run(ComfyCloudWorkflowInputs(prompt=prompt, seed=seed))


class ComfyCloudIdeogram4TextToImageNode(_ComfyCloudWorkflowNode):
    workflow = "image.ideogram-4-text-to-image.v1"
    node_id = "ComfyCloudIdeogram4TextToImageNode"
    display_name = "Comfy Cloud Ideogram 4"
    summary = (
        "Generates an image from a text prompt with Ideogram 4, a model aimed at typography "
        "and graphic layout work."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            [
                _prompt_input(),
                _seed_input(),
            ],
        )

    @classmethod
    # pylint: disable=arguments-renamed
    async def execute(cls, prompt: str, seed: int = 0) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        return await cls._run(ComfyCloudWorkflowInputs(prompt=prompt, seed=seed))


class ComfyCloudLongCatTextToImageNode(_ComfyCloudWorkflowNode):
    workflow = "image.longcat-text-to-image.v1"
    node_id = "ComfyCloudLongCatTextToImageNode"
    display_name = "Comfy Cloud LongCat"
    summary = (
        "Generates a 1024x1024 image from a text prompt with LongCat, running a full 20-step "
        "sampler instead of a distilled shortcut. Slower and dearer per run than the turbo "
        "models, and steadier on fine detail."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            [
                _prompt_input(),
                _seed_input(),
            ],
        )

    @classmethod
    # pylint: disable=arguments-renamed
    async def execute(cls, prompt: str, seed: int = 0) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        return await cls._run(ComfyCloudWorkflowInputs(prompt=prompt, seed=seed))


class ComfyCloudFlux2TextToImageNode(_ComfyCloudWorkflowNode):
    workflow = "image.flux-2-text-to-image.v1"
    node_id = "ComfyCloudFlux2TextToImageNode"
    display_name = "Comfy Cloud Flux 2"
    summary = (
        "Generates an image from a text prompt with Flux 2 dev plus its Turbo LoRA, which "
        "trades a little fidelity for a much shorter run."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            [
                _prompt_input(),
                _seed_input(),
            ],
        )

    @classmethod
    # pylint: disable=arguments-renamed
    async def execute(cls, prompt: str, seed: int = 0) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        return await cls._run(ComfyCloudWorkflowInputs(prompt=prompt, seed=seed))


class ComfyCloudZImageTurboNode(_ComfyCloudWorkflowNode):
    workflow = "image.z-image-turbo.v1"
    node_id = "ComfyCloudZImageTurboNode"
    display_name = "Comfy Cloud Z-Image Turbo"
    summary = (
        "Generates a 1024x1024 image from a text prompt with Z-Image Turbo in 8 steps. One of "
        "the quickest and cheapest nodes here, which makes it the one to iterate on."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            [
                _prompt_input(),
                _seed_input(),
            ],
        )

    @classmethod
    # pylint: disable=arguments-renamed
    async def execute(cls, prompt: str, seed: int = 0) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        return await cls._run(ComfyCloudWorkflowInputs(prompt=prompt, seed=seed))


class ComfyCloudKrea2CreativeImageNode(_ComfyCloudWorkflowNode):
    workflow = "image.krea-2-creative-image.v1"
    node_id = "ComfyCloudKrea2CreativeImageNode"
    display_name = "Comfy Cloud Krea 2"
    summary = (
        "Generates an image from a text prompt with Krea 2 Turbo in 8 steps. The Krea 2 "
        "darkbrush style LoRA is baked into this graph, so output carries that look by "
        "default."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            [
                _prompt_input(),
                IO.Boolean.Input("prompt_enhance", default=True),
                _aspect_ratio_input(),
                _seed_input(),
            ],
        )

    @classmethod
    # pylint: disable=arguments-renamed
    async def execute(
        cls, prompt: str, prompt_enhance: bool = True, aspect_ratio: str = "1:1", seed: int = 0
    ) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        return await cls._run(
            ComfyCloudWorkflowInputs(
                prompt=prompt, prompt_enhance=prompt_enhance, aspect_ratio=aspect_ratio, seed=seed
            )
        )


class ComfyCloudQwenImageEdit2511Node(_ComfyCloudWorkflowNode):
    workflow = "image.qwen-image-edit-2511.v1"
    node_id = "ComfyCloudQwenImageEdit2511Node"
    display_name = "Comfy Cloud Qwen Image Edit 2511"
    summary = (
        "Edits an image from a written instruction with Qwen Image Edit 2511, cut to 4 steps "
        "by a Lightning LoRA. Describe the change you want, not the whole scene."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = True
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            [
                IO.Image.Input("image"),
                _prompt_input("instruction"),
                IO.Combo.Input("quality_mode", options=["quality", "fast"], default="quality"),
                _seed_input(),
            ],
        )

    @classmethod
    # pylint: disable=arguments-renamed
    async def execute(
        cls,
        image: Input.Image,
        instruction: str,
        quality_mode: str = "quality",
        seed: int = 0,
    ) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        instruction = values["instruction"]
        return await cls._run(
            ComfyCloudWorkflowInputs(
                assets={
                    "image": ComfyCloudAssetInput(
                        type="IMAGE", url=await cls._upload_image(image, total_pixels=None)
                    )
                },
                instruction=instruction,
                quality_mode=quality_mode,
                seed=seed,
            )
        )


class ComfyCloudSeedVR2ImageUpscaleNode(_ComfyCloudWorkflowNode):
    workflow = "image.seedvr2-image-upscale.v1"
    node_id = "ComfyCloudSeedVR2ImageUpscaleNode"
    display_name = "Comfy Cloud SeedVR2 Upscale"
    summary = (
        "Upscales and restores an image with the SeedVR2 7B diffusion upscaler in a single "
        "step. It rebuilds detail rather than resampling, so it suits soft or heavily "
        "compressed sources."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = True
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            [IO.Image.Input("image"), IO.Combo.Input("scale", options=["2x", "4x"], default="4x")],
        )

    @classmethod
    # pylint: disable=arguments-renamed
    async def execute(cls, image: Input.Image, scale: str = "4x") -> IO.NodeOutput:
        _validate_node_inputs(cls, locals())
        return await cls._run(
            ComfyCloudWorkflowInputs(
                assets={
                    "image": ComfyCloudAssetInput(
                        type="IMAGE", url=await cls._upload_image(image, total_pixels=None)
                    )
                },
                scale=scale,
            )
        )


async def _run_video_workflow(cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs) -> IO.NodeOutput:
    task = await sync_op(cls, _GENERATE_ENDPOINT, response_model=ComfyCloudGenerateResponse, data=ComfyCloudGenerateRequest(workflow=workflow, inputs=inputs))
    result = await _poll_task(cls, task.task_id)
    if not result.output_url:
        raise RuntimeError("Comfy Cloud task completed without an output URL.")
    return IO.NodeOutput(
        await download_url_to_video_output(
            _validated_output_url(result.output_url),
            timeout=_OUTPUT_DOWNLOAD_TIMEOUT,
            cls=cls,
            allow_redirects=False,
        )
    )


def _video_schema(node_id: str, display_name: str, summary: str, inputs: list[IO.Input]) -> IO.Schema:
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        category="partner/video/Comfy Cloud",
        description=_comfy_cloud_description(summary),
        inputs=_with_input_sockets(inputs),
        outputs=[IO.Video.Output()],
        hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
        is_api_node=True,
        price_badge=_COMFY_CLOUD_PRICE_BADGE,
    )


def _video_seed_input(default: int) -> IO.Int.Input:
    return IO.Int.Input("seed", default=default, min=0, max=_UINT64_MAX, control_after_generate=True)


class ComfyCloudMiniMaxH3TextSoundNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudMiniMaxH3TextSoundNode",
            "Comfy Cloud MiniMax H3 Text to Video",
            (
                "Generates a video with a matching soundtrack from a text prompt, using MiniMax "
                "H3. Picture and audio come out of the same pass rather than being dubbed on "
                "afterwards."
            ),
            [
                _prompt_input(),
                IO.Combo.Input("aspect_ratio", options=["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"], default="1:1"),
                IO.Float.Input("duration_seconds", default=5, min=5, max=15, step=0.01),
                _video_seed_input(168866841893410),
            ],
        )

    @classmethod
    async def execute(cls, prompt: str, aspect_ratio: str, duration_seconds: float, seed: int) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        return await _run_video_workflow(cls, "video.minimax-h3-text-sound.v1", ComfyCloudWorkflowInputs(prompt=prompt, aspect_ratio=aspect_ratio, duration_seconds=duration_seconds, seed=seed))


class ComfyCloudMiniMaxH3ImageSoundNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudMiniMaxH3ImageSoundNode",
            "Comfy Cloud MiniMax H3 Image to Video",
            (
                "Animates a still image into a video with a matching soundtrack, using MiniMax "
                "H3. Picture and audio come out of the same pass rather than being dubbed on "
                "afterwards."
            ),
            [
                IO.Image.Input("image"),
                _prompt_input(),
                IO.Combo.Input("aspect_ratio", options=["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"], default="1:1"),
                IO.Float.Input("duration_seconds", default=5, min=5, max=15, step=0.01),
                _video_seed_input(168866841893410),
            ],
        )

    @classmethod
    async def execute(cls, image: Input.Image, prompt: str, aspect_ratio: str, duration_seconds: float, seed: int) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        _validate_image_upload(image)
        image_url = await upload_image_to_comfyapi(cls, image)
        return await _run_video_workflow(cls, "video.minimax-h3-image-sound.v1", ComfyCloudWorkflowInputs(prompt=prompt, image_url=image_url, aspect_ratio=aspect_ratio, duration_seconds=duration_seconds, seed=seed))


class ComfyCloudLTX23ImageAudioPerformanceNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudLTX23ImageAudioPerformanceNode",
            "Comfy Cloud LTX-2.3 Performance",
            (
                "Drives a still image with an audio track to produce an audio-led performance "
                "video, using LTX-2.3 22B with a 2x spatial upscaler. Duration cannot exceed the "
                "length of the audio you supply."
            ),
            [
                IO.Image.Input("image"), IO.Audio.Input("audio"), _prompt_input(),
                IO.Boolean.Input("enhance_prompt", default=True),
                IO.Float.Input("duration_seconds", default=9, min=1, max=15, step=0.01, tooltip="Must not exceed the input audio duration."),
                _video_seed_input(225158785956033),
            ],
        )

    @classmethod
    async def execute(cls, image: Input.Image, audio: Input.Audio, prompt: str, enhance_prompt: bool, duration_seconds: float, seed: int) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        _validate_image_upload(image)
        _validate_audio_upload(audio)
        audio_duration = _audio_duration(audio)
        if duration_seconds - min(1 / float(audio["sample_rate"]), 1e-3) > audio_duration:
            raise ValueError(f"Duration ({duration_seconds:g}s) exceeds input audio duration ({audio_duration:.2f}s).")
        image_url = await upload_image_to_comfyapi(cls, image)
        audio_url = await upload_audio_to_comfyapi(cls, audio)
        return await _run_video_workflow(cls, "video.ltx-2-3-image-audio-performance.v1", ComfyCloudWorkflowInputs(prompt=prompt, image_url=image_url, audio_url=audio_url, enhance_prompt=enhance_prompt, duration_seconds=duration_seconds, seed=seed))


class ComfyCloudWan22FirstLastFrameNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudWan22FirstLastFrameNode",
            "Comfy Cloud Wan 2.2 First & Last Frame",
            (
                "Generates the video between a first and a last frame with Wan 2.2 14B, inventing "
                "the motion that connects them. Frames that differ wildly need a longer duration "
                "to transition cleanly."
            ),
            [IO.Image.Input("first_frame"), IO.Image.Input("last_frame"), _prompt_input(), IO.String.Input("negative_prompt", multiline=True, default="graph tested Chinese quality negative"), IO.Int.Input("duration_seconds", default=5, min=2, max=8, step=1, tooltip="Graph frame count is floor(duration × 16 + 1)."), _video_seed_input(984937593540091)],
        )

    @classmethod
    async def execute(cls, first_frame: Input.Image, last_frame: Input.Image, prompt: str, negative_prompt: str, duration_seconds: int, seed: int) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        negative_prompt = values["negative_prompt"]
        if get_number_of_images(first_frame) != 1 or get_number_of_images(last_frame) != 1:
            raise ValueError("Exactly one first frame and one last frame are required.")
        _validate_image_upload(first_frame)
        _validate_image_upload(last_frame)
        first_url = await upload_image_to_comfyapi(cls, first_frame, wait_label="Uploading first frame")
        last_url = await upload_image_to_comfyapi(cls, last_frame, wait_label="Uploading last frame")
        return await _run_video_workflow(cls, "video.wan-2-2-14b-first-last-frame.v1", ComfyCloudWorkflowInputs(prompt=prompt, negative_prompt=negative_prompt, first_frame_url=first_url, last_frame_url=last_url, duration_seconds=duration_seconds, seed=seed))


_UINT32_MAX = 0xFFFFFFFF


def _audio_duration(audio: Input.Audio) -> float:
    sample_rate = float(audio["sample_rate"])
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Audio sample rate must be a positive number.")
    return audio["waveform"].shape[-1] / sample_rate


class ComfyCloudExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            # Capability nodes. The backend maps each alias to a curated workflow
            # (text-to-video -> MiniMax H3, image-to-video -> LTX-2.3, image-edit ->
            # Qwen-Image-Edit-2511), so the model behind a shape can be swapped
            # server-side without a ComfyUI release. Stable class_type per output shape.
            ComfyCloudTextToImageNode,
            ComfyCloudTextToVideoNode,
            ComfyCloudImageToVideoNode,
            ComfyCloudImageEditNode,
            # Named workflows, for control the generic shape cannot express.
            ComfyCloudMiniMaxH3TextSoundNode,
            ComfyCloudMiniMaxH3ImageSoundNode,
            ComfyCloudWan22FirstLastFrameNode,
            ComfyCloudLTX23ImageAudioPerformanceNode,
            ComfyCloudFlux2TextToImageNode,
            ComfyCloudIdeogram4TextToImageNode,
            ComfyCloudLongCatTextToImageNode,
            ComfyCloudCapybaraTextToImageNode,
            ComfyCloudZImageTurboNode,
            ComfyCloudKrea2CreativeImageNode,
            ComfyCloudQwenImageEdit2511Node,
            ComfyCloudSeedVR2ImageUpscaleNode,
        ]


async def comfy_entrypoint() -> ComfyCloudExtension:
    return ComfyCloudExtension()

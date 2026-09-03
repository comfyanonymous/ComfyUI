import contextlib
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


# Must stay in step with comfyCloudOutputBuckets in cloud's services/comfy-api/config/config.go.
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
_RUN_TIMEOUT_SECONDS = 2100
_POLL_INTERVAL_SECONDS = 5.0
_POLL_MAX_ATTEMPTS = int(_RUN_TIMEOUT_SECONDS / _POLL_INTERVAL_SECONDS) + 24  # +2 min of slack
_OUTPUT_DOWNLOAD_TIMEOUT = 30 * 60
_MAX_UPLOAD_IMAGE_PIXELS = 32_000_000
_MAX_UPLOAD_IMAGE_DIMENSION = 8192
_MAX_DECODED_AUDIO_BYTES = 256 * 1024 * 1024
# Hardcoded mirror of Metronome's rtx_pro_6000 rate. Nothing links the two, so it
# must be changed by hand when the rate card moves.
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


def _cloud_schema(
    node_id: str,
    display_name: str,
    summary: str,
    category: str,
    inputs: list[IO.Input],
    output: IO.Output,
) -> IO.Schema:
    """Every Comfy Cloud node is the same schema but for its id, name, blurb, category,
    inputs and output type, so they are all built here."""
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        category=category,
        description=_comfy_cloud_description(summary),
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


def _validated_output_url(url: str) -> str:
    parsed = urlsplit(url)
    decoded_path = unquote(parsed.path)
    is_proxy_path = (
        not parsed.scheme
        and not parsed.netloc
        and decoded_path.startswith("/proxy/comfy-cloud/")
        and posixpath.normpath(decoded_path) == decoded_path
    )
    # normpath first: a client resolves dot segments before sending, so
    # ".../comfy-cloud-assets/../other/x.png" would advertise an allowed bucket here
    # and fetch from another one on the wire.
    bucket = decoded_path.lstrip("/").split("/", 1)[0]
    is_signed_https_url = (
        parsed.scheme == "https"
        and parsed.hostname == "storage.googleapis.com"
        and parsed.port is None
        and parsed.username is None
        and parsed.password is None
        and posixpath.normpath(decoded_path) == decoded_path
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
        with contextlib.suppress(Exception):
            await sync_op_raw(cls, cancel_endpoint, max_retries=0)
        raise


def _validate_node_inputs(cls: type[IO.ComfyNode], values: dict) -> dict:
    validated = dict(values)
    for input_spec in cls.define_schema().inputs:
        if not isinstance(input_spec, IO.WidgetInput) or input_spec.id not in values:
            continue
        value = values[input_spec.id]
        io_type = input_spec.get_io_type()
        if io_type == "STRING":
            if not isinstance(value, str):
                raise ValueError(f"{input_spec.id} must be a string.")
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


_ASPECT_RATIOS = ["1:1", "3:4", "2:3", "3:2", "4:3", "16:9", "9:16", "21:9"]
# One megapixel-class render per ratio. Every side is a multiple of 64, so a
# pipeline that renders larger still lands on the 16-pixel grid latents want.
_ASPECT_DIMENSIONS = {
    "1:1": (1024, 1024),
    "3:4": (768, 1024),
    "4:3": (1024, 768),
    "2:3": (768, 1152),
    "3:2": (1152, 768),
    "9:16": (768, 1344),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
}
_VIDEO_RESOLUTIONS = ["480p", "720p"]
_LTX_RESOLUTIONS = ["1280x720", "960x960", "720x1280"]
_UINT64_MAX = 0xFFFFFFFFFFFFFFFF
_NEGATIVE_PROMPT_TOOLTIP = "Leave empty to keep the negative prompt this pipeline was tuned with."


def _dimensions(aspect_ratio: str, scale: float = 1.0) -> tuple[int, int]:
    width, height = _ASPECT_DIMENSIONS[aspect_ratio]
    return round(width * scale), round(height * scale)


def _prompt_input(name: str = "prompt") -> IO.String.Input:
    return IO.String.Input(name, multiline=True, default="")


def _negative_prompt_input() -> IO.String.Input:
    return IO.String.Input(
        "negative_prompt", multiline=True, default="", advanced=True, tooltip=_NEGATIVE_PROMPT_TOOLTIP
    )


def _aspect_ratio_input(default: str = "1:1") -> IO.Combo.Input:
    return IO.Combo.Input("aspect_ratio", options=_ASPECT_RATIOS, default=default)


def _seed_input() -> IO.Int.Input:
    return IO.Int.Input("seed", default=42, min=0, max=_UINT64_MAX, control_after_generate=True)


def _steps_input(default: int, maximum: int, name: str = "steps", tooltip: str | None = None) -> IO.Int.Input:
    return IO.Int.Input(name, default=default, min=1, max=maximum, advanced=True, tooltip=tooltip)


def _tuning_input(
    name: str, default: float, maximum: float, step: float = 0.1, tooltip: str | None = None
) -> IO.Float.Input:
    return IO.Float.Input(
        name, default=default, min=0.0, max=maximum, step=step, advanced=True, tooltip=tooltip
    )


def _video_resolution_input() -> IO.Combo.Input:
    return IO.Combo.Input(
        "resolution",
        options=_VIDEO_RESOLUTIONS,
        default="480p",
        advanced=True,
        tooltip="Frame size budget. 720p costs roughly twice the GPU-seconds of 480p.",
    )


async def _submit_workflow(
    cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs
) -> str:
    task = await sync_op(
        cls,
        _GENERATE_ENDPOINT,
        response_model=ComfyCloudGenerateResponse,
        data=ComfyCloudGenerateRequest(workflow=workflow, inputs=inputs),
    )
    result = await _poll_task(cls, task.task_id)
    if not result.output_url:
        raise RuntimeError("Comfy Cloud task completed without an output URL.")
    return _validated_output_url(result.output_url)


async def _run_image_workflow(
    cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs
) -> IO.NodeOutput:
    url = await _submit_workflow(cls, workflow, inputs)
    return IO.NodeOutput(
        await download_url_to_image_tensor(
            url, timeout=_OUTPUT_DOWNLOAD_TIMEOUT, cls=cls, allow_redirects=False
        )
    )


async def _run_video_workflow(
    cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs
) -> IO.NodeOutput:
    url = await _submit_workflow(cls, workflow, inputs)
    return IO.NodeOutput(
        await download_url_to_video_output(
            url, timeout=_OUTPUT_DOWNLOAD_TIMEOUT, cls=cls, allow_redirects=False
        )
    )


async def _upload_workflow_image(cls: type[IO.ComfyNode], image: Input.Image, **options) -> str:
    if get_number_of_images(image) != 1:
        raise ValueError("Exactly one input image is required.")
    _validate_image_upload(image)
    return await upload_image_to_comfyapi(cls, image, **options)


def _image_asset(url: str) -> dict[str, ComfyCloudAssetInput]:
    return {"image": ComfyCloudAssetInput(type="IMAGE", url=url)}


def _image_schema(cls: type[IO.ComfyNode], inputs: list[IO.Input]) -> IO.Schema:
    return _cloud_schema(
        cls.node_id, cls.display_name, cls.summary, "partner/image/Comfy Cloud", inputs, IO.Image.Output()
    )


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
        inputs.append(_seed_input())

        return _cloud_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            cls.category,
            inputs,
            IO.Video.Output() if cls.returns_video else IO.Image.Output(),
        )

    @classmethod
    async def execute(cls, prompt: str, image: Input.Image | None = None, seed: int = 42) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]

        image_url = None
        if cls.requires_image:
            image_url = await _upload_workflow_image(cls, image, total_pixels=2048 * 2048)

        return await cls._run(ComfyCloudWorkflowInputs(prompt=prompt, image_url=image_url, seed=seed))

    @classmethod
    async def _run(cls, inputs: ComfyCloudWorkflowInputs) -> IO.NodeOutput:
        run = _run_video_workflow if cls.returns_video else _run_image_workflow
        return await run(cls, cls.workflow, inputs)


class ComfyCloudTextToImageNode(_ComfyCloudWorkflowNode):
    workflow = "default/text-to-image"
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
    workflow = "default/text-to-video"
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
    workflow = "default/image-to-video"
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
    workflow = "default/image-edit"
    node_id = "ComfyCloudImageEditNode"
    display_name = "Comfy Cloud Image Edit"
    summary = (
        "Edits an image from a written instruction. Comfy Cloud chooses the model and moves "
        "it to a better one over time, so the graph keeps improving without you editing it."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = True
    returns_video = False


class ComfyCloudFlux2TextToImageNode(IO.ComfyNode):
    node_id = "ComfyCloudFlux2TextToImageNode"
    display_name = "Comfy Cloud Flux 2 Text to Image"
    summary = (
        "Generates an image from a text prompt with Flux 2 dev. Turbo swaps in the Flux 2 Turbo "
        "LoRA and a short schedule, trading a little fidelity for a much quicker run; switch it "
        "off for the full-length dev pass."
    )

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls,
            [
                _prompt_input(),
                _aspect_ratio_input(),
                IO.Boolean.Input(
                    "turbo",
                    default=True,
                    tooltip="Run the Turbo LoRA at turbo_steps instead of the full dev pass.",
                ),
                _seed_input(),
                _steps_input(20, 100, tooltip="Steps for the full dev pass, used when turbo is off."),
                _steps_input(8, 50, name="turbo_steps", tooltip="Steps for the Turbo LoRA pass."),
                _tuning_input("turbo_strength", 1.0, 2.0, step=0.05),
                _tuning_input("guidance", 4.0, 20.0),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str = "1:1",
        turbo: bool = True,
        seed: int = 42,
        steps: int = 20,
        turbo_steps: int = 8,
        turbo_strength: float = 1.0,
        guidance: float = 4.0,
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        width, height = _dimensions(aspect_ratio)
        return await _run_image_workflow(
            cls,
            "flux-2/text-to-image",
            ComfyCloudWorkflowInputs(
                prompt=prompt, width=width, height=height, turbo=turbo, seed=seed, steps=steps,
                turbo_steps=turbo_steps, turbo_strength=turbo_strength, guidance=guidance,
            ),
        )


class ComfyCloudIdeogram4TextToImageNode(IO.ComfyNode):
    node_id = "ComfyCloudIdeogram4TextToImageNode"
    display_name = "Comfy Cloud Ideogram 4 Text to Image"
    summary = (
        "Generates an image from a text prompt with Ideogram 4, a model aimed at typography and "
        "graphic layout work. Rendering speed picks a step count and sigma schedule together."
    )

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls,
            [
                _prompt_input(),
                _aspect_ratio_input(),
                IO.Combo.Input(
                    "rendering_speed",
                    options=["turbo", "default", "quality"],
                    default="default",
                    tooltip="Ideogram's own presets: turbo is 12 steps, default 20, quality 48.",
                ),
                _seed_input(),
                _tuning_input("guidance", 7.0, 20.0),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str = "1:1",
        rendering_speed: str = "default",
        seed: int = 42,
        guidance: float = 7.0,
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        width, height = _dimensions(aspect_ratio)
        return await _run_image_workflow(
            cls,
            "ideogram-4/text-to-image",
            ComfyCloudWorkflowInputs(
                prompt=prompt, width=width, height=height, rendering_speed=rendering_speed,
                seed=seed, guidance=guidance,
            ),
        )


class ComfyCloudLongCatTextToImageNode(IO.ComfyNode):
    node_id = "ComfyCloudLongCatTextToImageNode"
    display_name = "Comfy Cloud LongCat Text to Image"
    summary = (
        "Generates an image from a text prompt with LongCat, running a full 20-step sampler "
        "instead of a distilled shortcut. Slower and dearer per run than the turbo models, and "
        "steadier on fine detail."
    )

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls,
            [
                _prompt_input(),
                _aspect_ratio_input(),
                _seed_input(),
                _negative_prompt_input(),
                _steps_input(20, 100),
                _tuning_input("cfg", 4.0, 20.0),
                _tuning_input("guidance", 4.0, 20.0),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str = "1:1",
        seed: int = 42,
        negative_prompt: str = "",
        steps: int = 20,
        cfg: float = 4.0,
        guidance: float = 4.0,
    ) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        width, height = _dimensions(aspect_ratio)
        return await _run_image_workflow(
            cls,
            "longcat/text-to-image",
            ComfyCloudWorkflowInputs(
                prompt=values["prompt"], negative_prompt=values["negative_prompt"] or None,
                width=width, height=height, seed=seed, steps=steps, cfg=cfg, guidance=guidance,
            ),
        )


class ComfyCloudCapybaraTextToImageNode(IO.ComfyNode):
    node_id = "ComfyCloudCapybaraTextToImageNode"
    display_name = "Comfy Cloud Capybara 0.1 Text to Image"
    summary = (
        "Generates an image from a text prompt with Capybara 0.1, rendered at its native 1280 "
        "pixel class rather than upscaled from a smaller canvas."
    )

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls,
            [
                _prompt_input(),
                _aspect_ratio_input(),
                _seed_input(),
                _negative_prompt_input(),
                _steps_input(20, 100),
                _tuning_input("cfg", 6.0, 20.0),
                _tuning_input("shift", 7.0, 20.0),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str = "1:1",
        seed: int = 42,
        negative_prompt: str = "",
        steps: int = 20,
        cfg: float = 6.0,
        shift: float = 7.0,
    ) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        width, height = _dimensions(aspect_ratio, scale=1.25)
        return await _run_image_workflow(
            cls,
            "capybara-0.1/text-to-image",
            ComfyCloudWorkflowInputs(
                prompt=values["prompt"], negative_prompt=values["negative_prompt"] or None,
                width=width, height=height, seed=seed, steps=steps, cfg=cfg, shift=shift,
            ),
        )


class ComfyCloudZImageTurboNode(IO.ComfyNode):
    node_id = "ComfyCloudZImageTurboNode"
    display_name = "Comfy Cloud Z-Image Turbo Text to Image"
    summary = (
        "Generates an image from a text prompt with Z-Image Turbo in 8 steps. One of the "
        "quickest and cheapest nodes here, which makes it the one to iterate on."
    )

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls,
            [
                _prompt_input(),
                _aspect_ratio_input(),
                _seed_input(),
                _steps_input(8, 50),
                _tuning_input("shift", 3.0, 20.0),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str = "1:1",
        seed: int = 42,
        steps: int = 8,
        shift: float = 3.0,
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        width, height = _dimensions(aspect_ratio)
        return await _run_image_workflow(
            cls,
            "z-image-turbo/text-to-image",
            ComfyCloudWorkflowInputs(
                prompt=prompt, width=width, height=height, seed=seed, steps=steps, shift=shift
            ),
        )


class ComfyCloudKrea2CreativeImageNode(IO.ComfyNode):
    node_id = "ComfyCloudKrea2CreativeImageNode"
    display_name = "Comfy Cloud Krea 2 Text to Image"
    summary = (
        "Generates an image from a text prompt with Krea 2 Turbo in 8 steps. Style lora adds the "
        "Krea 2 darkbrush look and its trigger word to the prompt."
    )

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls,
            [
                _prompt_input(),
                IO.Boolean.Input("prompt_enhance", default=True),
                _aspect_ratio_input(),
                _seed_input(),
                IO.Boolean.Input(
                    "style_lora",
                    default=False,
                    advanced=True,
                    tooltip="Load the darkbrush LoRA and append its trigger word to the prompt.",
                ),
                _tuning_input("style_strength", 0.8, 2.0, step=0.05),
                _steps_input(8, 50),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        prompt_enhance: bool = True,
        aspect_ratio: str = "1:1",
        seed: int = 42,
        style_lora: bool = False,
        style_strength: float = 0.8,
        steps: int = 8,
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        return await _run_image_workflow(
            cls,
            "krea-2/text-to-image",
            ComfyCloudWorkflowInputs(
                prompt=prompt, prompt_enhance=prompt_enhance, aspect_ratio=aspect_ratio, seed=seed,
                style_lora=style_lora, style_strength=style_strength, steps=steps,
            ),
        )


class ComfyCloudQwenImageEdit2511Node(IO.ComfyNode):
    node_id = "ComfyCloudQwenImageEdit2511Node"
    display_name = "Comfy Cloud Qwen Image Edit 2511"
    summary = (
        "Edits an image from a written instruction with Qwen Image Edit 2511. Fast cuts the run "
        "to a few steps with a Lightning LoRA. Describe the change you want, not the whole scene."
    )

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls,
            [
                IO.Image.Input("image"),
                _prompt_input("instruction"),
                IO.Combo.Input("quality_mode", options=["quality", "fast"], default="quality"),
                _seed_input(),
                _negative_prompt_input(),
                _steps_input(40, 100, tooltip="Steps in quality mode."),
                _steps_input(4, 20, name="fast_steps", tooltip="Steps in fast mode."),
                _tuning_input("cfg", 4.0, 20.0, tooltip="Guidance in quality mode; fast mode is fixed at 1."),
                _tuning_input("shift", 3.1, 20.0),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        instruction: str,
        quality_mode: str = "quality",
        seed: int = 42,
        negative_prompt: str = "",
        steps: int = 40,
        fast_steps: int = 4,
        cfg: float = 4.0,
        shift: float = 3.1,
    ) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        return await _run_image_workflow(
            cls,
            "qwen-image-edit-2511/image-edit",
            ComfyCloudWorkflowInputs(
                assets=_image_asset(await _upload_workflow_image(cls, image, total_pixels=None)),
                instruction=values["instruction"],
                negative_prompt=values["negative_prompt"] or None,
                quality_mode=quality_mode, seed=seed, steps=steps, fast_steps=fast_steps,
                cfg=cfg, shift=shift,
            ),
        )


class ComfyCloudSeedVR2ImageUpscaleNode(IO.ComfyNode):
    node_id = "ComfyCloudSeedVR2ImageUpscaleNode"
    display_name = "Comfy Cloud SeedVR2 Upscale Image"
    summary = (
        "Upscales and restores an image with the SeedVR2 7B diffusion upscaler in a single step. "
        "It rebuilds detail rather than resampling, so it suits soft or heavily compressed sources."
    )

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls,
            [
                IO.Image.Input("image"),
                IO.Combo.Input("scale", options=["2x", "4x"], default="4x"),
                _seed_input(),
                IO.Combo.Input(
                    "color_correction",
                    options=["none", "lab", "wavelet", "adain"],
                    default="none",
                    advanced=True,
                    tooltip="Match the restored colours back to the source. lab is the most faithful.",
                ),
                _steps_input(1, 10),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        scale: str = "4x",
        seed: int = 42,
        color_correction: str = "none",
        steps: int = 1,
    ) -> IO.NodeOutput:
        _validate_node_inputs(cls, locals())
        return await _run_image_workflow(
            cls,
            "seedvr2/upscale-image",
            ComfyCloudWorkflowInputs(
                assets=_image_asset(await _upload_workflow_image(cls, image, total_pixels=None)),
                scale=scale, seed=seed, color_correction=color_correction, steps=steps,
            ),
        )


def _video_schema(node_id: str, display_name: str, summary: str, inputs: list[IO.Input]) -> IO.Schema:
    return _cloud_schema(
        node_id, display_name, summary, "partner/video/Comfy Cloud", inputs, IO.Video.Output()
    )


def _minimax_inputs(image: bool) -> list[IO.Input]:
    inputs = [IO.Image.Input("image")] if image else []
    return inputs + [
        _prompt_input(),
        _aspect_ratio_input("1:1" if image else "16:9"),
        IO.Float.Input("duration_seconds", default=5, min=5, max=15, step=0.01),
        _seed_input(),
        _video_resolution_input(),
        _steps_input(20, 60),
    ]


class ComfyCloudMiniMaxH3TextSoundNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudMiniMaxH3TextSoundNode",
            "Comfy Cloud MiniMax H3 Text to Video with Audio",
            (
                "Generates a video with a matching soundtrack from a text prompt, using MiniMax "
                "H3. Picture and audio come out of the same pass rather than being dubbed on "
                "afterwards."
            ),
            _minimax_inputs(image=False),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str = "16:9",
        duration_seconds: float = 5,
        seed: int = 42,
        resolution: str = "480p",
        steps: int = 20,
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        inputs = ComfyCloudWorkflowInputs(
            prompt=prompt, aspect_ratio=aspect_ratio, duration_seconds=duration_seconds,
            seed=seed, resolution=resolution, steps=steps,
        )
        return await _run_video_workflow(cls, "minimax-h3/text-to-video", inputs)


class ComfyCloudMiniMaxH3ImageSoundNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudMiniMaxH3ImageSoundNode",
            "Comfy Cloud MiniMax H3 Image to Video with Audio",
            (
                "Animates a still image into a video with a matching soundtrack, using MiniMax "
                "H3. Picture and audio come out of the same pass rather than being dubbed on "
                "afterwards."
            ),
            _minimax_inputs(image=True),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        prompt: str,
        aspect_ratio: str = "1:1",
        duration_seconds: float = 5,
        seed: int = 42,
        resolution: str = "480p",
        steps: int = 20,
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        image_url = await _upload_workflow_image(cls, image)
        inputs = ComfyCloudWorkflowInputs(
            prompt=prompt, image_url=image_url, aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds, seed=seed, resolution=resolution, steps=steps,
        )
        return await _run_video_workflow(cls, "minimax-h3/image-to-video", inputs)


class ComfyCloudLTX23ImageAudioPerformanceNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudLTX23ImageAudioPerformanceNode",
            "Comfy Cloud LTX-2.3 Image to Video",
            (
                "Drives a still image with an audio track to produce an audio-led performance "
                "video, using LTX-2.3 22B with a 2x spatial upscaler. Duration cannot exceed the "
                "length of the audio you supply."
            ),
            [
                IO.Image.Input("image"),
                IO.Audio.Input("audio"),
                _prompt_input(),
                IO.Boolean.Input("enhance_prompt", default=True),
                IO.Float.Input(
                    "duration_seconds",
                    default=9,
                    min=1,
                    max=15,
                    step=0.01,
                    tooltip="Must not exceed the input audio duration.",
                ),
                _seed_input(),
                _negative_prompt_input(),
                IO.Combo.Input(
                    "resolution", options=_LTX_RESOLUTIONS, default="1280x720", advanced=True
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        audio: Input.Audio,
        prompt: str,
        enhance_prompt: bool = True,
        duration_seconds: float = 9,
        seed: int = 42,
        negative_prompt: str = "",
        resolution: str = "1280x720",
    ) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        _validate_audio_upload(audio)
        audio_duration = _audio_duration(audio)
        if duration_seconds - min(1 / float(audio["sample_rate"]), 1e-3) > audio_duration:
            raise ValueError(
                f"Duration ({duration_seconds:g}s) exceeds input audio duration ({audio_duration:.2f}s)."
            )
        image_url = await _upload_workflow_image(cls, image)
        audio_url = await upload_audio_to_comfyapi(cls, audio)
        width, height = (int(side) for side in resolution.split("x"))
        inputs = ComfyCloudWorkflowInputs(
            prompt=values["prompt"], negative_prompt=values["negative_prompt"] or None,
            image_url=image_url, audio_url=audio_url, enhance_prompt=enhance_prompt,
            duration_seconds=duration_seconds, seed=seed, width=width, height=height,
        )
        return await _run_video_workflow(cls, "ltx-2.3/image-to-video", inputs)


class ComfyCloudWan22FirstLastFrameNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudWan22FirstLastFrameNode",
            "Comfy Cloud Wan 2.2 First-Last-Frame to Video",
            (
                "Generates the video between a first and a last frame with Wan 2.2 14B, inventing "
                "the motion that connects them. Frames that differ wildly need a longer duration "
                "to transition cleanly."
            ),
            [
                IO.Image.Input("first_frame"),
                IO.Image.Input("last_frame"),
                _prompt_input(),
                IO.Int.Input(
                    "duration_seconds",
                    default=5,
                    min=2,
                    max=8,
                    step=1,
                    tooltip="Graph frame count is floor(duration × 16 + 1).",
                ),
                _seed_input(),
                _negative_prompt_input(),
                _video_resolution_input(),
                _steps_input(20, 60, tooltip="Split evenly between the high-noise and low-noise experts."),
                _tuning_input("cfg", 4.0, 20.0),
                _tuning_input("shift", 8.0, 20.0),
            ],
        )

    @classmethod
    async def execute(
        cls,
        first_frame: Input.Image,
        last_frame: Input.Image,
        prompt: str,
        duration_seconds: int = 5,
        seed: int = 42,
        negative_prompt: str = "",
        resolution: str = "480p",
        steps: int = 20,
        cfg: float = 4.0,
        shift: float = 8.0,
    ) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        first_url = await _upload_workflow_image(cls, first_frame, wait_label="Uploading first frame")
        last_url = await _upload_workflow_image(cls, last_frame, wait_label="Uploading last frame")
        # Omitted, not "", so the backend applies the graph's own negative prompt.
        inputs = ComfyCloudWorkflowInputs(
            prompt=values["prompt"], negative_prompt=values["negative_prompt"] or None,
            first_frame_url=first_url, last_frame_url=last_url, duration_seconds=duration_seconds,
            seed=seed, resolution=resolution, steps=steps, cfg=cfg, shift=shift,
        )
        return await _run_video_workflow(cls, "wan-2.2/first-last-frame-to-video", inputs)


def _audio_duration(audio: Input.Audio) -> float:
    sample_rate = float(audio["sample_rate"])
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Audio sample rate must be a positive number.")
    return audio["waveform"].shape[-1] / sample_rate


class ComfyCloudExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ComfyCloudTextToImageNode,
            ComfyCloudTextToVideoNode,
            ComfyCloudImageToVideoNode,
            ComfyCloudImageEditNode,
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

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
# Weight pickers. Keys, not filenames: cloud holds the file each key maps to.
_Z_IMAGE_MODELS = ["bf16", "int8", "nvfp4"]
_FLUX2_MODELS = ["dev-fp8", "dev-bf16"]
_FLUX2_LORAS = [
    "turbo", "turbo-v2", "analog-film", "berthe-morisot", "boring-reality",
    "chatgpt-4o", "detailed-portraits", "manga-posters", "neo-victorian",
    "soares", "spy-world-50s", "ultrareal",
]
# The default/* pickers key on the TRADE-OFF rather than the model, because those
# ids are pointers: the model behind one is re-pointed over time and the id does
# not change. A saved graph stores the key, so a key that named a model would
# break every saved graph the day the pointer moved.
_DEFAULT_MODELS = ["balanced", "quality"]
_DEFAULT_EDIT_MODELS = ["balanced", "quality", "fast"]
_DEFAULT_LORAS = ["balanced", "fast"]
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


def _weights_input(name: str, options: list[str], tooltip: str) -> IO.Combo.Input:
    return IO.Combo.Input(name, options=options, default=options[0], advanced=True, tooltip=tooltip)


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
    # Whatever the pipeline behind this pointer happens to expose. Empty means the
    # graph has no equivalent control, not that one was left off.
    turbo_tooltip: ClassVar[str] = ""
    model_options: ClassVar[list[str]] = []
    model_tooltip: ClassVar[str] = ""
    lora_options: ClassVar[list[str]] = []
    lora_tooltip: ClassVar[str] = ""

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
        if cls.turbo_tooltip:
            inputs.append(IO.Boolean.Input("turbo", default=False, tooltip=cls.turbo_tooltip))
        inputs.append(_seed_input())
        if cls.model_options:
            inputs.append(_weights_input("model", cls.model_options, cls.model_tooltip))
        if cls.lora_options:
            inputs.append(_weights_input("lora", cls.lora_options, cls.lora_tooltip))

        return _cloud_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            cls.category,
            inputs,
            IO.Video.Output() if cls.returns_video else IO.Image.Output(),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        image: Input.Image | None = None,
        seed: int = 42,
        turbo: bool = False,
        model: str = "balanced",
        lora: str = "balanced",
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]

        image_url = None
        if cls.requires_image:
            image_url = await _upload_workflow_image(cls, image, total_pixels=2048 * 2048)

        # Send only what this node declares: cloud rejects an input the pipeline
        # behind the id has no binding for.
        controls = {}
        if cls.turbo_tooltip:
            controls["turbo"] = turbo
        if cls.model_options:
            controls["model"] = model
        if cls.lora_options:
            controls["lora"] = lora
        return await cls._run(
            ComfyCloudWorkflowInputs(prompt=prompt, image_url=image_url, seed=seed, **controls)
        )

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
        "a better one over time, so the graph keeps improving without you editing it. Turbo "
        "trades a little fidelity for a run around ten times quicker."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False
    turbo_tooltip = (
        "Run the short accelerated schedule instead of the full one. Around ten times "
        "quicker and correspondingly cheaper, for a small loss of detail."
    )
    model_options = _DEFAULT_MODELS
    model_tooltip = (
        "How much precision to spend on the weights. balanced is what this pipeline ships "
        "with; quality is the full-range weights and costs more GPU-seconds."
    )
    lora_options = _DEFAULT_LORAS
    lora_tooltip = (
        "Which accelerator the turbo pass uses; it has no effect while turbo is off. "
        "balanced is a four-step distillation, fast a two-step one."
    )


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
        "it to a better one over time, so the graph keeps improving without you editing it. "
        "Turbo trades fidelity for a run around seven times quicker."
    )
    category = "partner/image/Comfy Cloud"
    requires_image = True
    returns_video = False
    turbo_tooltip = (
        "Run a four-step pass instead of the full forty-step one. Around seven times "
        "quicker, and visibly softer: this pipeline has no accelerator behind the switch, "
        "so the short schedule is the whole saving."
    )
    model_options = _DEFAULT_EDIT_MODELS
    model_tooltip = (
        "How much precision to spend on the weights. balanced is what this pipeline ships "
        "with; quality is the full-range weights, fast a quantised build that loads and "
        "runs quicker."
    )


class ComfyCloudFlux2TextToImageNode(IO.ComfyNode):
    node_id = "ComfyCloudFlux2TextToImageNode"
    display_name = "Comfy Cloud Flux 2 Text to Image"
    summary = (
        "Generates an image from a text prompt with Flux 2 dev. Turbo swaps in the chosen LoRA "
        "and a short schedule, trading a little fidelity for a much quicker run; switch it off "
        "for the full-length dev pass with no LoRA."
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
                    tooltip="Run the chosen LoRA at turbo_steps instead of the full dev pass.",
                ),
                _seed_input(),
                _weights_input(
                    "model",
                    _FLUX2_MODELS,
                    "Flux 2 dev precision. fp8 loads quicker; bf16 is the full-range weights.",
                ),
                _weights_input(
                    "lora",
                    _FLUX2_LORAS,
                    "LoRA to apply, loaded only while turbo is on; switch turbo off to run dev "
                    "with no LoRA at all. The two turbo entries are trained for the short "
                    "turbo_steps schedule, so a style entry wants turbo_steps raised toward steps.",
                ),
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
        model: str = "dev-fp8",
        lora: str = "turbo",
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
                prompt=prompt, width=width, height=height, turbo=turbo, seed=seed, model=model,
                lora=lora, steps=steps, turbo_steps=turbo_steps, turbo_strength=turbo_strength,
                guidance=guidance,
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
                _weights_input(
                    "model",
                    _Z_IMAGE_MODELS,
                    "Checkpoint precision. int8 and nvfp4 are quantised and load quicker; bf16 "
                    "is the reference weights.",
                ),
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
        model: str = "bf16",
        steps: int = 8,
        shift: float = 3.0,
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        width, height = _dimensions(aspect_ratio)
        return await _run_image_workflow(
            cls,
            "z-image-turbo/text-to-image",
            ComfyCloudWorkflowInputs(
                prompt=prompt, width=width, height=height, seed=seed, model=model, steps=steps,
                shift=shift,
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
            ComfyCloudFlux2TextToImageNode,
            ComfyCloudZImageTurboNode,
        ]


async def comfy_entrypoint() -> ComfyCloudExtension:
    return ComfyCloudExtension()

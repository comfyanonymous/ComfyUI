import contextlib
import math
import posixpath
from io import BytesIO
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
    audio_bytes_to_audio_input,
    download_url_to_bytesio,
    download_url_to_file_3d,
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
_MODEL3D_IO_TYPE = "FILE_3D_GLB"
_MODEL3D_FILE_FORMAT = "glb"
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
    "lyrics": (0, 4096),
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
_Z_IMAGE_MODELS = ["z_image_turbo_bf16", "z_image_turbo_int8_convrot", "z_image_turbo_nvfp4"]
_FLUX2_MODELS = ["flux2_dev_fp8mixed", "flux2-dev"]
_MAGE_FLOW_MODELS = ["mage_flow_int8_convrot", "mage_flow_bf16", "mage_flow_base_bf16"]
_MAGE_FLOW_TURBO_MODELS = ["mage_flow_turbo_int8_convrot", "mage_flow_turbo_bf16"]
_MAGE_FLOW_TEXT_ENCODERS = ["qwen3vl_4b_bf16", "qwen3vl_4b_fp8_scaled"]
_MINIMAX_MUSIC3_MODELS = [
    "minimax_music3_dit_fp16", "minimax_music3_dit_fp32", "minimax_music3_dit_int8_convrot",
]
_MINIMAX_MUSIC3_TEXT_ENCODERS = [
    "minimax_music3_text_encoder_pruned_int8_convrot", "minimax_music3_text_encoder_pruned_bf16",
]
_MINIMAX_MUSIC3_QUALITIES = ["V0", "128k", "320k"]
# The samplers and schedulers the curated graphs offer. ComfyUI ships far more;
# these are the ones executed on cloud against every graph that lists them.
_SAMPLERS = ["euler", "dpmpp_2m", "res_multistep"]
_SCHEDULERS = ["simple", "karras", "beta"]
_FLUX2_LORAS = [
    "Flux_2-Turbo-LoRA_comfyui", "Flux2TurboComfyv2", "flux2-herbst_photo_analog_film", "flux2_berthe_morisot", "flux2-boreal_dev2_boring_reality_for_dev",
    "flux2-yfg_chatgpt_4o_style", "flux2-wanderer_s_detailed_portraits", "flux2-yfg_fonts_japanese_manga_posters", "flux2-neo_victorian_style",
    "flux2-yfg_soares", "flux2-yfg_spy_world_50s_dev_and_dev", "flux2-lenovo_ultrareal",
]
# The default/* pickers key on the TRADE-OFF rather than the model, because those
# ids are pointers: the model behind one is re-pointed over time and the id does
# not change. A saved graph stores the key, so a key that named a model would
# break every saved graph the day the pointer moved.
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


def _seed_input(maximum: int = _UINT64_MAX) -> IO.Int.Input:
    return IO.Int.Input("seed", default=42, min=0, max=maximum, control_after_generate=True)


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


def _sampling_inputs(steps: int, cfg: float, cfg_max: float, sampler_node: str) -> list[IO.Input]:
    """The KSampler dials a curated graph exposes. Every graph that has a sampler
    has the same five, so they are described once."""
    return [
        _steps_input(steps, 100),
        _tuning_input("cfg", cfg, cfg_max, tooltip=f"Classifier-free guidance scale for the {sampler_node} pass."),
        _tuning_input("denoise", 1.0, 1.0, step=0.01),
        IO.Combo.Input("sampler", options=_SAMPLERS, default="euler", advanced=True),
        IO.Combo.Input("scheduler", options=_SCHEDULERS, default="simple", advanced=True),
    ]


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


async def _run_audio_workflow(
    cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs
) -> IO.NodeOutput:
    url = await _submit_workflow(cls, workflow, inputs)
    buffer = BytesIO()
    await download_url_to_bytesio(
        url, buffer, timeout=_OUTPUT_DOWNLOAD_TIMEOUT, cls=cls, allow_redirects=False
    )
    return IO.NodeOutput(audio_bytes_to_audio_input(buffer.getvalue()))


async def _run_model3d_workflow(
    cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs
) -> IO.NodeOutput:
    url = await _submit_workflow(cls, workflow, inputs)
    return IO.NodeOutput(
        await download_url_to_file_3d(
            url, _MODEL3D_FILE_FORMAT, timeout=_OUTPUT_DOWNLOAD_TIMEOUT, cls=cls, allow_redirects=False
        )
    )


# Output kind -> (schema output, runner). The bucket pin and signed-URL check in
# _submit_workflow apply to every kind: they guard where the bytes come from, not
# what they decode to.
_OUTPUT_KINDS = {
    "image": (IO.Image.Output, _run_image_workflow),
    "video": (IO.Video.Output, _run_video_workflow),
    "audio": (IO.Audio.Output, _run_audio_workflow),
    "model3d": (lambda: IO.Custom(_MODEL3D_IO_TYPE).Output(), _run_model3d_workflow),
}


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
    output_kind: ClassVar[str] = "image"
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
            _OUTPUT_KINDS[cls.output_kind][0](),
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
        run = _OUTPUT_KINDS[cls.output_kind][1]
        return await run(cls, cls.workflow, inputs)


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
        model: str = "flux2_dev_fp8mixed",
        lora: str = "Flux_2-Turbo-LoRA_comfyui",
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
        model: str = "z_image_turbo_bf16",
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


class _ComfyCloudMageFlowNode(IO.ComfyNode):
    """Mage-Flow text to image. The base and turbo graphs are the same pipeline at
    two schedule lengths, so they differ only in their step and cfg defaults and
    which checkpoints load."""

    workflow: ClassVar[ComfyCloudWorkflow]
    node_id: ClassVar[str]
    display_name: ClassVar[str]
    summary: ClassVar[str]
    default_steps: ClassVar[int]
    default_cfg: ClassVar[float]
    model_options: ClassVar[list[str]]

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls,
            [
                _prompt_input(),
                _aspect_ratio_input(),
                _seed_input(),
                _negative_prompt_input(),
                # The graph sizes itself through a resolution selector, so the
                # caller picks a ratio and a pixel budget rather than a raw pair.
                IO.Float.Input(
                    "megapixels", default=1.0, min=0.1, max=16.0, step=0.1, advanced=True,
                    tooltip="Total pixel budget. 1.0 is about 1024x1024 at a square ratio.",
                ),
                IO.Int.Input(
                    "size_multiple", default=16, min=8, max=128, step=4, advanced=True,
                    tooltip="Round each side to this multiple.",
                ),
                *_sampling_inputs(cls.default_steps, cls.default_cfg, 20.0, "diffusion"),
                _weights_input(
                    "model", cls.model_options,
                    "Checkpoint precision. int8 loads quicker; bf16 is the reference weights.",
                ),
                _weights_input(
                    "text_encoder", _MAGE_FLOW_TEXT_ENCODERS,
                    "Qwen3-VL 4B precision used to encode the prompt.",
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str = "1:1",
        seed: int = 42,
        negative_prompt: str = "",
        megapixels: float = 1.0,
        size_multiple: int = 16,
        steps: int | None = None,
        cfg: float | None = None,
        denoise: float = 1.0,
        sampler: str = "euler",
        scheduler: str = "simple",
        model: str | None = None,
        text_encoder: str = "qwen3vl_4b_bf16",
    ) -> IO.NodeOutput:
        # The three per-variant defaults are resolved before validation, so the
        # subclass's schema defaults are what an omitted argument falls back to.
        steps = cls.default_steps if steps is None else steps
        cfg = cls.default_cfg if cfg is None else cfg
        model = cls.model_options[0] if model is None else model
        validated = _validate_node_inputs(cls, locals())
        return await _run_image_workflow(
            cls,
            cls.workflow,
            ComfyCloudWorkflowInputs(
                prompt=validated["prompt"], negative_prompt=validated["negative_prompt"],
                aspect_ratio=aspect_ratio, megapixels=megapixels, size_multiple=size_multiple,
                seed=seed, steps=steps, cfg=cfg, denoise=denoise, sampler=sampler,
                scheduler=scheduler, model=model, text_encoder=text_encoder,
            ),
        )


class ComfyCloudMageFlowTextToImageNode(_ComfyCloudMageFlowNode):
    workflow = "mage-flow/text-to-image"
    node_id = "ComfyCloudMageFlowTextToImageNode"
    display_name = "Comfy Cloud Mage Flow Text to Image"
    summary = (
        "Generates an image from a text prompt with Mage-Flow over a full 30-step pass. "
        "It takes a negative prompt, which the distilled turbo variant cannot use well."
    )
    default_steps = 30
    default_cfg = 5.0
    model_options = _MAGE_FLOW_MODELS


class ComfyCloudMageFlowTurboTextToImageNode(_ComfyCloudMageFlowNode):
    workflow = "mage-flow-turbo/text-to-image"
    node_id = "ComfyCloudMageFlowTurboTextToImageNode"
    display_name = "Comfy Cloud Mage Flow Turbo Text to Image"
    summary = (
        "Generates an image from a text prompt with distilled Mage-Flow in 4 steps at cfg 1. "
        "Roughly a seventh of the GPU time of the full pass, which makes it the one to iterate on."
    )
    default_steps = 4
    default_cfg = 1.0
    model_options = _MAGE_FLOW_TURBO_MODELS


class ComfyCloudMiniMaxMusic3TextToAudioNode(IO.ComfyNode):
    output_kind = "audio"

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _cloud_schema(
            "ComfyCloudMiniMaxMusic3TextToAudioNode",
            "Comfy Cloud MiniMax Music 3 Text to Audio",
            (
                "Generates a full song from a description with MiniMax Music 3. The prompt "
                "carries the style, instrumentation and mood; lyrics are sung rather than "
                "described, and an empty lyric leaves the track instrumental."
            ),
            "partner/audio/Comfy Cloud",
            [
                _prompt_input(),
                IO.String.Input(
                    "lyrics", multiline=True, default="",
                    tooltip="Words to sing. Leave empty for an instrumental.",
                ),
                IO.Float.Input(
                    "max_duration", default=120.0, min=0.04, max=360.0, step=0.04,
                    tooltip="Longest the track may run. The model can end the song earlier.",
                ),
                IO.Boolean.Input(
                    "tiled_decode", default=True,
                    tooltip="Decode the waveform in tiles. Off decodes in one pass, which is "
                            "quicker on short tracks and needs far more memory on long ones.",
                ),
                # SeedNode caps at int64, below the uint64 the other graphs take.
                _seed_input(0x7FFFFFFFFFFFFFFF),
                _tuning_input(
                    "caption_cfg", 1.5, 100.0,
                    tooltip="How closely the arrangement follows the prompt.",
                ),
                IO.Int.Input(
                    "top_k", default=50, min=1, max=16384, advanced=True,
                    tooltip="Token sampling width while the prompt is turned into an arrangement.",
                ),
                *_sampling_inputs(30, 1.7, 100.0, "audio diffusion"),
                IO.Int.Input("tile_size", default=1536, min=32, max=8192, step=8, advanced=True),
                IO.Int.Input("tile_overlap", default=64, min=0, max=1024, step=8, advanced=True),
                _weights_input("model", _MINIMAX_MUSIC3_MODELS, "Diffusion transformer precision."),
                _weights_input(
                    "text_encoder", _MINIMAX_MUSIC3_TEXT_ENCODERS, "Prompt encoder precision."
                ),
                _weights_input(
                    "audio_quality", _MINIMAX_MUSIC3_QUALITIES,
                    "mp3 bitrate. V0 is variable and the highest quality of the three.",
                ),
            ],
            _OUTPUT_KINDS[cls.output_kind][0](),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        lyrics: str = "",
        max_duration: float = 120.0,
        tiled_decode: bool = True,
        seed: int = 42,
        caption_cfg: float = 1.5,
        top_k: int = 50,
        steps: int = 30,
        cfg: float = 1.7,
        denoise: float = 1.0,
        sampler: str = "euler",
        scheduler: str = "simple",
        tile_size: int = 1536,
        tile_overlap: int = 64,
        model: str = "minimax_music3_dit_fp16",
        text_encoder: str = "minimax_music3_text_encoder_pruned_int8_convrot",
        audio_quality: str = "V0",
    ) -> IO.NodeOutput:
        validated = _validate_node_inputs(cls, locals())
        run = _OUTPUT_KINDS[cls.output_kind][1]
        return await run(
            cls,
            "minimax-music-3/text-to-audio",
            ComfyCloudWorkflowInputs(
                prompt=validated["prompt"], lyrics=validated["lyrics"],
                max_duration=max_duration, tiled_decode=tiled_decode, seed=seed,
                caption_cfg=caption_cfg, top_k=top_k, steps=steps, cfg=cfg, denoise=denoise,
                sampler=sampler, scheduler=scheduler, tile_size=tile_size,
                tile_overlap=tile_overlap, model=model, text_encoder=text_encoder,
                audio_quality=audio_quality,
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
            ComfyCloudMiniMaxH3TextSoundNode,
            ComfyCloudMiniMaxMusic3TextToAudioNode,
            ComfyCloudFlux2TextToImageNode,
            ComfyCloudZImageTurboNode,
            ComfyCloudMageFlowTextToImageNode,
            ComfyCloudMageFlowTurboTextToImageNode,
        ]


async def comfy_entrypoint() -> ComfyCloudExtension:
    return ComfyCloudExtension()

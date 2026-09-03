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

        return _cloud_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            cls.category,
            inputs,
            IO.Video.Output() if cls.returns_video else IO.Image.Output(),
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


_ASPECT_RATIOS = ["1:1", "3:4", "2:3", "3:2", "4:3", "16:9", "9:16", "21:9"]
_UINT64_MAX = 0xFFFFFFFFFFFFFFFF


def _prompt_input(name: str = "prompt") -> IO.String.Input:
    return IO.String.Input(name, multiline=True, default="")


def _aspect_ratio_input() -> IO.Combo.Input:
    return IO.Combo.Input("aspect_ratio", options=_ASPECT_RATIOS, default="1:1")


def _seed_input() -> IO.Int.Input:
    return IO.Int.Input("seed", default=0, min=0, max=_UINT64_MAX, control_after_generate=True)


class _ComfyCloudPromptSeedImageNode(_ComfyCloudWorkflowNode):
    """A named image workflow whose whole surface is a prompt and a seed.

    Five of the shipped nodes differ only in workflow id, name and blurb, so they
    subclass this and declare nothing else.
    """

    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _cloud_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            cls.category,
            [_prompt_input(), _seed_input()],
            IO.Image.Output(),
        )

    @classmethod
    # pylint: disable=arguments-renamed
    async def execute(cls, prompt: str, seed: int = 0) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        return await cls._run(ComfyCloudWorkflowInputs(prompt=prompt, seed=seed))


class ComfyCloudCapybaraTextToImageNode(_ComfyCloudPromptSeedImageNode):
    workflow = "capybara-0.1/text-to-image"
    node_id = "ComfyCloudCapybaraTextToImageNode"
    display_name = "Comfy Cloud Capybara 0.1 Text to Image"
    summary = (
        "Generates an image from a text prompt with Capybara 0.1, rendered at 1280x1280 "
        "rather than upscaled from a smaller canvas."
    )


class ComfyCloudIdeogram4TextToImageNode(_ComfyCloudPromptSeedImageNode):
    workflow = "ideogram-4/text-to-image"
    node_id = "ComfyCloudIdeogram4TextToImageNode"
    display_name = "Comfy Cloud Ideogram 4 Text to Image"
    summary = (
        "Generates an image from a text prompt with Ideogram 4, a model aimed at typography "
        "and graphic layout work."
    )


class ComfyCloudLongCatTextToImageNode(_ComfyCloudPromptSeedImageNode):
    workflow = "longcat/text-to-image"
    node_id = "ComfyCloudLongCatTextToImageNode"
    display_name = "Comfy Cloud LongCat Text to Image"
    summary = (
        "Generates a 1024x1024 image from a text prompt with LongCat, running a full 20-step "
        "sampler instead of a distilled shortcut. Slower and dearer per run than the turbo "
        "models, and steadier on fine detail."
    )


class ComfyCloudFlux2TextToImageNode(_ComfyCloudPromptSeedImageNode):
    workflow = "flux-2/text-to-image"
    node_id = "ComfyCloudFlux2TextToImageNode"
    display_name = "Comfy Cloud Flux 2 Text to Image"
    summary = (
        "Generates an image from a text prompt with Flux 2 dev plus its Turbo LoRA, which "
        "trades a little fidelity for a much shorter run."
    )


class ComfyCloudZImageTurboNode(_ComfyCloudPromptSeedImageNode):
    workflow = "z-image-turbo/text-to-image"
    node_id = "ComfyCloudZImageTurboNode"
    display_name = "Comfy Cloud Z-Image Turbo Text to Image"
    summary = (
        "Generates a 1024x1024 image from a text prompt with Z-Image Turbo in 8 steps. One of "
        "the quickest and cheapest nodes here, which makes it the one to iterate on."
    )


class ComfyCloudKrea2CreativeImageNode(_ComfyCloudWorkflowNode):
    workflow = "krea-2/text-to-image"
    node_id = "ComfyCloudKrea2CreativeImageNode"
    display_name = "Comfy Cloud Krea 2 Text to Image"
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
        return _cloud_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            cls.category,
            [
                _prompt_input(),
                IO.Boolean.Input("prompt_enhance", default=True),
                _aspect_ratio_input(),
                _seed_input(),
            ],
            IO.Image.Output(),
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
    workflow = "qwen-image-edit-2511/image-edit"
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
        return _cloud_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            cls.category,
            [
                IO.Image.Input("image"),
                _prompt_input("instruction"),
                IO.Combo.Input("quality_mode", options=["quality", "fast"], default="quality"),
                _seed_input(),
            ],
            IO.Image.Output(),
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
    workflow = "seedvr2/upscale-image"
    node_id = "ComfyCloudSeedVR2ImageUpscaleNode"
    display_name = "Comfy Cloud SeedVR2 Upscale Image"
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
        return _cloud_schema(
            cls.node_id,
            cls.display_name,
            cls.summary,
            cls.category,
            [IO.Image.Input("image"), IO.Combo.Input("scale", options=["2x", "4x"], default="4x")],
            IO.Image.Output(),
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


async def _run_video_workflow(
    cls: type[IO.ComfyNode],
    workflow: ComfyCloudWorkflow,
    inputs: ComfyCloudWorkflowInputs,
) -> IO.NodeOutput:
    task = await sync_op(
        cls,
        _GENERATE_ENDPOINT,
        response_model=ComfyCloudGenerateResponse,
        data=ComfyCloudGenerateRequest(workflow=workflow, inputs=inputs),
    )
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
    return _cloud_schema(
        node_id, display_name, summary, "partner/video/Comfy Cloud", inputs, IO.Video.Output()
    )


def _video_seed_input(default: int) -> IO.Int.Input:
    return IO.Int.Input("seed", default=default, min=0, max=_UINT64_MAX, control_after_generate=True)


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
            [
                _prompt_input(),
                _aspect_ratio_input(),
                IO.Float.Input("duration_seconds", default=5, min=5, max=15, step=0.01),
                _video_seed_input(168866841893410),
            ],
        )

    @classmethod
    async def execute(cls, prompt: str, aspect_ratio: str, duration_seconds: float, seed: int) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        inputs = ComfyCloudWorkflowInputs(
            prompt=prompt, aspect_ratio=aspect_ratio, duration_seconds=duration_seconds, seed=seed
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
            [
                IO.Image.Input("image"),
                _prompt_input(),
                _aspect_ratio_input(),
                IO.Float.Input("duration_seconds", default=5, min=5, max=15, step=0.01),
                _video_seed_input(168866841893410),
            ],
        )

    @classmethod
    async def execute(
        cls, image: Input.Image, prompt: str, aspect_ratio: str, duration_seconds: float, seed: int
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        _validate_image_upload(image)
        image_url = await upload_image_to_comfyapi(cls, image)
        inputs = ComfyCloudWorkflowInputs(
            prompt=prompt, image_url=image_url, aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds, seed=seed,
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
                _video_seed_input(225158785956033),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        audio: Input.Audio,
        prompt: str,
        enhance_prompt: bool,
        duration_seconds: float,
        seed: int,
    ) -> IO.NodeOutput:
        prompt = _validate_node_inputs(cls, locals())["prompt"]
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        _validate_image_upload(image)
        _validate_audio_upload(audio)
        audio_duration = _audio_duration(audio)
        if duration_seconds - min(1 / float(audio["sample_rate"]), 1e-3) > audio_duration:
            raise ValueError(
                f"Duration ({duration_seconds:g}s) exceeds input audio duration ({audio_duration:.2f}s)."
            )
        image_url = await upload_image_to_comfyapi(cls, image)
        audio_url = await upload_audio_to_comfyapi(cls, audio)
        inputs = ComfyCloudWorkflowInputs(
            prompt=prompt, image_url=image_url, audio_url=audio_url, enhance_prompt=enhance_prompt,
            duration_seconds=duration_seconds, seed=seed,
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
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Leave empty to use the negative prompt the frozen Wan 2.2 graph was "
                        "tested with."
                    ),
                ),
                IO.Int.Input(
                    "duration_seconds",
                    default=5,
                    min=2,
                    max=8,
                    step=1,
                    tooltip="Graph frame count is floor(duration × 16 + 1).",
                ),
                _video_seed_input(984937593540091),
            ],
        )

    @classmethod
    async def execute(
        cls,
        first_frame: Input.Image,
        last_frame: Input.Image,
        prompt: str,
        negative_prompt: str,
        duration_seconds: int,
        seed: int,
    ) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        negative_prompt = values["negative_prompt"]
        if get_number_of_images(first_frame) != 1 or get_number_of_images(last_frame) != 1:
            raise ValueError("Exactly one first frame and one last frame are required.")
        _validate_image_upload(first_frame)
        _validate_image_upload(last_frame)
        first_url = await upload_image_to_comfyapi(cls, first_frame, wait_label="Uploading first frame")
        last_url = await upload_image_to_comfyapi(cls, last_frame, wait_label="Uploading last frame")
        # Omitted, not "", so the backend applies the graph's own negative prompt.
        inputs = ComfyCloudWorkflowInputs(
            prompt=prompt, negative_prompt=negative_prompt or None, first_frame_url=first_url,
            last_frame_url=last_url, duration_seconds=duration_seconds, seed=seed,
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

import math
import posixpath
import re
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
    download_url_to_audio_input,
    download_url_to_file_3d,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    sync_op_raw,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
)


_GENERATE_ENDPOINT = ApiEndpoint(path="/proxy/comfy-cloud/workflow/generate", method="POST")
_OUTPUT_DOWNLOAD_TIMEOUT = 30 * 60
_MAX_UPLOAD_IMAGE_PIXELS = 32_000_000
_MAX_UPLOAD_IMAGE_DIMENSION = 8192
_MAX_DECODED_AUDIO_BYTES = 256 * 1024 * 1024
COMFY_CLOUD_GPU_SECOND_USD = 0.001295
COMFY_CLOUD_CREDITS_PER_USD = 211
COMFY_CLOUD_GPU_SECOND_CREDITS = COMFY_CLOUD_GPU_SECOND_USD * COMFY_CLOUD_CREDITS_PER_USD
COMFY_CLOUD_GPU_HOUR_USD = COMFY_CLOUD_GPU_SECOND_USD * 3600
COMFY_CLOUD_GPU_HOUR_CREDITS = COMFY_CLOUD_GPU_SECOND_CREDITS * 3600
_COMFY_CLOUD_PRICE_BADGE = IO.PriceBadge(
    expr=(
        f'{{"type":"usd","usd":{COMFY_CLOUD_GPU_SECOND_USD:.6f},'
        '"format":{"suffix":"/GPU-second","approximate":true}}'
    )
)
_COMFY_CLOUD_RATE_DESCRIPTION = (
    f" Estimated compute rate: ${COMFY_CLOUD_GPU_SECOND_USD:.6f}/GPU-second "
    f"({COMFY_CLOUD_GPU_SECOND_CREDITS:.6f} credits/GPU-second using "
    f"{COMFY_CLOUD_CREDITS_PER_USD} credits/USD; ${COMFY_CLOUD_GPU_HOUR_USD:.3f} or "
    f"{COMFY_CLOUD_GPU_HOUR_CREDITS:.3f} credits/GPU-hour). "
    "Actual final cost depends on GPU runtime."
)
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
    is_signed_https_url = (
        parsed.scheme == "https"
        and parsed.hostname == "storage.googleapis.com"
        and parsed.port is None
        and parsed.username is None
        and parsed.password is None
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
            description="Runs this workflow on Comfy Cloud and returns the generated media."
            + _COMFY_CLOUD_RATE_DESCRIPTION,
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
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False


class ComfyCloudTextToVideoNode(_ComfyCloudWorkflowNode):
    workflow = "text-to-video"
    node_id = "ComfyCloudTextToVideoNode"
    display_name = "Comfy Cloud Text to Video"
    category = "partner/video/Comfy Cloud"
    requires_image = False
    returns_video = True


class ComfyCloudImageToVideoNode(_ComfyCloudWorkflowNode):
    workflow = "image-to-video"
    node_id = "ComfyCloudImageToVideoNode"
    display_name = "Comfy Cloud Image to Video"
    category = "partner/video/Comfy Cloud"
    requires_image = True
    returns_video = True


class ComfyCloudImageEditNode(_ComfyCloudWorkflowNode):
    workflow = "image-edit"
    node_id = "ComfyCloudImageEditNode"
    display_name = "Comfy Cloud Image Edit"
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


def _image_schema(node_id: str, display_name: str, inputs: list[IO.Input]) -> IO.Schema:
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        category="partner/image/Comfy Cloud",
        description="Runs this image workflow on Comfy Cloud and returns the generated image."
        + _COMFY_CLOUD_RATE_DESCRIPTION,
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


class ComfyCloudIdeogram4DesignNode(_ComfyCloudWorkflowNode):
    workflow = "image.ideogram-4-design.v1"
    node_id = "ComfyCloudIdeogram4DesignNode"
    display_name = "Ideogram 4 Design"
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            [
                _prompt_input(),
                _aspect_ratio_input(),
                IO.Combo.Input(
                    "quality_mode", options=["quality", "balanced", "fast"], default="balanced"
                ),
                _seed_input(),
            ],
        )

    @classmethod
    # pylint: disable=arguments-renamed
    async def execute(
        cls, prompt: str, aspect_ratio: str = "1:1", quality_mode: str = "balanced", seed: int = 0
    ) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        return await cls._run(
            ComfyCloudWorkflowInputs(
                prompt=prompt, aspect_ratio=aspect_ratio, quality_mode=quality_mode, seed=seed
            )
        )


class ComfyCloudKrea2CreativeImageNode(_ComfyCloudWorkflowNode):
    workflow = "image.krea-2-creative-image.v1"
    node_id = "ComfyCloudKrea2CreativeImageNode"
    display_name = "Krea 2 Creative Image"
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
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


class ComfyCloudMageFlowImageNode(_ComfyCloudWorkflowNode):
    workflow = "image.mage-flow-image.v1"
    node_id = "ComfyCloudMageFlowImageNode"
    display_name = "Mage-Flow Image"
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            [
                _prompt_input(),
                IO.String.Input("negative_prompt", multiline=True, default=""),
                _aspect_ratio_input(),
                _seed_input(),
            ],
        )

    @classmethod
    # pylint: disable=arguments-renamed
    async def execute(
        cls, prompt: str, negative_prompt: str = "", aspect_ratio: str = "1:1", seed: int = 0
    ) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        negative_prompt = values["negative_prompt"]
        return await cls._run(
            ComfyCloudWorkflowInputs(
                prompt=prompt, negative_prompt=negative_prompt, aspect_ratio=aspect_ratio, seed=seed
            )
        )


class ComfyCloudFlux2ReferenceEditNode(_ComfyCloudWorkflowNode):
    workflow = "image.flux-2-reference-edit.v1"
    node_id = "ComfyCloudFlux2ReferenceEditNode"
    display_name = "FLUX.2 Reference Edit"
    category = "partner/image/Comfy Cloud"
    requires_image = True
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            [
                IO.Image.Input("image"),
                _prompt_input("instruction"),
                IO.Float.Input("guidance", default=4.0, min=1.0, max=10.0, step=0.1),
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
        guidance: float = 4.0,
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
                guidance=guidance,
                quality_mode=quality_mode,
                seed=seed,
            )
        )


class ComfyCloudQwenImageEdit2511Node(_ComfyCloudWorkflowNode):
    workflow = "image.qwen-image-edit-2511.v1"
    node_id = "ComfyCloudQwenImageEdit2511Node"
    display_name = "Qwen Image Edit 2511"
    category = "partner/image/Comfy Cloud"
    requires_image = True
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
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
    display_name = "SeedVR2 Image Upscale"
    category = "partner/image/Comfy Cloud"
    requires_image = True
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
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


def _video_schema(node_id: str, display_name: str, inputs: list[IO.Input]) -> IO.Schema:
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        category="partner/video/Comfy Cloud",
        description="Runs this video workflow on Comfy Cloud and returns the generated video."
        + _COMFY_CLOUD_RATE_DESCRIPTION,
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
            "MiniMax H3 Text + Sound",
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
            "MiniMax H3 Image + Sound",
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
            "LTX-2.3 Image + Audio Performance",
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


class ComfyCloudLTX23FirstLastFrameNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudLTX23FirstLastFrameNode",
            "LTX-2.3 First & Last Frame",
            [IO.Image.Input("first_frame"), IO.Image.Input("last_frame"), _prompt_input(), IO.Int.Input("duration_seconds", default=5, min=2, max=10, step=1, tooltip="25 fps; output frame count is duration × 25 + 1."), _video_seed_input(315253765879496)],
        )

    @classmethod
    async def execute(cls, first_frame: Input.Image, last_frame: Input.Image, prompt: str, duration_seconds: int, seed: int) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        if get_number_of_images(first_frame) != 1 or get_number_of_images(last_frame) != 1:
            raise ValueError("Exactly one first frame and one last frame are required.")
        _validate_image_upload(first_frame)
        _validate_image_upload(last_frame)
        first_url = await upload_image_to_comfyapi(cls, first_frame, wait_label="Uploading first frame")
        last_url = await upload_image_to_comfyapi(cls, last_frame, wait_label="Uploading last frame")
        return await _run_video_workflow(cls, "video.ltx-2-3-first-last-frame.v1", ComfyCloudWorkflowInputs(prompt=prompt, first_frame_url=first_url, last_frame_url=last_url, duration_seconds=duration_seconds, seed=seed))


class ComfyCloudWan22FirstLastFrameNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudWan22FirstLastFrameNode",
            "Wan 2.2 14B First & Last Frame",
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


class ComfyCloudSCAIL2CharacterReplacementNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudSCAIL2CharacterReplacementNode",
            "SCAIL-2 Character Replacement",
            [IO.Image.Input("reference_character"), IO.Video.Input("driving_video", tooltip="Must contain 81–157 decoded frames."), _prompt_input("scene_prompt"), IO.String.Input("driving_subject", default="human"), IO.String.Input("reference_subject", default="human"), _video_seed_input(1)],
        )

    @classmethod
    async def execute(cls, reference_character: Input.Image, driving_video: Input.Video, scene_prompt: str, driving_subject: str, reference_subject: str, seed: int) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        scene_prompt = values["scene_prompt"]
        driving_subject = values["driving_subject"]
        reference_subject = values["reference_subject"]
        if get_number_of_images(reference_character) != 1:
            raise ValueError("Exactly one reference character image is required.")
        _validate_image_upload(reference_character)
        try:
            frame_count = driving_video.get_frame_count()
        except Exception as error:
            raise ValueError("Unable to determine video frame count.") from error
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise ValueError("Unable to determine video frame count.")
        if not 81 <= frame_count <= 157:
            raise ValueError(f"Video frame count must be between 81 and 157, got {frame_count}.")
        image_url = await upload_image_to_comfyapi(cls, reference_character)
        video_url = await upload_video_to_comfyapi(cls, driving_video)
        return await _run_video_workflow(cls, "video.scail-2-character-replacement.v1", ComfyCloudWorkflowInputs(scene_prompt=scene_prompt, driving_subject=driving_subject, reference_subject=reference_subject, reference_character_url=image_url, driving_video_url=video_url, seed=seed))


_UINT32_MAX = 0xFFFFFFFF
_ACE_LANGUAGES = ["ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh", "unknown"]
_ACE_KEYS = [f"{root} {mode}" for mode in ("major", "minor") for root in ("C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B")]
_CHATTERBOX_LANGUAGES = ["Arabic (ar)", "Danish (da)", "German (de)", "Greek (el)", "English (en)", "Spanish (es)", "Finnish (fi)", "French (fr)", "Hebrew (he)", "Hindi (hi)", "Italian (it)", "Japanese (ja)", "Korean (ko)", "Malay (ms)", "Dutch (nl)", "Norwegian (no)", "Polish (pl)", "Portuguese (pt)", "Russian (ru)", "Swedish (sv)", "Swahili (sw)", "Turkish (tr)", "Chinese (zh)"]


def _audio_schema(node_id: str, display_name: str, inputs: list[IO.Input], outputs: list[IO.Output] | None = None) -> IO.Schema:
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        category="partner/audio/Comfy Cloud",
        description="Runs this audio workflow on Comfy Cloud and returns the generated audio."
        + _COMFY_CLOUD_RATE_DESCRIPTION,
        inputs=_with_input_sockets(inputs),
        outputs=outputs or [IO.Audio.Output()],
        hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
        is_api_node=True,
        price_badge=_COMFY_CLOUD_PRICE_BADGE,
    )


def _audio_duration(audio: Input.Audio) -> float:
    sample_rate = float(audio["sample_rate"])
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Audio sample rate must be a positive number.")
    return audio["waveform"].shape[-1] / sample_rate


def _validate_audio_duration(name: str, audio: Input.Audio, minimum: float, maximum: float) -> None:
    duration = _audio_duration(audio)
    tolerance = min(1 / float(audio["sample_rate"]), 1e-3)
    if duration + tolerance < minimum or duration - tolerance > maximum:
        raise ValueError(f"{name} duration must be between {minimum:g} and {maximum:g} seconds.")


def _normalize_dialogue(script: str) -> str:
    utterances: list[tuple[str, list[str]]] = []
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(r"(?:SPEAKER\s+)?([A-Z])\s*:\s*(.*)", line, flags=re.IGNORECASE)
        if match:
            speaker, text = match.groups()
            if speaker.upper() not in ("A", "B"):
                raise ValueError("Dialogue supports only speakers A and B.")
            if utterances and not utterances[-1][1]:
                raise ValueError("Dialogue utterances cannot be blank.")
            utterances.append((speaker.upper(), [text.strip()] if text.strip() else []))
        elif re.match(r"(?:SPEAKER\s+[A-Z]|NARRATOR)\s*:", line, flags=re.IGNORECASE):
            raise ValueError("Dialogue supports only speakers A and B.")
        elif utterances:
            utterances[-1][1].append(line)
        else:
            raise ValueError("Dialogue must start with speaker A or B.")
    if not utterances:
        raise ValueError("Dialogue must contain at least one utterance.")
    if not utterances[-1][1]:
        raise ValueError("Dialogue utterances cannot be blank.")
    return "\n".join(f"SPEAKER {speaker}: {' '.join(lines)}" for speaker, lines in utterances)


async def _audio_asset(cls: type[IO.ComfyNode], name: str, audio: Input.Audio) -> dict[str, ComfyCloudAssetInput]:
    _validate_audio_upload(audio)
    return {name: ComfyCloudAssetInput(type="AUDIO", url=await upload_audio_to_comfyapi(cls, audio))}


async def _run_audio_workflow(cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs, output_names: tuple[str, ...] = ()) -> IO.NodeOutput:
    task = await sync_op(cls, _GENERATE_ENDPOINT, response_model=ComfyCloudGenerateResponse, data=ComfyCloudGenerateRequest(workflow=workflow, inputs=inputs))
    result = await _poll_task(cls, task.task_id)
    if output_names:
        if not result.output_urls or any(not result.output_urls.get(name) for name in output_names):
            raise RuntimeError("Comfy Cloud task completed without all named output URLs.")
        outputs = [
            await download_url_to_audio_input(
                _validated_output_url(result.output_urls[name]),
                timeout=_OUTPUT_DOWNLOAD_TIMEOUT,
                cls=cls,
                allow_redirects=False,
            )
            for name in output_names
        ]
        return IO.NodeOutput(*outputs)
    if not result.output_url:
        raise RuntimeError("Comfy Cloud task completed without an output URL.")
    return IO.NodeOutput(
        await download_url_to_audio_input(
            _validated_output_url(result.output_url),
            timeout=_OUTPUT_DOWNLOAD_TIMEOUT,
            cls=cls,
            allow_redirects=False,
        )
    )


class ComfyCloudACEStep15XLTurboNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _audio_schema(
            "ComfyCloudACEStep15XLTurboNode",
            "ACE-Step 1.5 XL Turbo",
            [
                _prompt_input("style_prompt"),
                IO.String.Input("lyrics", multiline=True, default=""),
                IO.Float.Input("duration_seconds", default=120, min=10, max=300, step=0.1),
                _seed_input(),
                IO.Int.Input("bpm", default=120, min=10, max=300),
                IO.Combo.Input("time_signature", options=["2", "3", "4", "6"], default="4"),
                IO.Combo.Input("language", options=_ACE_LANGUAGES, default="en"),
                IO.Combo.Input("key", options=_ACE_KEYS, default="E minor"),
            ],
        )

    @classmethod
    async def execute(cls, style_prompt: str, lyrics: str, duration_seconds: float, seed: int, bpm: int, time_signature: str, language: str, key: str) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        style_prompt = values["style_prompt"]
        lyrics = values["lyrics"]
        return await _run_audio_workflow(cls, "audio.ace-step-1-5-xl-turbo.v1", ComfyCloudWorkflowInputs(style_prompt=style_prompt, lyrics=lyrics, duration_seconds=duration_seconds, seed=seed, bpm=bpm, time_signature=time_signature, language=language, key=key))


class ComfyCloudStableAudio3MediumNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _audio_schema(
            "ComfyCloudStableAudio3MediumNode",
            "Stable Audio 3 Medium",
            [
                _prompt_input(),
                IO.Float.Input("duration_seconds", default=30, min=1, max=300, step=0.1),
                _seed_input(),
                IO.Boolean.Input("expand_prompt", default=True),
                IO.Combo.Input("category", options=["Music", "Instrument", "SFX", "One-shot"], default="Music"),
            ],
        )

    @classmethod
    async def execute(cls, prompt: str, duration_seconds: float, seed: int, expand_prompt: bool, category: str) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        prompt = values["prompt"]
        return await _run_audio_workflow(cls, "audio.stable-audio-3-medium.v1", ComfyCloudWorkflowInputs(prompt=prompt, duration_seconds=duration_seconds, seed=seed, expand_prompt=expand_prompt, category=category))


class ComfyCloudChatterboxMultilingualVoiceCloneNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _audio_schema(
            "ComfyCloudChatterboxMultilingualVoiceCloneNode",
            "Chatterbox Multilingual Voice Clone",
            [
                _prompt_input("text"), IO.Audio.Input("voice_reference"),
                IO.Combo.Input("language", options=_CHATTERBOX_LANGUAGES, default="English (en)"),
                IO.Float.Input("exaggeration", default=0.5, min=0, max=2, step=0.05),
                IO.Float.Input("cfg_weight", default=0.5, min=0, max=1, step=0.05),
                IO.Float.Input("temperature", default=0.8, min=0.05, max=2, step=0.05),
                IO.Int.Input("seed", default=0, min=0, max=_UINT32_MAX, control_after_generate=True),
            ],
        )

    @classmethod
    async def execute(cls, text: str, voice_reference: Input.Audio, language: str, exaggeration: float, cfg_weight: float, temperature: float, seed: int) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        text = values["text"]
        _validate_audio_duration("Voice reference", voice_reference, 1, 30)
        return await _run_audio_workflow(cls, "audio.chatterbox-multilingual-voice-clone.v1", ComfyCloudWorkflowInputs(text=text, assets=await _audio_asset(cls, "voice_reference", voice_reference), language=language, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature, seed=seed))


class ComfyCloudChatterboxDialogueNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _audio_schema(
            "ComfyCloudChatterboxDialogueNode",
            "Chatterbox Dialogue",
            [
                _prompt_input("script"), IO.Audio.Input("speaker_a_reference"), IO.Audio.Input("speaker_b_reference"),
                IO.Float.Input("exaggeration", default=0.5, min=0.25, max=2, step=0.05),
                IO.Float.Input("cfg_weight", default=0.5, min=0.2, max=1, step=0.05),
                IO.Float.Input("temperature", default=0.8, min=0.05, max=5, step=0.05),
                IO.Int.Input("seed", default=0, min=0, max=_UINT32_MAX, control_after_generate=True),
            ],
        )

    @classmethod
    async def execute(cls, script: str, speaker_a_reference: Input.Audio, speaker_b_reference: Input.Audio, exaggeration: float, cfg_weight: float, temperature: float, seed: int) -> IO.NodeOutput:
        values = _validate_node_inputs(cls, locals())
        script = values["script"]
        script = _normalize_dialogue(script)
        validate_string(script, min_length=1, max_length=10000, field_name="script")
        _validate_audio_upload(speaker_a_reference)
        _validate_audio_upload(speaker_b_reference)
        _validate_audio_duration("Speaker A reference", speaker_a_reference, 1, 30)
        _validate_audio_duration("Speaker B reference", speaker_b_reference, 1, 30)
        assets = {
            "speaker_a_reference": ComfyCloudAssetInput(type="AUDIO", url=await upload_audio_to_comfyapi(cls, speaker_a_reference)),
            "speaker_b_reference": ComfyCloudAssetInput(type="AUDIO", url=await upload_audio_to_comfyapi(cls, speaker_b_reference)),
        }
        return await _run_audio_workflow(cls, "audio.chatterbox-dialogue.v1", ComfyCloudWorkflowInputs(script=script, assets=assets, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature, seed=seed))


class ComfyCloudChatterboxVoiceConversionNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _audio_schema(
            "ComfyCloudChatterboxVoiceConversionNode",
            "Chatterbox Voice Conversion",
            [IO.Audio.Input("source_audio"), IO.Audio.Input("target_voice_reference"), IO.Int.Input("seed", default=0, min=0, max=_UINT32_MAX, control_after_generate=True)],
        )

    @classmethod
    async def execute(cls, source_audio: Input.Audio, target_voice_reference: Input.Audio, seed: int) -> IO.NodeOutput:
        _validate_node_inputs(cls, locals())
        _validate_audio_upload(source_audio)
        _validate_audio_upload(target_voice_reference)
        _validate_audio_duration("Source audio", source_audio, 0.5, 300)
        _validate_audio_duration("Target voice reference", target_voice_reference, 1, 30)
        assets = {
            "source_audio": ComfyCloudAssetInput(type="AUDIO", url=await upload_audio_to_comfyapi(cls, source_audio)),
            "target_voice_reference": ComfyCloudAssetInput(type="AUDIO", url=await upload_audio_to_comfyapi(cls, target_voice_reference)),
        }
        return await _run_audio_workflow(cls, "audio.chatterbox-voice-conversion.v1", ComfyCloudWorkflowInputs(assets=assets, seed=seed))


class ComfyCloudMelBandRoFormerStemSeparationNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _audio_schema(
            "ComfyCloudMelBandRoFormerStemSeparationNode",
            "MelBandRoFormer Stem Separation",
            [IO.Audio.Input("audio")],
            [IO.Audio.Output("vocals"), IO.Audio.Output("instruments")],
        )

    @classmethod
    async def execute(cls, audio: Input.Audio) -> IO.NodeOutput:
        _validate_audio_duration("Audio", audio, 0.5, 600)
        return await _run_audio_workflow(cls, "audio.melbandroformer-stem-separation.v1", ComfyCloudWorkflowInputs(assets=await _audio_asset(cls, "audio", audio)), ("vocals", "instruments"))


async def _run_3d_workflow(cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs, file_format: str) -> IO.NodeOutput:
    task = await sync_op(cls, _GENERATE_ENDPOINT, response_model=ComfyCloudGenerateResponse, data=ComfyCloudGenerateRequest(workflow=workflow, inputs=inputs))
    result = await _poll_task(cls, task.task_id)
    if not result.output_url:
        raise RuntimeError("Comfy Cloud task completed without an output URL.")
    return IO.NodeOutput(
        await download_url_to_file_3d(
            _validated_output_url(result.output_url),
            file_format,
            timeout=_OUTPUT_DOWNLOAD_TIMEOUT,
            cls=cls,
            allow_redirects=False,
        )
    )


def _3d_schema(node_id: str, display_name: str, inputs: list[IO.Input], output: IO.Output) -> IO.Schema:
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        category="partner/3d/Comfy Cloud",
        description="Runs this 3D workflow on Comfy Cloud and returns the generated 3D file."
        + _COMFY_CLOUD_RATE_DESCRIPTION,
        inputs=_with_input_sockets(inputs),
        outputs=[output],
        hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
        is_api_node=True,
        price_badge=_COMFY_CLOUD_PRICE_BADGE,
    )


async def _image_asset(cls: type[IO.ComfyNode], name: str, image: Input.Image, wait_label: str | None = None) -> dict[str, ComfyCloudAssetInput]:
    if get_number_of_images(image) != 1:
        raise ValueError(f"Exactly one {name.replace('_', ' ')} is required.")
    _validate_image_upload(image)
    kwargs = {"total_pixels": None}
    if wait_label is not None:
        kwargs["wait_label"] = wait_label
    return {name: ComfyCloudAssetInput(type="IMAGE", url=await upload_image_to_comfyapi(cls, image, **kwargs))}


class ComfyCloudTripoSplatImageToGaussianSplatNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _3d_schema(
            "ComfyCloudTripoSplatImageToGaussianSplatNode",
            "TripoSplat Image to Gaussian Splat",
            [
                IO.Image.Input("image"),
                IO.Boolean.Input("remove_background", default=True),
                IO.Int.Input("seed", default=46, min=0, max=_UINT64_MAX, control_after_generate=True),
                IO.Int.Input("gaussian_count", default=262144, min=32768, max=262144),
            ],
            IO.File3DSPZ.Output(tooltip="SPZ Gaussian splat (.spz; POC MIME application/octet-stream)."),
        )

    @classmethod
    async def execute(cls, image: Input.Image, remove_background: bool, seed: int, gaussian_count: int) -> IO.NodeOutput:
        _validate_node_inputs(cls, locals())
        return await _run_3d_workflow(cls, "3d.triposplat-image-to-gaussian-splat.v1", ComfyCloudWorkflowInputs(assets=await _image_asset(cls, "image", image), remove_background=remove_background, seed=seed, gaussian_count=gaussian_count), "spz")


class ComfyCloudHunyuan3D21ImageTo3DNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _3d_schema(
            "ComfyCloudHunyuan3D21ImageTo3DNode",
            "Hunyuan3D 2.1 Image to 3D",
            [IO.Image.Input("image"), IO.Int.Input("seed", default=952805179515179, min=0, max=_UINT64_MAX, control_after_generate=True)],
            IO.File3DGLB.Output(),
        )

    @classmethod
    async def execute(cls, image: Input.Image, seed: int) -> IO.NodeOutput:
        _validate_node_inputs(cls, locals())
        return await _run_3d_workflow(cls, "3d.hunyuan3d-2-1-image-to-3d.v1", ComfyCloudWorkflowInputs(assets=await _image_asset(cls, "image", image), seed=seed), "glb")


class ComfyCloudHunyuan3DMultiViewTo3DNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _3d_schema(
            "ComfyCloudHunyuan3DMultiViewTo3DNode",
            "Hunyuan3D Multi-View to 3D",
            [IO.Image.Input("front_image"), IO.Image.Input("back_image"), IO.Int.Input("seed", default=502126049100058, min=0, max=_UINT64_MAX, control_after_generate=True)],
            IO.File3DGLB.Output(),
        )

    @classmethod
    async def execute(cls, front_image: Input.Image, back_image: Input.Image, seed: int) -> IO.NodeOutput:
        _validate_node_inputs(cls, locals())
        if get_number_of_images(front_image) != 1 or get_number_of_images(back_image) != 1:
            raise ValueError("Exactly one front image and one back image are required.")
        assets = await _image_asset(cls, "front_image", front_image, "Uploading front image")
        assets.update(await _image_asset(cls, "back_image", back_image, "Uploading back image"))
        return await _run_3d_workflow(cls, "3d.hunyuan3d-multiview-to-3d.v1", ComfyCloudWorkflowInputs(assets=assets, seed=seed), "glb")


class ComfyCloudMoGe2PhotoToTexturedMeshNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _3d_schema(
            "ComfyCloudMoGe2PhotoToTexturedMeshNode",
            "MoGe 2 Photo to Textured Mesh",
            [
                IO.Image.Input("image"),
                IO.Float.Input("fov_degrees", default=0, min=0, max=170, step=0.1, tooltip="0 selects automatic field-of-view estimation."),
                IO.Int.Input("detail", default=9, min=0, max=9),
                IO.Int.Input("mesh_decimation", default=1, min=1, max=8),
                IO.Float.Input("gap_threshold", default=0.04, min=0, max=1, step=0.01),
                IO.Boolean.Input("texture", default=True),
            ],
            IO.File3DGLB.Output(),
        )

    @classmethod
    async def execute(cls, image: Input.Image, fov_degrees: float, detail: int, mesh_decimation: int, gap_threshold: float, texture: bool) -> IO.NodeOutput:
        _validate_node_inputs(cls, locals())
        inputs = ComfyCloudWorkflowInputs(assets=await _image_asset(cls, "image", image), fov_degrees=fov_degrees, detail=detail, mesh_decimation=mesh_decimation, gap_threshold=gap_threshold, texture=texture)
        return await _run_3d_workflow(cls, "3d.moge-2-photo-to-textured-mesh.v1", inputs, "glb")


class ComfyCloudMoGe2PanoramaTo3DSceneNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _3d_schema(
            "ComfyCloudMoGe2PanoramaTo3DSceneNode",
            "MoGe 2 Panorama to 3D Scene",
            [
                IO.Image.Input("panorama", tooltip="Equirectangular panorama."),
                IO.Int.Input("detail", default=5, min=0, max=9),
                IO.Int.Input("split_resolution", default=512, min=256, max=1024),
                IO.Int.Input("merge_resolution", default=1024, min=256, max=8192),
                IO.Int.Input("mesh_decimation", default=1, min=1, max=8),
                IO.Float.Input("gap_threshold", default=0.04, min=0, max=1, step=0.01),
                IO.Boolean.Input("texture", default=True),
            ],
            IO.File3DGLB.Output(),
        )

    @classmethod
    async def execute(cls, panorama: Input.Image, detail: int, split_resolution: int, merge_resolution: int, mesh_decimation: int, gap_threshold: float, texture: bool) -> IO.NodeOutput:
        _validate_node_inputs(cls, locals())
        inputs = ComfyCloudWorkflowInputs(assets=await _image_asset(cls, "panorama", panorama), detail=detail, split_resolution=split_resolution, merge_resolution=merge_resolution, mesh_decimation=mesh_decimation, gap_threshold=gap_threshold, texture=texture)
        return await _run_3d_workflow(cls, "3d.moge-2-panorama-to-3d-scene.v1", inputs, "glb")


class ComfyCloudExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ComfyCloudTextToImageNode,
            ComfyCloudTextToVideoNode,
            ComfyCloudImageToVideoNode,
            ComfyCloudImageEditNode,
            ComfyCloudIdeogram4DesignNode,
            ComfyCloudKrea2CreativeImageNode,
            ComfyCloudMageFlowImageNode,
            ComfyCloudFlux2ReferenceEditNode,
            ComfyCloudQwenImageEdit2511Node,
            ComfyCloudSeedVR2ImageUpscaleNode,
            ComfyCloudMiniMaxH3TextSoundNode,
            ComfyCloudMiniMaxH3ImageSoundNode,
            ComfyCloudLTX23ImageAudioPerformanceNode,
            ComfyCloudLTX23FirstLastFrameNode,
            ComfyCloudWan22FirstLastFrameNode,
            ComfyCloudSCAIL2CharacterReplacementNode,
            ComfyCloudACEStep15XLTurboNode,
            ComfyCloudStableAudio3MediumNode,
            ComfyCloudChatterboxMultilingualVoiceCloneNode,
            ComfyCloudChatterboxDialogueNode,
            ComfyCloudChatterboxVoiceConversionNode,
            ComfyCloudMelBandRoFormerStemSeparationNode,
            ComfyCloudTripoSplatImageToGaussianSplatNode,
            ComfyCloudHunyuan3D21ImageTo3DNode,
            ComfyCloudHunyuan3DMultiViewTo3DNode,
            ComfyCloudMoGe2PhotoToTexturedMeshNode,
            ComfyCloudMoGe2PanoramaTo3DSceneNode,
        ]


async def comfy_entrypoint() -> ComfyCloudExtension:
    return ComfyCloudExtension()

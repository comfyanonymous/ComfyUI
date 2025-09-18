import logging
from enum import Enum
from typing import Any, Callable, Optional, Literal, TypeVar
from typing_extensions import override

import torch
from pydantic import BaseModel, Field

from comfy_api.latest import ComfyExtension, io as comfy_io
from comfy_api_nodes.util.validation_utils import (
    validate_aspect_ratio_closeness,
    validate_image_dimensions,
    validate_image_aspect_ratio_range,
    get_number_of_images,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import download_url_to_video_output, upload_images_to_comfyapi


VIDU_TEXT_TO_VIDEO = "/proxy/vidu/text2video"
VIDU_IMAGE_TO_VIDEO = "/proxy/vidu/img2video"
VIDU_REFERENCE_VIDEO = "/proxy/vidu/reference2video"
VIDU_START_END_VIDEO = "/proxy/vidu/start-end2video"
VIDU_GET_GENERATION_STATUS = "/proxy/vidu/tasks/%s/creations"

R = TypeVar("R")

class VideoModelName(str, Enum):
    vidu_q1 = 'viduq1'


class AspectRatio(str, Enum):
    r_16_9 = "16:9"
    r_9_16 = "9:16"
    r_1_1 = "1:1"


class Resolution(str, Enum):
    r_1080p = "1080p"


class MovementAmplitude(str, Enum):
    auto = "auto"
    small = "small"
    medium = "medium"
    large = "large"


class TaskCreationRequest(BaseModel):
    model: VideoModelName = VideoModelName.vidu_q1
    prompt: Optional[str] = Field(None, max_length=1500)
    duration: Optional[Literal[5]] = 5
    seed: Optional[int] = Field(0, ge=0, le=2147483647)
    aspect_ratio: Optional[AspectRatio] = AspectRatio.r_16_9
    resolution: Optional[Resolution] = Resolution.r_1080p
    movement_amplitude: Optional[MovementAmplitude] = MovementAmplitude.auto
    images: Optional[list[str]] = Field(None, description="Base64 encoded string or image URL")


class TaskStatus(str, Enum):
    created = "created"
    queueing = "queueing"
    processing = "processing"
    success = "success"
    failed = "failed"


class TaskCreationResponse(BaseModel):
    task_id: str = Field(...)
    state: TaskStatus = Field(...)
    created_at: str = Field(...)
    code: Optional[int] = Field(None, description="Error code")


class TaskResult(BaseModel):
    id: str = Field(..., description="Creation id")
    url: str = Field(..., description="The URL of the generated results, valid for one hour")
    cover_url: str = Field(..., description="The cover URL of the generated results, valid for one hour")


class TaskStatusResponse(BaseModel):
    state: TaskStatus = Field(...)
    err_code: Optional[str] = Field(None)
    creations: list[TaskResult] = Field(..., description="Generated results")


async def poll_until_finished(
    auth_kwargs: dict[str, str],
    api_endpoint: ApiEndpoint[Any, R],
    result_url_extractor: Optional[Callable[[R], str]] = None,
    estimated_duration: Optional[int] = None,
    node_id: Optional[str] = None,
) -> R:
    return await PollingOperation(
        poll_endpoint=api_endpoint,
        completed_statuses=[TaskStatus.success.value],
        failed_statuses=[TaskStatus.failed.value],
        status_extractor=lambda response: response.state.value,
        auth_kwargs=auth_kwargs,
        result_url_extractor=result_url_extractor,
        estimated_duration=estimated_duration,
        node_id=node_id,
        poll_interval=16.0,
        max_poll_attempts=256,
    ).execute()


def get_video_url_from_response(response) -> Optional[str]:
    if response.creations:
        return response.creations[0].url
    return None


def get_video_from_response(response) -> TaskResult:
    if not response.creations:
        error_msg = f"Vidu request does not contain results. State: {response.state}, Error Code: {response.err_code}"
        logging.info(error_msg)
        raise RuntimeError(error_msg)
    logging.info("Vidu task %s succeeded. Video URL: %s", response.creations[0].id, response.creations[0].url)
    return response.creations[0]


async def execute_task(
    vidu_endpoint: str,
    auth_kwargs: Optional[dict[str, str]],
    payload: TaskCreationRequest,
    estimated_duration: int,
    node_id: str,
) -> R:
    response = await SynchronousOperation(
        endpoint=ApiEndpoint(
            path=vidu_endpoint,
            method=HttpMethod.POST,
            request_model=TaskCreationRequest,
            response_model=TaskCreationResponse,
        ),
        request=payload,
        auth_kwargs=auth_kwargs,
    ).execute()
    if response.state == TaskStatus.failed:
        error_msg = f"Vidu request failed. Code: {response.code}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    return await poll_until_finished(
        auth_kwargs,
        ApiEndpoint(
            path=VIDU_GET_GENERATION_STATUS % response.task_id,
            method=HttpMethod.GET,
            request_model=EmptyRequest,
            response_model=TaskStatusResponse,
        ),
        result_url_extractor=get_video_url_from_response,
        estimated_duration=estimated_duration,
        node_id=node_id,
    )


class ViduTextToVideoNode(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="ViduTextToVideoNode",
            display_name="Vidu Text To Video Generation",
            category="api node/video/Vidu",
            description="Generate video from text prompt",
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=[model.value for model in VideoModelName],
                    default=VideoModelName.vidu_q1.value,
                    tooltip="Model name",
                ),
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation",
                ),
                comfy_io.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "aspect_ratio",
                    options=[model.value for model in AspectRatio],
                    default=AspectRatio.r_16_9.value,
                    tooltip="The aspect ratio of the output video",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "resolution",
                    options=[model.value for model in Resolution],
                    default=Resolution.r_1080p.value,
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "movement_amplitude",
                    options=[model.value for model in MovementAmplitude],
                    default=MovementAmplitude.auto.value,
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
                ),
            ],
            outputs=[
                comfy_io.Video.Output(),
            ],
            hidden=[
                comfy_io.Hidden.auth_token_comfy_org,
                comfy_io.Hidden.api_key_comfy_org,
                comfy_io.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        duration: int,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        movement_amplitude: str,
    ) -> comfy_io.NodeOutput:
        if not prompt:
            raise ValueError("The prompt field is required and cannot be empty.")
        payload = TaskCreationRequest(
            model_name=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        results = await execute_task(VIDU_TEXT_TO_VIDEO, auth, payload, 320, cls.hidden.unique_id)
        return comfy_io.NodeOutput(await download_url_to_video_output(get_video_from_response(results).url))


class ViduImageToVideoNode(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="ViduImageToVideoNode",
            display_name="Vidu Image To Video Generation",
            category="api node/video/Vidu",
            description="Generate video from image and optional prompt",
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=[model.value for model in VideoModelName],
                    default=VideoModelName.vidu_q1.value,
                    tooltip="Model name",
                ),
                comfy_io.Image.Input(
                    "image",
                    tooltip="An image to be used as the start frame of the generated video",
                ),
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="A textual description for video generation",
                    optional=True,
                ),
                comfy_io.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "resolution",
                    options=[model.value for model in Resolution],
                    default=Resolution.r_1080p.value,
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "movement_amplitude",
                    options=[model.value for model in MovementAmplitude],
                    default=MovementAmplitude.auto.value,
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
                ),
            ],
            outputs=[
                comfy_io.Video.Output(),
            ],
            hidden=[
                comfy_io.Hidden.auth_token_comfy_org,
                comfy_io.Hidden.api_key_comfy_org,
                comfy_io.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: torch.Tensor,
        prompt: str,
        duration: int,
        seed: int,
        resolution: str,
        movement_amplitude: str,
    ) -> comfy_io.NodeOutput:
        if get_number_of_images(image) > 1:
            raise ValueError("Only one input image is allowed.")
        validate_image_aspect_ratio_range(image, (1, 4), (4, 1))
        payload = TaskCreationRequest(
            model_name=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        payload.images = await upload_images_to_comfyapi(
            image,
            max_images=1,
            mime_type="image/png",
            auth_kwargs=auth,
        )
        results = await execute_task(VIDU_IMAGE_TO_VIDEO, auth, payload, 120, cls.hidden.unique_id)
        return comfy_io.NodeOutput(await download_url_to_video_output(get_video_from_response(results).url))


class ViduReferenceVideoNode(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="ViduReferenceVideoNode",
            display_name="Vidu Reference To Video Generation",
            category="api node/video/Vidu",
            description="Generate video from multiple images and prompt",
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=[model.value for model in VideoModelName],
                    default=VideoModelName.vidu_q1.value,
                    tooltip="Model name",
                ),
                comfy_io.Image.Input(
                    "images",
                    tooltip="Images to use as references to generate a video with consistent subjects (max 7 images).",
                ),
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation",
                ),
                comfy_io.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "aspect_ratio",
                    options=[model.value for model in AspectRatio],
                    default=AspectRatio.r_16_9.value,
                    tooltip="The aspect ratio of the output video",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "resolution",
                    options=[model.value for model in Resolution],
                    default=Resolution.r_1080p.value,
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "movement_amplitude",
                    options=[model.value for model in MovementAmplitude],
                    default=MovementAmplitude.auto.value,
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
                ),
            ],
            outputs=[
                comfy_io.Video.Output(),
            ],
            hidden=[
                comfy_io.Hidden.auth_token_comfy_org,
                comfy_io.Hidden.api_key_comfy_org,
                comfy_io.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        images: torch.Tensor,
        prompt: str,
        duration: int,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        movement_amplitude: str,
    ) -> comfy_io.NodeOutput:
        if not prompt:
            raise ValueError("The prompt field is required and cannot be empty.")
        a = get_number_of_images(images)
        if a > 7:
            raise ValueError("Too many images, maximum allowed is 7.")
        for image in images:
            validate_image_aspect_ratio_range(image, (1, 4), (4, 1))
            validate_image_dimensions(image, min_width=128, min_height=128)
        payload = TaskCreationRequest(
            model_name=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        payload.images = await upload_images_to_comfyapi(
            images,
            max_images=7,
            mime_type="image/png",
            auth_kwargs=auth,
        )
        results = await execute_task(VIDU_REFERENCE_VIDEO, auth, payload, 120, cls.hidden.unique_id)
        return comfy_io.NodeOutput(await download_url_to_video_output(get_video_from_response(results).url))


class ViduStartEndToVideoNode(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="ViduStartEndToVideoNode",
            display_name="Vidu Start End To Video Generation",
            category="api node/video/Vidu",
            description="Generate a video from start and end frames and a prompt",
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=[model.value for model in VideoModelName],
                    default=VideoModelName.vidu_q1.value,
                    tooltip="Model name",
                ),
                comfy_io.Image.Input(
                    "first_frame",
                    tooltip="Start frame",
                ),
                comfy_io.Image.Input(
                    "end_frame",
                    tooltip="End frame",
                ),
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation",
                    optional=True,
                ),
                comfy_io.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "resolution",
                    options=[model.value for model in Resolution],
                    default=Resolution.r_1080p.value,
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "movement_amplitude",
                    options=[model.value for model in MovementAmplitude],
                    default=MovementAmplitude.auto.value,
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
                ),
            ],
            outputs=[
                comfy_io.Video.Output(),
            ],
            hidden=[
                comfy_io.Hidden.auth_token_comfy_org,
                comfy_io.Hidden.api_key_comfy_org,
                comfy_io.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        first_frame: torch.Tensor,
        end_frame: torch.Tensor,
        prompt: str,
        duration: int,
        seed: int,
        resolution: str,
        movement_amplitude: str,
    ) -> comfy_io.NodeOutput:
        validate_aspect_ratio_closeness(first_frame, end_frame, min_rel=0.8, max_rel=1.25, strict=False)
        payload = TaskCreationRequest(
            model_name=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        payload.images = [
            (await upload_images_to_comfyapi(frame, max_images=1, mime_type="image/png", auth_kwargs=auth))[0]
            for frame in (first_frame, end_frame)
        ]
        results = await execute_task(VIDU_START_END_VIDEO, auth, payload, 96, cls.hidden.unique_id)
        return comfy_io.NodeOutput(await download_url_to_video_output(get_video_from_response(results).url))


class ViduExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[comfy_io.ComfyNode]]:
        return [
            ViduTextToVideoNode,
            ViduImageToVideoNode,
            ViduReferenceVideoNode,
            ViduStartEndToVideoNode,
        ]

async def comfy_entrypoint() -> ViduExtension:
    return ViduExtension()

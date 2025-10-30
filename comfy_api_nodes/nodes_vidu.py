import logging
from enum import Enum
from typing import Literal, Optional, TypeVar

import torch
from pydantic import BaseModel, Field
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    validate_image_aspect_ratio,
    validate_image_dimensions,
    validate_images_aspect_ratio_closeness,
)

VIDU_TEXT_TO_VIDEO = "/proxy/vidu/text2video"
VIDU_IMAGE_TO_VIDEO = "/proxy/vidu/img2video"
VIDU_REFERENCE_VIDEO = "/proxy/vidu/reference2video"
VIDU_START_END_VIDEO = "/proxy/vidu/start-end2video"
VIDU_GET_GENERATION_STATUS = "/proxy/vidu/tasks/%s/creations"

R = TypeVar("R")


class VideoModelName(str, Enum):
    vidu_q1 = "viduq1"


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


class TaskCreationResponse(BaseModel):
    task_id: str = Field(...)
    state: str = Field(...)
    created_at: str = Field(...)
    code: Optional[int] = Field(None, description="Error code")


class TaskResult(BaseModel):
    id: str = Field(..., description="Creation id")
    url: str = Field(..., description="The URL of the generated results, valid for one hour")
    cover_url: str = Field(..., description="The cover URL of the generated results, valid for one hour")


class TaskStatusResponse(BaseModel):
    state: str = Field(...)
    err_code: Optional[str] = Field(None)
    creations: list[TaskResult] = Field(..., description="Generated results")


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
    cls: type[IO.ComfyNode],
    vidu_endpoint: str,
    payload: TaskCreationRequest,
    estimated_duration: int,
) -> R:
    response = await sync_op(
        cls,
        endpoint=ApiEndpoint(path=vidu_endpoint, method="POST"),
        response_model=TaskCreationResponse,
        data=payload,
    )
    if response.state == "failed":
        error_msg = f"Vidu request failed. Code: {response.code}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    return await poll_op(
        cls,
        ApiEndpoint(path=VIDU_GET_GENERATION_STATUS % response.task_id),
        response_model=TaskStatusResponse,
        status_extractor=lambda r: r.state,
        estimated_duration=estimated_duration,
    )


class ViduTextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduTextToVideoNode",
            display_name="Vidu Text To Video Generation",
            category="api node/video/Vidu",
            description="Generate video from text prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=VideoModelName,
                    default=VideoModelName.vidu_q1,
                    tooltip="Model name",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=AspectRatio,
                    default=AspectRatio.r_16_9,
                    tooltip="The aspect ratio of the output video",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=Resolution,
                    default=Resolution.r_1080p,
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=MovementAmplitude,
                    default=MovementAmplitude.auto,
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
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
    ) -> IO.NodeOutput:
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
        results = await execute_task(cls, VIDU_TEXT_TO_VIDEO, payload, 320)
        return IO.NodeOutput(await download_url_to_video_output(get_video_from_response(results).url))


class ViduImageToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduImageToVideoNode",
            display_name="Vidu Image To Video Generation",
            category="api node/video/Vidu",
            description="Generate video from image and optional prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=VideoModelName,
                    default=VideoModelName.vidu_q1,
                    tooltip="Model name",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="An image to be used as the start frame of the generated video",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="A textual description for video generation",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=Resolution,
                    default=Resolution.r_1080p,
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=MovementAmplitude,
                    default=MovementAmplitude.auto.value,
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
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
    ) -> IO.NodeOutput:
        if get_number_of_images(image) > 1:
            raise ValueError("Only one input image is allowed.")
        validate_image_aspect_ratio(image, (1, 4), (4, 1))
        payload = TaskCreationRequest(
            model_name=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        payload.images = await upload_images_to_comfyapi(
            cls,
            image,
            max_images=1,
            mime_type="image/png",
        )
        results = await execute_task(cls, VIDU_IMAGE_TO_VIDEO, payload, 120)
        return IO.NodeOutput(await download_url_to_video_output(get_video_from_response(results).url))


class ViduReferenceVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduReferenceVideoNode",
            display_name="Vidu Reference To Video Generation",
            category="api node/video/Vidu",
            description="Generate video from multiple images and prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=VideoModelName,
                    default=VideoModelName.vidu_q1,
                    tooltip="Model name",
                ),
                IO.Image.Input(
                    "images",
                    tooltip="Images to use as references to generate a video with consistent subjects (max 7 images).",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=AspectRatio,
                    default=AspectRatio.r_16_9,
                    tooltip="The aspect ratio of the output video",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=[model.value for model in Resolution],
                    default=Resolution.r_1080p.value,
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=[model.value for model in MovementAmplitude],
                    default=MovementAmplitude.auto.value,
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
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
    ) -> IO.NodeOutput:
        if not prompt:
            raise ValueError("The prompt field is required and cannot be empty.")
        a = get_number_of_images(images)
        if a > 7:
            raise ValueError("Too many images, maximum allowed is 7.")
        for image in images:
            validate_image_aspect_ratio(image, (1, 4), (4, 1))
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
        payload.images = await upload_images_to_comfyapi(
            cls,
            images,
            max_images=7,
            mime_type="image/png",
        )
        results = await execute_task(cls, VIDU_REFERENCE_VIDEO, payload, 120)
        return IO.NodeOutput(await download_url_to_video_output(get_video_from_response(results).url))


class ViduStartEndToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduStartEndToVideoNode",
            display_name="Vidu Start End To Video Generation",
            category="api node/video/Vidu",
            description="Generate a video from start and end frames and a prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=[model.value for model in VideoModelName],
                    default=VideoModelName.vidu_q1.value,
                    tooltip="Model name",
                ),
                IO.Image.Input(
                    "first_frame",
                    tooltip="Start frame",
                ),
                IO.Image.Input(
                    "end_frame",
                    tooltip="End frame",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=[model.value for model in Resolution],
                    default=Resolution.r_1080p.value,
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=[model.value for model in MovementAmplitude],
                    default=MovementAmplitude.auto.value,
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
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
    ) -> IO.NodeOutput:
        validate_images_aspect_ratio_closeness(first_frame, end_frame, min_rel=0.8, max_rel=1.25, strict=False)
        payload = TaskCreationRequest(
            model_name=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        payload.images = [
            (await upload_images_to_comfyapi(cls, frame, max_images=1, mime_type="image/png"))[0]
            for frame in (first_frame, end_frame)
        ]
        results = await execute_task(cls, VIDU_START_END_VIDEO, payload, 96)
        return IO.NodeOutput(await download_url_to_video_output(get_video_from_response(results).url))


class ViduExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ViduTextToVideoNode,
            ViduImageToVideoNode,
            ViduReferenceVideoNode,
            ViduStartEndToVideoNode,
        ]


async def comfy_entrypoint() -> ViduExtension:
    return ViduExtension()

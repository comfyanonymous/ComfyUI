"""Kling API Nodes

For source of truth on the allowed permutations of request fields, please reference:
- [Compatibility Table](https://app.klingai.com/global/dev/document-api/apiReference/model/skillsMap)
"""

from __future__ import annotations
from typing import Optional, TypeVar, Any
from collections.abc import Callable
import math
import logging

import torch

from comfy_api_nodes.apis import (
    KlingTaskStatus,
    KlingCameraControl,
    KlingCameraConfig,
    KlingCameraControlType,
    KlingVideoGenDuration,
    KlingVideoGenMode,
    KlingVideoGenAspectRatio,
    KlingVideoGenModelName,
    KlingText2VideoRequest,
    KlingText2VideoResponse,
    KlingImage2VideoRequest,
    KlingImage2VideoResponse,
    KlingVideoExtendRequest,
    KlingVideoExtendResponse,
    KlingLipSyncVoiceLanguage,
    KlingLipSyncInputObject,
    KlingLipSyncRequest,
    KlingLipSyncResponse,
    KlingVirtualTryOnModelName,
    KlingVirtualTryOnRequest,
    KlingVirtualTryOnResponse,
    KlingVideoResult,
    KlingImageResult,
    KlingImageGenerationsRequest,
    KlingImageGenerationsResponse,
    KlingImageGenImageReferenceType,
    KlingImageGenModelName,
    KlingImageGenAspectRatio,
    KlingVideoEffectsRequest,
    KlingVideoEffectsResponse,
    KlingDualCharacterEffectsScene,
    KlingSingleImageEffectsScene,
    KlingDualCharacterEffectInput,
    KlingSingleImageEffectInput,
    KlingCharacterEffectModelName,
    KlingSingleImageEffectModelName,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import (
    tensor_to_base64_string,
    download_url_to_video_output,
    upload_video_to_comfyapi,
    upload_audio_to_comfyapi,
    download_url_to_image_tensor,
)
from comfy_api_nodes.mapper_utils import model_field_to_node_input
from comfy_api_nodes.util.validation_utils import (
    validate_image_dimensions,
    validate_image_aspect_ratio,
    validate_video_dimensions,
    validate_video_duration,
)
from comfy_api.input.basic_types import AudioInput
from comfy_api.input.video_types import VideoInput
from comfy_api.input_impl import VideoFromFile
from comfy.comfy_types.node_typing import IO, InputTypeOptions, ComfyNodeABC

KLING_API_VERSION = "v1"
PATH_TEXT_TO_VIDEO = f"/proxy/kling/{KLING_API_VERSION}/videos/text2video"
PATH_IMAGE_TO_VIDEO = f"/proxy/kling/{KLING_API_VERSION}/videos/image2video"
PATH_VIDEO_EXTEND = f"/proxy/kling/{KLING_API_VERSION}/videos/video-extend"
PATH_LIP_SYNC = f"/proxy/kling/{KLING_API_VERSION}/videos/lip-sync"
PATH_VIDEO_EFFECTS = f"/proxy/kling/{KLING_API_VERSION}/videos/effects"
PATH_CHARACTER_IMAGE = f"/proxy/kling/{KLING_API_VERSION}/images/generations"
PATH_VIRTUAL_TRY_ON = f"/proxy/kling/{KLING_API_VERSION}/images/kolors-virtual-try-on"
PATH_IMAGE_GENERATIONS = f"/proxy/kling/{KLING_API_VERSION}/images/generations"

MAX_PROMPT_LENGTH_T2V = 2500
MAX_PROMPT_LENGTH_I2V = 500
MAX_PROMPT_LENGTH_IMAGE_GEN = 500
MAX_NEGATIVE_PROMPT_LENGTH_IMAGE_GEN = 200
MAX_PROMPT_LENGTH_LIP_SYNC = 120

AVERAGE_DURATION_T2V = 319
AVERAGE_DURATION_I2V = 164
AVERAGE_DURATION_LIP_SYNC = 455
AVERAGE_DURATION_VIRTUAL_TRY_ON = 19
AVERAGE_DURATION_IMAGE_GEN = 32
AVERAGE_DURATION_VIDEO_EFFECTS = 320
AVERAGE_DURATION_VIDEO_EXTEND = 320

R = TypeVar("R")


class KlingApiError(Exception):
    """Base exception for Kling API errors."""

    pass


async def poll_until_finished(
    auth_kwargs: dict[str, str],
    api_endpoint: ApiEndpoint[Any, R],
    result_url_extractor: Optional[Callable[[R], str]] = None,
    estimated_duration: Optional[int] = None,
    node_id: Optional[str] = None,
) -> R:
    """Polls the Kling API endpoint until the task reaches a terminal state, then returns the response."""
    return await PollingOperation(
        poll_endpoint=api_endpoint,
        completed_statuses=[
            KlingTaskStatus.succeed.value,
        ],
        failed_statuses=[KlingTaskStatus.failed.value],
        status_extractor=lambda response: (
            response.data.task_status.value
            if response.data and response.data.task_status
            else None
        ),
        auth_kwargs=auth_kwargs,
        result_url_extractor=result_url_extractor,
        estimated_duration=estimated_duration,
        node_id=node_id,
        poll_interval=16.0,
        max_poll_attempts=256,
    ).execute()


def is_valid_camera_control_configs(configs: list[float]) -> bool:
    """Verifies that at least one camera control configuration is non-zero."""
    return any(not math.isclose(value, 0.0) for value in configs)


def is_valid_prompt(prompt: str) -> bool:
    """Verifies that the prompt is not empty."""
    return bool(prompt)


def is_valid_task_creation_response(response: KlingText2VideoResponse) -> bool:
    """Verifies that the initial response contains a task ID."""
    return bool(response.data.task_id)


def is_valid_video_response(response: KlingText2VideoResponse) -> bool:
    """Verifies that the response contains a task result with at least one video."""
    return (
        response.data is not None
        and response.data.task_result is not None
        and response.data.task_result.videos is not None
        and len(response.data.task_result.videos) > 0
    )


def is_valid_image_response(response: KlingVirtualTryOnResponse) -> bool:
    """Verifies that the response contains a task result with at least one image."""
    return (
        response.data is not None
        and response.data.task_result is not None
        and response.data.task_result.images is not None
        and len(response.data.task_result.images) > 0
    )


def validate_prompts(prompt: str, negative_prompt: str, max_length: int) -> bool:
    """Verifies that the positive prompt is not empty and that neither promt is too long."""
    if not prompt:
        raise ValueError("Positive prompt is empty")
    if len(prompt) > max_length:
        raise ValueError(f"Positive prompt is too long: {len(prompt)} characters")
    if negative_prompt and len(negative_prompt) > max_length:
        raise ValueError(
            f"Negative prompt is too long: {len(negative_prompt)} characters"
        )
    return True


def validate_task_creation_response(response) -> None:
    """Validates that the Kling task creation request was successful."""
    if not is_valid_task_creation_response(response):
        error_msg = f"Kling initial request failed. Code: {response.code}, Message: {response.message}, Data: {response.data}"
        logging.error(error_msg)
        raise KlingApiError(error_msg)


def validate_video_result_response(response) -> None:
    """Validates that the Kling task result contains a video."""
    if not is_valid_video_response(response):
        error_msg = f"Kling task {response.data.task_id} succeeded but no video data found in response."
        logging.error(f"Error: {error_msg}.\nResponse: {response}")
        raise KlingApiError(error_msg)


def validate_image_result_response(response) -> None:
    """Validates that the Kling task result contains an image."""
    if not is_valid_image_response(response):
        error_msg = f"Kling task {response.data.task_id} succeeded but no image data found in response."
        logging.error(f"Error: {error_msg}.\nResponse: {response}")
        raise KlingApiError(error_msg)


def validate_input_image(image: torch.Tensor) -> None:
    """
    Validates the input image adheres to the expectations of the Kling API:
    - The image resolution should not be less than 300*300px
    - The aspect ratio of the image should be between 1:2.5 ~ 2.5:1

    See: https://app.klingai.com/global/dev/document-api/apiReference/model/imageToVideo
    """
    validate_image_dimensions(image, min_width=300, min_height=300)
    validate_image_aspect_ratio(image, min_aspect_ratio=1 / 2.5, max_aspect_ratio=2.5)


def get_camera_control_input_config(
    tooltip: str, default: float = 0.0
) -> tuple[IO, InputTypeOptions]:
    """Returns common InputTypeOptions for Kling camera control configurations."""
    input_config = {
        "default": default,
        "min": -10.0,
        "max": 10.0,
        "step": 0.25,
        "display": "slider",
        "tooltip": tooltip,
    }
    return IO.FLOAT, input_config


def get_video_from_response(response) -> KlingVideoResult:
    """Returns the first video object from the Kling video generation task result.
    Will raise an error if the response is not valid.
    """
    video = response.data.task_result.videos[0]
    logging.info(
        "Kling task %s succeeded. Video URL: %s", response.data.task_id, video.url
    )
    return video


def get_video_url_from_response(response) -> Optional[str]:
    """Returns the first video url from the Kling video generation task result.
    Will not raise an error if the response is not valid.
    """
    if response and is_valid_video_response(response):
        return str(get_video_from_response(response).url)
    else:
        return None


def get_images_from_response(response) -> list[KlingImageResult]:
    """Returns the list of image objects from the Kling image generation task result.
    Will raise an error if the response is not valid.
    """
    images = response.data.task_result.images
    logging.info("Kling task %s succeeded. Images: %s", response.data.task_id, images)
    return images


def get_images_urls_from_response(response) -> Optional[str]:
    """Returns the list of image urls from the Kling image generation task result.
    Will not raise an error if the response is not valid. If there is only one image, returns the url as a string. If there are multiple images, returns a list of urls.
    """
    if response and is_valid_image_response(response):
        images = get_images_from_response(response)
        image_urls = [str(image.url) for image in images]
        return "\n".join(image_urls)
    else:
        return None


async def video_result_to_node_output(
    video: KlingVideoResult,
) -> tuple[VideoFromFile, str, str]:
    """Converts a KlingVideoResult to a tuple of (VideoFromFile, str, str) to be used as a ComfyUI node output."""
    return (
        await download_url_to_video_output(str(video.url)),
        str(video.id),
        str(video.duration),
    )


async def image_result_to_node_output(
    images: list[KlingImageResult],
) -> torch.Tensor:
    """
    Converts a KlingImageResult to a tuple containing a [B, H, W, C] tensor.
    If multiple images are returned, they will be stacked along the batch dimension.
    """
    if len(images) == 1:
        return await download_url_to_image_tensor(str(images[0].url))
    else:
        return torch.cat([await download_url_to_image_tensor(str(image.url)) for image in images])


class KlingNodeBase(ComfyNodeABC):
    """Base class for Kling nodes."""

    FUNCTION = "api_call"
    CATEGORY = "api node/video/Kling"
    API_NODE = True


class KlingCameraControls(KlingNodeBase):
    """Kling Camera Controls Node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_control_type": model_field_to_node_input(
                    IO.COMBO,
                    KlingCameraControl,
                    "type",
                    enum_type=KlingCameraControlType,
                ),
                "horizontal_movement": get_camera_control_input_config(
                    "Controls camera's movement along horizontal axis (x-axis). Negative indicates left, positive indicates right"
                ),
                "vertical_movement": get_camera_control_input_config(
                    "Controls camera's movement along vertical axis (y-axis). Negative indicates downward, positive indicates upward."
                ),
                "pan": get_camera_control_input_config(
                    "Controls camera's rotation in vertical plane (x-axis). Negative indicates downward rotation, positive indicates upward rotation.",
                    default=0.5,
                ),
                "tilt": get_camera_control_input_config(
                    "Controls camera's rotation in horizontal plane (y-axis). Negative indicates left rotation, positive indicates right rotation.",
                ),
                "roll": get_camera_control_input_config(
                    "Controls camera's rolling amount (z-axis). Negative indicates counterclockwise, positive indicates clockwise.",
                ),
                "zoom": get_camera_control_input_config(
                    "Controls change in camera's focal length. Negative indicates narrower field of view, positive indicates wider field of view.",
                ),
            }
        }

    DESCRIPTION = "Allows specifying configuration options for Kling Camera Controls and motion control effects."
    RETURN_TYPES = ("CAMERA_CONTROL",)
    RETURN_NAMES = ("camera_control",)
    FUNCTION = "main"
    API_NODE = False  # This is just a helper node, it doesn't make an API call

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        horizontal_movement: float,
        vertical_movement: float,
        pan: float,
        tilt: float,
        roll: float,
        zoom: float,
    ) -> bool | str:
        if not is_valid_camera_control_configs(
            [
                horizontal_movement,
                vertical_movement,
                pan,
                tilt,
                roll,
                zoom,
            ]
        ):
            return "Invalid camera control configs: at least one of the values must be non-zero"
        return True

    def main(
        self,
        camera_control_type: str,
        horizontal_movement: float,
        vertical_movement: float,
        pan: float,
        tilt: float,
        roll: float,
        zoom: float,
    ) -> tuple[KlingCameraControl]:
        return (
            KlingCameraControl(
                type=KlingCameraControlType(camera_control_type),
                config=KlingCameraConfig(
                    horizontal=horizontal_movement,
                    vertical=vertical_movement,
                    pan=pan,
                    roll=roll,
                    tilt=tilt,
                    zoom=zoom,
                ),
            ),
        )


class KlingTextToVideoNode(KlingNodeBase):
    """Kling Text to Video Node"""

    @staticmethod
    def get_mode_string_mapping() -> dict[str, tuple[str, str, str]]:
        """
        Returns a mapping of mode strings to their corresponding (mode, duration, model_name) tuples.
        Only includes config combos that support the `image_tail` request field.

        See: [Kling API Docs Capability Map](https://app.klingai.com/global/dev/document-api/apiReference/model/skillsMap)
        """
        return {
            "standard mode / 5s duration / kling-v1": ("std", "5", "kling-v1"),
            "standard mode / 10s duration / kling-v1": ("std", "10", "kling-v1"),
            "pro mode / 5s duration / kling-v1": ("pro", "5", "kling-v1"),
            "pro mode / 10s duration / kling-v1": ("pro", "10", "kling-v1"),
            "standard mode / 5s duration / kling-v1-6": ("std", "5", "kling-v1-6"),
            "standard mode / 10s duration / kling-v1-6": ("std", "10", "kling-v1-6"),
            "pro mode / 5s duration / kling-v2-master": ("pro", "5", "kling-v2-master"),
            "pro mode / 10s duration / kling-v2-master": ("pro", "10", "kling-v2-master"),
            "standard mode / 5s duration / kling-v2-master": ("std", "5", "kling-v2-master"),
            "standard mode / 10s duration / kling-v2-master": ("std", "10", "kling-v2-master"),
        }

    @classmethod
    def INPUT_TYPES(s):
        modes = list(KlingTextToVideoNode.get_mode_string_mapping().keys())
        return {
            "required": {
                "prompt": model_field_to_node_input(
                    IO.STRING, KlingText2VideoRequest, "prompt", multiline=True
                ),
                "negative_prompt": model_field_to_node_input(
                    IO.STRING, KlingText2VideoRequest, "negative_prompt", multiline=True
                ),
                "cfg_scale": model_field_to_node_input(
                    IO.FLOAT,
                    KlingText2VideoRequest,
                    "cfg_scale",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                ),
                "aspect_ratio": model_field_to_node_input(
                    IO.COMBO,
                    KlingText2VideoRequest,
                    "aspect_ratio",
                    enum_type=KlingVideoGenAspectRatio,
                ),
                "mode": (
                    modes,
                    {
                        "default": modes[4],
                        "tooltip": "The configuration to use for the video generation following the format: mode / duration / model_name.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("VIDEO", "video_id", "duration")
    DESCRIPTION = "Kling Text to Video Node"

    async def get_response(
        self, task_id: str, auth_kwargs: dict[str, str], node_id: Optional[str] = None
    ) -> KlingText2VideoResponse:
        return await poll_until_finished(
            auth_kwargs,
            ApiEndpoint(
                path=f"{PATH_TEXT_TO_VIDEO}/{task_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=KlingText2VideoResponse,
            ),
            result_url_extractor=get_video_url_from_response,
            estimated_duration=AVERAGE_DURATION_T2V,
            node_id=node_id,
        )

    async def api_call(
        self,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        mode: str,
        aspect_ratio: str,
        camera_control: Optional[KlingCameraControl] = None,
        model_name: Optional[str] = None,
        duration: Optional[str] = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[VideoFromFile, str, str]:
        validate_prompts(prompt, negative_prompt, MAX_PROMPT_LENGTH_T2V)
        if model_name is None:
            mode, duration, model_name = self.get_mode_string_mapping()[mode]
        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_TEXT_TO_VIDEO,
                method=HttpMethod.POST,
                request_model=KlingText2VideoRequest,
                response_model=KlingText2VideoResponse,
            ),
            request=KlingText2VideoRequest(
                prompt=prompt if prompt else None,
                negative_prompt=negative_prompt if negative_prompt else None,
                duration=KlingVideoGenDuration(duration),
                mode=KlingVideoGenMode(mode),
                model_name=KlingVideoGenModelName(model_name),
                cfg_scale=cfg_scale,
                aspect_ratio=KlingVideoGenAspectRatio(aspect_ratio),
                camera_control=camera_control,
            ),
            auth_kwargs=kwargs,
        )

        task_creation_response = await initial_operation.execute()
        validate_task_creation_response(task_creation_response)

        task_id = task_creation_response.data.task_id
        final_response = await self.get_response(
            task_id, auth_kwargs=kwargs, node_id=unique_id
        )
        validate_video_result_response(final_response)

        video = get_video_from_response(final_response)
        return await video_result_to_node_output(video)


class KlingCameraControlT2VNode(KlingTextToVideoNode):
    """
    Kling Text to Video Camera Control Node. This node is a text to video node, but it supports controlling the camera.
    Duration, mode, and model_name request fields are hard-coded because camera control is only supported in pro mode with the kling-v1-5 model at 5s duration as of 2025-05-02.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": model_field_to_node_input(
                    IO.STRING, KlingText2VideoRequest, "prompt", multiline=True
                ),
                "negative_prompt": model_field_to_node_input(
                    IO.STRING,
                    KlingText2VideoRequest,
                    "negative_prompt",
                    multiline=True,
                ),
                "cfg_scale": model_field_to_node_input(
                    IO.FLOAT,
                    KlingText2VideoRequest,
                    "cfg_scale",
                    default=0.75,
                    min=0.0,
                    max=1.0,
                ),
                "aspect_ratio": model_field_to_node_input(
                    IO.COMBO,
                    KlingText2VideoRequest,
                    "aspect_ratio",
                    enum_type=KlingVideoGenAspectRatio,
                ),
                "camera_control": (
                    "CAMERA_CONTROL",
                    {
                        "tooltip": "Can be created using the Kling Camera Controls node. Controls the camera movement and motion during the video generation.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Transform text into cinematic videos with professional camera movements that simulate real-world cinematography. Control virtual camera actions including zoom, rotation, pan, tilt, and first-person view, while maintaining focus on your original text."

    async def api_call(
        self,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        aspect_ratio: str,
        camera_control: Optional[KlingCameraControl] = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        return await super().api_call(
            model_name=KlingVideoGenModelName.kling_v1,
            cfg_scale=cfg_scale,
            mode=KlingVideoGenMode.std,
            aspect_ratio=KlingVideoGenAspectRatio(aspect_ratio),
            duration=KlingVideoGenDuration.field_5,
            prompt=prompt,
            negative_prompt=negative_prompt,
            camera_control=camera_control,
            **kwargs,
        )


class KlingImage2VideoNode(KlingNodeBase):
    """Kling Image to Video Node"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_frame": model_field_to_node_input(
                    IO.IMAGE,
                    KlingImage2VideoRequest,
                    "image",
                    tooltip="The reference image used to generate the video.",
                ),
                "prompt": model_field_to_node_input(
                    IO.STRING, KlingImage2VideoRequest, "prompt", multiline=True
                ),
                "negative_prompt": model_field_to_node_input(
                    IO.STRING,
                    KlingImage2VideoRequest,
                    "negative_prompt",
                    multiline=True,
                ),
                "model_name": model_field_to_node_input(
                    IO.COMBO,
                    KlingImage2VideoRequest,
                    "model_name",
                    enum_type=KlingVideoGenModelName,
                ),
                "cfg_scale": model_field_to_node_input(
                    IO.FLOAT,
                    KlingImage2VideoRequest,
                    "cfg_scale",
                    default=0.8,
                    min=0.0,
                    max=1.0,
                ),
                "mode": model_field_to_node_input(
                    IO.COMBO,
                    KlingImage2VideoRequest,
                    "mode",
                    enum_type=KlingVideoGenMode,
                ),
                "aspect_ratio": model_field_to_node_input(
                    IO.COMBO,
                    KlingImage2VideoRequest,
                    "aspect_ratio",
                    enum_type=KlingVideoGenAspectRatio,
                ),
                "duration": model_field_to_node_input(
                    IO.COMBO,
                    KlingImage2VideoRequest,
                    "duration",
                    enum_type=KlingVideoGenDuration,
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("VIDEO", "video_id", "duration")
    DESCRIPTION = "Kling Image to Video Node"

    async def get_response(
        self, task_id: str, auth_kwargs: dict[str, str], node_id: Optional[str] = None
    ) -> KlingImage2VideoResponse:
        return await poll_until_finished(
            auth_kwargs,
            ApiEndpoint(
                path=f"{PATH_IMAGE_TO_VIDEO}/{task_id}",
                method=HttpMethod.GET,
                request_model=KlingImage2VideoRequest,
                response_model=KlingImage2VideoResponse,
            ),
            result_url_extractor=get_video_url_from_response,
            estimated_duration=AVERAGE_DURATION_I2V,
            node_id=node_id,
        )

    async def api_call(
        self,
        start_frame: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        model_name: str,
        cfg_scale: float,
        mode: str,
        aspect_ratio: str,
        duration: str,
        camera_control: Optional[KlingCameraControl] = None,
        end_frame: Optional[torch.Tensor] = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[VideoFromFile]:
        validate_prompts(prompt, negative_prompt, MAX_PROMPT_LENGTH_I2V)
        validate_input_image(start_frame)

        if camera_control is not None:
            # Camera control type for image 2 video is always `simple`
            camera_control.type = KlingCameraControlType.simple

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_IMAGE_TO_VIDEO,
                method=HttpMethod.POST,
                request_model=KlingImage2VideoRequest,
                response_model=KlingImage2VideoResponse,
            ),
            request=KlingImage2VideoRequest(
                model_name=KlingVideoGenModelName(model_name),
                image=tensor_to_base64_string(start_frame),
                image_tail=(
                    tensor_to_base64_string(end_frame)
                    if end_frame is not None
                    else None
                ),
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                cfg_scale=cfg_scale,
                mode=KlingVideoGenMode(mode),
                duration=KlingVideoGenDuration(duration),
                camera_control=camera_control,
            ),
            auth_kwargs=kwargs,
        )

        task_creation_response = await initial_operation.execute()
        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.data.task_id

        final_response = await self.get_response(
            task_id, auth_kwargs=kwargs, node_id=unique_id
        )
        validate_video_result_response(final_response)

        video = get_video_from_response(final_response)
        return await video_result_to_node_output(video)


class KlingCameraControlI2VNode(KlingImage2VideoNode):
    """
    Kling Image to Video Camera Control Node. This node is a image to video node, but it supports controlling the camera.
    Duration, mode, and model_name request fields are hard-coded because camera control is only supported in pro mode with the kling-v1-5 model at 5s duration as of 2025-05-02.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_frame": model_field_to_node_input(
                    IO.IMAGE, KlingImage2VideoRequest, "image"
                ),
                "prompt": model_field_to_node_input(
                    IO.STRING, KlingImage2VideoRequest, "prompt", multiline=True
                ),
                "negative_prompt": model_field_to_node_input(
                    IO.STRING,
                    KlingImage2VideoRequest,
                    "negative_prompt",
                    multiline=True,
                ),
                "cfg_scale": model_field_to_node_input(
                    IO.FLOAT,
                    KlingImage2VideoRequest,
                    "cfg_scale",
                    default=0.75,
                    min=0.0,
                    max=1.0,
                ),
                "aspect_ratio": model_field_to_node_input(
                    IO.COMBO,
                    KlingImage2VideoRequest,
                    "aspect_ratio",
                    enum_type=KlingVideoGenAspectRatio,
                ),
                "camera_control": (
                    "CAMERA_CONTROL",
                    {
                        "tooltip": "Can be created using the Kling Camera Controls node. Controls the camera movement and motion during the video generation.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Transform still images into cinematic videos with professional camera movements that simulate real-world cinematography. Control virtual camera actions including zoom, rotation, pan, tilt, and first-person view, while maintaining focus on your original image."

    async def api_call(
        self,
        start_frame: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        aspect_ratio: str,
        camera_control: KlingCameraControl,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        return await super().api_call(
            model_name=KlingVideoGenModelName.kling_v1_5,
            start_frame=start_frame,
            cfg_scale=cfg_scale,
            mode=KlingVideoGenMode.pro,
            aspect_ratio=KlingVideoGenAspectRatio(aspect_ratio),
            duration=KlingVideoGenDuration.field_5,
            prompt=prompt,
            negative_prompt=negative_prompt,
            camera_control=camera_control,
            unique_id=unique_id,
            **kwargs,
        )


class KlingStartEndFrameNode(KlingImage2VideoNode):
    """
    Kling First Last Frame Node. This node allows creation of a video from a first and last frame. It calls the normal image to video endpoint, but only allows the subset of input options that support the `image_tail` request field.
    """

    @staticmethod
    def get_mode_string_mapping() -> dict[str, tuple[str, str, str]]:
        """
        Returns a mapping of mode strings to their corresponding (mode, duration, model_name) tuples.
        Only includes config combos that support the `image_tail` request field.

        See: [Kling API Docs Capability Map](https://app.klingai.com/global/dev/document-api/apiReference/model/skillsMap)
        """
        return {
            "standard mode / 5s duration / kling-v1": ("std", "5", "kling-v1"),
            "pro mode / 5s duration / kling-v1": ("pro", "5", "kling-v1"),
            "pro mode / 5s duration / kling-v1-5": ("pro", "5", "kling-v1-5"),
            "pro mode / 10s duration / kling-v1-5": ("pro", "10", "kling-v1-5"),
            "pro mode / 5s duration / kling-v1-6": ("pro", "5", "kling-v1-6"),
            "pro mode / 10s duration / kling-v1-6": ("pro", "10", "kling-v1-6"),
        }

    @classmethod
    def INPUT_TYPES(s):
        modes = list(KlingStartEndFrameNode.get_mode_string_mapping().keys())
        return {
            "required": {
                "start_frame": model_field_to_node_input(
                    IO.IMAGE, KlingImage2VideoRequest, "image"
                ),
                "end_frame": model_field_to_node_input(
                    IO.IMAGE, KlingImage2VideoRequest, "image_tail"
                ),
                "prompt": model_field_to_node_input(
                    IO.STRING, KlingImage2VideoRequest, "prompt", multiline=True
                ),
                "negative_prompt": model_field_to_node_input(
                    IO.STRING,
                    KlingImage2VideoRequest,
                    "negative_prompt",
                    multiline=True,
                ),
                "cfg_scale": model_field_to_node_input(
                    IO.FLOAT,
                    KlingImage2VideoRequest,
                    "cfg_scale",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                ),
                "aspect_ratio": model_field_to_node_input(
                    IO.COMBO,
                    KlingImage2VideoRequest,
                    "aspect_ratio",
                    enum_type=KlingVideoGenAspectRatio,
                ),
                "mode": (
                    modes,
                    {
                        "default": modes[2],
                        "tooltip": "The configuration to use for the video generation following the format: mode / duration / model_name.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Generate a video sequence that transitions between your provided start and end images. The node creates all frames in between, producing a smooth transformation from the first frame to the last."

    async def api_call(
        self,
        start_frame: torch.Tensor,
        end_frame: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        aspect_ratio: str,
        mode: str,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        mode, duration, model_name = KlingStartEndFrameNode.get_mode_string_mapping()[
            mode
        ]
        return await super().api_call(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_name=model_name,
            start_frame=start_frame,
            cfg_scale=cfg_scale,
            mode=mode,
            aspect_ratio=aspect_ratio,
            duration=duration,
            end_frame=end_frame,
            unique_id=unique_id,
            **kwargs,
        )


class KlingVideoExtendNode(KlingNodeBase):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": model_field_to_node_input(
                    IO.STRING, KlingVideoExtendRequest, "prompt", multiline=True
                ),
                "negative_prompt": model_field_to_node_input(
                    IO.STRING,
                    KlingVideoExtendRequest,
                    "negative_prompt",
                    multiline=True,
                ),
                "cfg_scale": model_field_to_node_input(
                    IO.FLOAT,
                    KlingVideoExtendRequest,
                    "cfg_scale",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                ),
                "video_id": model_field_to_node_input(
                    IO.STRING, KlingVideoExtendRequest, "video_id", forceInput=True
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("VIDEO", "video_id", "duration")
    DESCRIPTION = "Kling Video Extend Node. Extend videos made by other Kling nodes. The video_id is created by using other Kling Nodes."

    async def get_response(
        self, task_id: str, auth_kwargs: dict[str, str], node_id: Optional[str] = None
    ) -> KlingVideoExtendResponse:
        return await poll_until_finished(
            auth_kwargs,
            ApiEndpoint(
                path=f"{PATH_VIDEO_EXTEND}/{task_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=KlingVideoExtendResponse,
            ),
            result_url_extractor=get_video_url_from_response,
            estimated_duration=AVERAGE_DURATION_VIDEO_EXTEND,
            node_id=node_id,
        )

    async def api_call(
        self,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        video_id: str,
        unique_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[VideoFromFile, str, str]:
        validate_prompts(prompt, negative_prompt, MAX_PROMPT_LENGTH_T2V)
        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_VIDEO_EXTEND,
                method=HttpMethod.POST,
                request_model=KlingVideoExtendRequest,
                response_model=KlingVideoExtendResponse,
            ),
            request=KlingVideoExtendRequest(
                prompt=prompt if prompt else None,
                negative_prompt=negative_prompt if negative_prompt else None,
                cfg_scale=cfg_scale,
                video_id=video_id,
            ),
            auth_kwargs=kwargs,
        )

        task_creation_response = await initial_operation.execute()
        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.data.task_id

        final_response = await self.get_response(
            task_id, auth_kwargs=kwargs, node_id=unique_id
        )
        validate_video_result_response(final_response)

        video = get_video_from_response(final_response)
        return await video_result_to_node_output(video)


class KlingVideoEffectsBase(KlingNodeBase):
    """Kling Video Effects Base"""

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("VIDEO", "video_id", "duration")

    async def get_response(
        self, task_id: str, auth_kwargs: dict[str, str], node_id: Optional[str] = None
    ) -> KlingVideoEffectsResponse:
        return await poll_until_finished(
            auth_kwargs,
            ApiEndpoint(
                path=f"{PATH_VIDEO_EFFECTS}/{task_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=KlingVideoEffectsResponse,
            ),
            result_url_extractor=get_video_url_from_response,
            estimated_duration=AVERAGE_DURATION_VIDEO_EFFECTS,
            node_id=node_id,
        )

    async def api_call(
        self,
        dual_character: bool,
        effect_scene: KlingDualCharacterEffectsScene | KlingSingleImageEffectsScene,
        model_name: str,
        duration: KlingVideoGenDuration,
        image_1: torch.Tensor,
        image_2: Optional[torch.Tensor] = None,
        mode: Optional[KlingVideoGenMode] = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        if dual_character:
            request_input_field = KlingDualCharacterEffectInput(
                model_name=model_name,
                mode=mode,
                images=[
                    tensor_to_base64_string(image_1),
                    tensor_to_base64_string(image_2),
                ],
                duration=duration,
            )
        else:
            request_input_field = KlingSingleImageEffectInput(
                model_name=model_name,
                image=tensor_to_base64_string(image_1),
                duration=duration,
            )

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_VIDEO_EFFECTS,
                method=HttpMethod.POST,
                request_model=KlingVideoEffectsRequest,
                response_model=KlingVideoEffectsResponse,
            ),
            request=KlingVideoEffectsRequest(
                effect_scene=effect_scene,
                input=request_input_field,
            ),
            auth_kwargs=kwargs,
        )

        task_creation_response = await initial_operation.execute()
        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.data.task_id

        final_response = await self.get_response(
            task_id, auth_kwargs=kwargs, node_id=unique_id
        )
        validate_video_result_response(final_response)

        video = get_video_from_response(final_response)
        return await video_result_to_node_output(video)


class KlingDualCharacterVideoEffectNode(KlingVideoEffectsBase):
    """Kling Dual Character Video Effect Node"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_left": (IO.IMAGE, {"tooltip": "Left side image"}),
                "image_right": (IO.IMAGE, {"tooltip": "Right side image"}),
                "effect_scene": model_field_to_node_input(
                    IO.COMBO,
                    KlingVideoEffectsRequest,
                    "effect_scene",
                    enum_type=KlingDualCharacterEffectsScene,
                ),
                "model_name": model_field_to_node_input(
                    IO.COMBO,
                    KlingDualCharacterEffectInput,
                    "model_name",
                    enum_type=KlingCharacterEffectModelName,
                ),
                "mode": model_field_to_node_input(
                    IO.COMBO,
                    KlingDualCharacterEffectInput,
                    "mode",
                    enum_type=KlingVideoGenMode,
                ),
                "duration": model_field_to_node_input(
                    IO.COMBO,
                    KlingDualCharacterEffectInput,
                    "duration",
                    enum_type=KlingVideoGenDuration,
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Achieve different special effects when generating a video based on the effect_scene. First image will be positioned on left side, second on right side of the composite."
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("VIDEO", "duration")

    async def api_call(
        self,
        image_left: torch.Tensor,
        image_right: torch.Tensor,
        effect_scene: KlingDualCharacterEffectsScene,
        model_name: KlingCharacterEffectModelName,
        mode: KlingVideoGenMode,
        duration: KlingVideoGenDuration,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        video, _, duration = await super().api_call(
            dual_character=True,
            effect_scene=effect_scene,
            model_name=model_name,
            mode=mode,
            duration=duration,
            image_1=image_left,
            image_2=image_right,
            unique_id=unique_id,
            **kwargs,
        )
        return video, duration


class KlingSingleImageVideoEffectNode(KlingVideoEffectsBase):
    """Kling Single Image Video Effect Node"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (
                    IO.IMAGE,
                    {
                        "tooltip": " Reference Image. URL or Base64 encoded string (without data:image prefix). File size cannot exceed 10MB, resolution not less than 300*300px, aspect ratio between 1:2.5 ~ 2.5:1"
                    },
                ),
                "effect_scene": model_field_to_node_input(
                    IO.COMBO,
                    KlingVideoEffectsRequest,
                    "effect_scene",
                    enum_type=KlingSingleImageEffectsScene,
                ),
                "model_name": model_field_to_node_input(
                    IO.COMBO,
                    KlingSingleImageEffectInput,
                    "model_name",
                    enum_type=KlingSingleImageEffectModelName,
                ),
                "duration": model_field_to_node_input(
                    IO.COMBO,
                    KlingSingleImageEffectInput,
                    "duration",
                    enum_type=KlingVideoGenDuration,
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Achieve different special effects when generating a video based on the effect_scene."

    async def api_call(
        self,
        image: torch.Tensor,
        effect_scene: KlingSingleImageEffectsScene,
        model_name: KlingSingleImageEffectModelName,
        duration: KlingVideoGenDuration,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        return await super().api_call(
            dual_character=False,
            effect_scene=effect_scene,
            model_name=model_name,
            duration=duration,
            image_1=image,
            unique_id=unique_id,
            **kwargs,
        )


class KlingLipSyncBase(KlingNodeBase):
    """Kling Lip Sync Base"""

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("VIDEO", "video_id", "duration")

    def validate_lip_sync_video(self, video: VideoInput):
        """
        Validates the input video adheres to the expectations of the Kling Lip Sync API:
        - Video length does not exceed 10s and is not shorter than 2s
        - Length and width dimensions should both be between 720px and 1920px

        See: https://app.klingai.com/global/dev/document-api/apiReference/model/videoTolip
        """
        validate_video_dimensions(video, 720, 1920)
        validate_video_duration(video, 2, 10)

    def validate_text(self, text: str):
        if not text:
            raise ValueError("Text is required")
        if len(text) > MAX_PROMPT_LENGTH_LIP_SYNC:
            raise ValueError(
                f"Text is too long. Maximum length is {MAX_PROMPT_LENGTH_LIP_SYNC} characters."
            )

    async def get_response(
        self, task_id: str, auth_kwargs: dict[str, str], node_id: Optional[str] = None
    ) -> KlingLipSyncResponse:
        """Polls the Kling API endpoint until the task reaches a terminal state."""
        return await poll_until_finished(
            auth_kwargs,
            ApiEndpoint(
                path=f"{PATH_LIP_SYNC}/{task_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=KlingLipSyncResponse,
            ),
            result_url_extractor=get_video_url_from_response,
            estimated_duration=AVERAGE_DURATION_LIP_SYNC,
            node_id=node_id,
        )

    async def api_call(
        self,
        video: VideoInput,
        audio: Optional[AudioInput] = None,
        voice_language: Optional[str] = None,
        mode: Optional[str] = None,
        text: Optional[str] = None,
        voice_speed: Optional[float] = None,
        voice_id: Optional[str] = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[VideoFromFile, str, str]:
        if text:
            self.validate_text(text)
        self.validate_lip_sync_video(video)

        # Upload video to Comfy API and get download URL
        video_url = await upload_video_to_comfyapi(video, auth_kwargs=kwargs)
        logging.info("Uploaded video to Comfy API. URL: %s", video_url)

        # Upload the audio file to Comfy API and get download URL
        if audio:
            audio_url = await upload_audio_to_comfyapi(audio, auth_kwargs=kwargs)
            logging.info("Uploaded audio to Comfy API. URL: %s", audio_url)
        else:
            audio_url = None

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_LIP_SYNC,
                method=HttpMethod.POST,
                request_model=KlingLipSyncRequest,
                response_model=KlingLipSyncResponse,
            ),
            request=KlingLipSyncRequest(
                input=KlingLipSyncInputObject(
                    video_url=video_url,
                    mode=mode,
                    text=text,
                    voice_language=voice_language,
                    voice_speed=voice_speed,
                    audio_type="url",
                    audio_url=audio_url,
                    voice_id=voice_id,
                ),
            ),
            auth_kwargs=kwargs,
        )

        task_creation_response = await initial_operation.execute()
        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.data.task_id

        final_response = await self.get_response(
            task_id, auth_kwargs=kwargs, node_id=unique_id
        )
        validate_video_result_response(final_response)

        video = get_video_from_response(final_response)
        return await video_result_to_node_output(video)


class KlingLipSyncAudioToVideoNode(KlingLipSyncBase):
    """Kling Lip Sync Audio to Video Node. Syncs mouth movements in a video file to the audio content of an audio file."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": (IO.VIDEO, {}),
                "audio": (IO.AUDIO, {}),
                "voice_language": model_field_to_node_input(
                    IO.COMBO,
                    KlingLipSyncInputObject,
                    "voice_language",
                    enum_type=KlingLipSyncVoiceLanguage,
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Kling Lip Sync Audio to Video Node. Syncs mouth movements in a video file to the audio content of an audio file. When using, ensure that the audio contains clearly distinguishable vocals and that the video contains a distinct face. The audio file should not be larger than 5MB. The video file should not be larger than 100MB, should have height/width between 720px and 1920px, and should be between 2s and 10s in length."

    async def api_call(
        self,
        video: VideoInput,
        audio: AudioInput,
        voice_language: str,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        return await super().api_call(
            video=video,
            audio=audio,
            voice_language=voice_language,
            mode="audio2video",
            unique_id=unique_id,
            **kwargs,
        )


class KlingLipSyncTextToVideoNode(KlingLipSyncBase):
    """Kling Lip Sync Text to Video Node. Syncs mouth movements in a video file to a text prompt."""

    @staticmethod
    def get_voice_config() -> dict[str, tuple[str, str]]:
        return {
            # English voices
            "Melody": ("girlfriend_4_speech02", "en"),
            "Sunny": ("genshin_vindi2", "en"),
            "Sage": ("zhinen_xuesheng", "en"),
            "Ace": ("AOT", "en"),
            "Blossom": ("ai_shatang", "en"),
            "Peppy": ("genshin_klee2", "en"),
            "Dove": ("genshin_kirara", "en"),
            "Shine": ("ai_kaiya", "en"),
            "Anchor": ("oversea_male1", "en"),
            "Lyric": ("ai_chenjiahao_712", "en"),
            "Tender": ("chat1_female_new-3", "en"),
            "Siren": ("chat_0407_5-1", "en"),
            "Zippy": ("cartoon-boy-07", "en"),
            "Bud": ("uk_boy1", "en"),
            "Sprite": ("cartoon-girl-01", "en"),
            "Candy": ("PeppaPig_platform", "en"),
            "Beacon": ("ai_huangzhong_712", "en"),
            "Rock": ("ai_huangyaoshi_712", "en"),
            "Titan": ("ai_laoguowang_712", "en"),
            "Grace": ("chengshu_jiejie", "en"),
            "Helen": ("you_pingjing", "en"),
            "Lore": ("calm_story1", "en"),
            "Crag": ("uk_man2", "en"),
            "Prattle": ("laopopo_speech02", "en"),
            "Hearth": ("heainainai_speech02", "en"),
            "The Reader": ("reader_en_m-v1", "en"),
            "Commercial Lady": ("commercial_lady_en_f-v1", "en"),
            # Chinese voices
            "阳光少年": ("genshin_vindi2", "zh"),
            "懂事小弟": ("zhinen_xuesheng", "zh"),
            "运动少年": ("tiyuxi_xuedi", "zh"),
            "青春少女": ("ai_shatang", "zh"),
            "温柔小妹": ("genshin_klee2", "zh"),
            "元气少女": ("genshin_kirara", "zh"),
            "阳光男生": ("ai_kaiya", "zh"),
            "幽默小哥": ("tiexin_nanyou", "zh"),
            "文艺小哥": ("ai_chenjiahao_712", "zh"),
            "甜美邻家": ("girlfriend_1_speech02", "zh"),
            "温柔姐姐": ("chat1_female_new-3", "zh"),
            "职场女青": ("girlfriend_2_speech02", "zh"),
            "活泼男童": ("cartoon-boy-07", "zh"),
            "俏皮女童": ("cartoon-girl-01", "zh"),
            "稳重老爸": ("ai_huangyaoshi_712", "zh"),
            "温柔妈妈": ("you_pingjing", "zh"),
            "严肃上司": ("ai_laoguowang_712", "zh"),
            "优雅贵妇": ("chengshu_jiejie", "zh"),
            "慈祥爷爷": ("zhuxi_speech02", "zh"),
            "唠叨爷爷": ("uk_oldman3", "zh"),
            "唠叨奶奶": ("laopopo_speech02", "zh"),
            "和蔼奶奶": ("heainainai_speech02", "zh"),
            "东北老铁": ("dongbeilaotie_speech02", "zh"),
            "重庆小伙": ("chongqingxiaohuo_speech02", "zh"),
            "四川妹子": ("chuanmeizi_speech02", "zh"),
            "潮汕大叔": ("chaoshandashu_speech02", "zh"),
            "台湾男生": ("ai_taiwan_man2_speech02", "zh"),
            "西安掌柜": ("xianzhanggui_speech02", "zh"),
            "天津姐姐": ("tianjinjiejie_speech02", "zh"),
            "新闻播报男": ("diyinnansang_DB_CN_M_04-v2", "zh"),
            "译制片男": ("yizhipiannan-v1", "zh"),
            "撒娇女友": ("tianmeixuemei-v1", "zh"),
            "刀片烟嗓": ("daopianyansang-v1", "zh"),
            "乖巧正太": ("mengwa-v1", "zh"),
        }

    @classmethod
    def INPUT_TYPES(s):
        voice_options = list(s.get_voice_config().keys())
        return {
            "required": {
                "video": (IO.VIDEO, {}),
                "text": model_field_to_node_input(
                    IO.STRING, KlingLipSyncInputObject, "text", multiline=True
                ),
                "voice": (voice_options, {"default": voice_options[0]}),
                "voice_speed": model_field_to_node_input(
                    IO.FLOAT, KlingLipSyncInputObject, "voice_speed", slider=True
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Kling Lip Sync Text to Video Node. Syncs mouth movements in a video file to a text prompt. The video file should not be larger than 100MB, should have height/width between 720px and 1920px, and should be between 2s and 10s in length."

    async def api_call(
        self,
        video: VideoInput,
        text: str,
        voice: str,
        voice_speed: float,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        voice_id, voice_language = KlingLipSyncTextToVideoNode.get_voice_config()[voice]
        return await super().api_call(
            video=video,
            text=text,
            voice_language=voice_language,
            voice_id=voice_id,
            voice_speed=voice_speed,
            mode="text2video",
            unique_id=unique_id,
            **kwargs,
        )


class KlingImageGenerationBase(KlingNodeBase):
    """Kling Image Generation Base Node."""

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "api node/image/Kling"

    def validate_prompt(self, prompt: str, negative_prompt: Optional[str] = None):
        if not prompt or len(prompt) > MAX_PROMPT_LENGTH_IMAGE_GEN:
            raise ValueError(
                f"Prompt must be less than {MAX_PROMPT_LENGTH_IMAGE_GEN} characters"
            )
        if negative_prompt and len(negative_prompt) > MAX_PROMPT_LENGTH_IMAGE_GEN:
            raise ValueError(
                f"Negative prompt must be less than {MAX_PROMPT_LENGTH_IMAGE_GEN} characters"
            )


class KlingVirtualTryOnNode(KlingImageGenerationBase):
    """Kling Virtual Try On Node."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "human_image": (IO.IMAGE, {}),
                "cloth_image": (IO.IMAGE, {}),
                "model_name": model_field_to_node_input(
                    IO.COMBO,
                    KlingVirtualTryOnRequest,
                    "model_name",
                    enum_type=KlingVirtualTryOnModelName,
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Kling Virtual Try On Node. Input a human image and a cloth image to try on the cloth on the human. You can merge multiple clothing item pictures into one image with a white background."

    async def get_response(
        self, task_id: str, auth_kwargs: dict[str, str], node_id: Optional[str] = None
    ) -> KlingVirtualTryOnResponse:
        return await poll_until_finished(
            auth_kwargs,
            ApiEndpoint(
                path=f"{PATH_VIRTUAL_TRY_ON}/{task_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=KlingVirtualTryOnResponse,
            ),
            result_url_extractor=get_images_urls_from_response,
            estimated_duration=AVERAGE_DURATION_VIRTUAL_TRY_ON,
            node_id=node_id,
        )

    async def api_call(
        self,
        human_image: torch.Tensor,
        cloth_image: torch.Tensor,
        model_name: KlingVirtualTryOnModelName,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_VIRTUAL_TRY_ON,
                method=HttpMethod.POST,
                request_model=KlingVirtualTryOnRequest,
                response_model=KlingVirtualTryOnResponse,
            ),
            request=KlingVirtualTryOnRequest(
                human_image=tensor_to_base64_string(human_image),
                cloth_image=tensor_to_base64_string(cloth_image),
                model_name=model_name,
            ),
            auth_kwargs=kwargs,
        )

        task_creation_response = await initial_operation.execute()
        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.data.task_id

        final_response = await self.get_response(
            task_id, auth_kwargs=kwargs, node_id=unique_id
        )
        validate_image_result_response(final_response)

        images = get_images_from_response(final_response)
        return (await image_result_to_node_output(images),)


class KlingImageGenerationNode(KlingImageGenerationBase):
    """Kling Image Generation Node. Generate an image from a text prompt with an optional reference image."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": model_field_to_node_input(
                    IO.STRING,
                    KlingImageGenerationsRequest,
                    "prompt",
                    multiline=True,
                    max_length=MAX_PROMPT_LENGTH_IMAGE_GEN,
                ),
                "negative_prompt": model_field_to_node_input(
                    IO.STRING,
                    KlingImageGenerationsRequest,
                    "negative_prompt",
                    multiline=True,
                ),
                "image_type": model_field_to_node_input(
                    IO.COMBO,
                    KlingImageGenerationsRequest,
                    "image_reference",
                    enum_type=KlingImageGenImageReferenceType,
                ),
                "image_fidelity": model_field_to_node_input(
                    IO.FLOAT,
                    KlingImageGenerationsRequest,
                    "image_fidelity",
                    slider=True,
                    step=0.01,
                ),
                "human_fidelity": model_field_to_node_input(
                    IO.FLOAT,
                    KlingImageGenerationsRequest,
                    "human_fidelity",
                    slider=True,
                    step=0.01,
                ),
                "model_name": model_field_to_node_input(
                    IO.COMBO,
                    KlingImageGenerationsRequest,
                    "model_name",
                    enum_type=KlingImageGenModelName,
                ),
                "aspect_ratio": model_field_to_node_input(
                    IO.COMBO,
                    KlingImageGenerationsRequest,
                    "aspect_ratio",
                    enum_type=KlingImageGenAspectRatio,
                ),
                "n": model_field_to_node_input(
                    IO.INT,
                    KlingImageGenerationsRequest,
                    "n",
                ),
            },
            "optional": {
                "image": (IO.IMAGE, {}),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Kling Image Generation Node. Generate an image from a text prompt with an optional reference image."

    async def get_response(
        self,
        task_id: str,
        auth_kwargs: Optional[dict[str, str]],
        node_id: Optional[str] = None,
    ) -> KlingImageGenerationsResponse:
        return await poll_until_finished(
            auth_kwargs,
            ApiEndpoint(
                path=f"{PATH_IMAGE_GENERATIONS}/{task_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=KlingImageGenerationsResponse,
            ),
            result_url_extractor=get_images_urls_from_response,
            estimated_duration=AVERAGE_DURATION_IMAGE_GEN,
            node_id=node_id,
        )

    async def api_call(
        self,
        model_name: KlingImageGenModelName,
        prompt: str,
        negative_prompt: str,
        image_type: KlingImageGenImageReferenceType,
        image_fidelity: float,
        human_fidelity: float,
        n: int,
        aspect_ratio: KlingImageGenAspectRatio,
        image: Optional[torch.Tensor] = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        self.validate_prompt(prompt, negative_prompt)

        if image is None:
            image_type = None
        elif model_name == KlingImageGenModelName.kling_v1:
            raise ValueError(f"The model {KlingImageGenModelName.kling_v1.value} does not support reference images.")
        else:
            image = tensor_to_base64_string(image)

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_IMAGE_GENERATIONS,
                method=HttpMethod.POST,
                request_model=KlingImageGenerationsRequest,
                response_model=KlingImageGenerationsResponse,
            ),
            request=KlingImageGenerationsRequest(
                model_name=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                image_reference=image_type,
                image_fidelity=image_fidelity,
                human_fidelity=human_fidelity,
                n=n,
                aspect_ratio=aspect_ratio,
            ),
            auth_kwargs=kwargs,
        )

        task_creation_response = await initial_operation.execute()
        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.data.task_id

        final_response = await self.get_response(
            task_id, auth_kwargs=kwargs, node_id=unique_id
        )
        validate_image_result_response(final_response)

        images = get_images_from_response(final_response)
        return (await image_result_to_node_output(images),)


NODE_CLASS_MAPPINGS = {
    "KlingCameraControls": KlingCameraControls,
    "KlingTextToVideoNode": KlingTextToVideoNode,
    "KlingImage2VideoNode": KlingImage2VideoNode,
    "KlingCameraControlI2VNode": KlingCameraControlI2VNode,
    "KlingCameraControlT2VNode": KlingCameraControlT2VNode,
    "KlingStartEndFrameNode": KlingStartEndFrameNode,
    "KlingVideoExtendNode": KlingVideoExtendNode,
    "KlingLipSyncAudioToVideoNode": KlingLipSyncAudioToVideoNode,
    "KlingLipSyncTextToVideoNode": KlingLipSyncTextToVideoNode,
    "KlingVirtualTryOnNode": KlingVirtualTryOnNode,
    "KlingImageGenerationNode": KlingImageGenerationNode,
    "KlingSingleImageVideoEffectNode": KlingSingleImageVideoEffectNode,
    "KlingDualCharacterVideoEffectNode": KlingDualCharacterVideoEffectNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingCameraControls": "Kling Camera Controls",
    "KlingTextToVideoNode": "Kling Text to Video",
    "KlingImage2VideoNode": "Kling Image to Video",
    "KlingCameraControlI2VNode": "Kling Image to Video (Camera Control)",
    "KlingCameraControlT2VNode": "Kling Text to Video (Camera Control)",
    "KlingStartEndFrameNode": "Kling Start-End Frame to Video",
    "KlingVideoExtendNode": "Kling Video Extend",
    "KlingLipSyncAudioToVideoNode": "Kling Lip Sync Video with Audio",
    "KlingLipSyncTextToVideoNode": "Kling Lip Sync Video with Text",
    "KlingVirtualTryOnNode": "Kling Virtual Try On",
    "KlingImageGenerationNode": "Kling Image Generation",
    "KlingSingleImageVideoEffectNode": "Kling Video Effects",
    "KlingDualCharacterVideoEffectNode": "Kling Dual Character Video Effects",
}

from inspect import cleandoc
from typing import Union, Optional
import math
import logging
import torch

from comfy_api_nodes.apis import (
    KlingText2VideoRequest,
    KlingText2VideoResponse,
    TaskStatus,
    CameraControl,
    Config as CameraConfig,
    Type as CameraType,
    Duration,
    Mode,
    AspectRatio,
    ModelName,
    KlingImage2VideoRequest,
    KlingImage2VideoResponse,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.nodes_api import (
    tensor_to_base64_string,
    download_url_to_bytesio,
)
from comfy.comfy_types.node_typing import IO, InputTypeOptions, ComfyNodeABC
from comfy_api.input_impl import VideoFromFile
from comfy_api_nodes.mapper_utils import model_field_to_node_input

KLING_API_VERSION = "v1"
PATH_TEXT_TO_VIDEO = f"/proxy/kling/{KLING_API_VERSION}/videos/text2video"
PATH_IMAGE_TO_VIDEO = f"/proxy/kling/{KLING_API_VERSION}/videos/image2video"
PATH_VIDEO_EXTEND = f"/proxy/kling/{KLING_API_VERSION}/videos/video-extend"
PATH_LIP_SYNC = f"/proxy/kling/{KLING_API_VERSION}/videos/lip-sync"
PATH_VIDEO_EFFECTS = f"/proxy/kling/{KLING_API_VERSION}/videos/effects"
PATH_CHARACTER_IMAGE = f"/proxy/kling/{KLING_API_VERSION}/images/generations"
PATH_VIRTUAL_TRY_ON = f"/proxy/kling/{KLING_API_VERSION}/images/kolors-virtual-try-on"


class KlingApiError(Exception):
    """Base exception for Kling API errors."""

    pass


def is_valid_camera_control_configs(configs: list[float]) -> bool:
    """Verifies that at least one camera control configuration is non-zero."""
    return any(not math.isclose(value, 0.0) for value in configs)


def is_valid_prompt(prompt: str) -> bool:
    """Verifies that the prompt is not empty."""
    return bool(prompt)


def is_valid_initial_response(response: KlingText2VideoResponse) -> bool:
    """Verifies that the initial response contains a task ID."""
    return bool(response.data.task_id)


def is_valid_video_response(response: KlingText2VideoResponse) -> bool:
    """Verifies that the response contains a task result with at least one video."""
    return (
        response.data.task_result
        and response.data.task_result.videos
        and len(response.data.task_result.videos) > 0
    )


def is_camera_control_supported(model_name: str, duration: str, mode: str) -> bool:
    """`camera_control` is only supported in `pro` mode with `5s` duration and `kling-v1-5`"""
    return model_name == "kling-v1-5" and duration == "5" and mode == "pro"


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


def _get_camera_control_inputs() -> dict[str, tuple[IO, InputTypeOptions]]:
    """Returns a dictionary of camera control inputs common to Kling video generation nodes."""
    return {
        "camera_control_type": (
            IO.COMBO,
            {
                "options": [
                    camera_control_type.value for camera_control_type in CameraType
                ],
                "default": "simple",
                "tooltip": "Predefined camera movements type. simple: Customizable camera movement. down_back: Camera descends and moves backward. forward_up: Camera moves forward and tilts up. right_turn_forward: Rotate right and move forward. left_turn_forward: Rotate left and move forward.",
            },
        ),
        "camera_control_horizontal": get_camera_control_input_config(
            "Controls camera's movement along horizontal axis (x-axis). Negative indicates left, positive indicates right"
        ),
        "camera_control_vertical": get_camera_control_input_config(
            "Controls camera's movement along vertical axis (y-axis). Negative indicates downward, positive indicates upward."
        ),
        "camera_control_pan": get_camera_control_input_config(
            "Controls camera's rotation in vertical plane (x-axis). Negative indicates downward rotation, positive indicates upward rotation.",
            default=0.5,
        ),
        "camera_control_roll": get_camera_control_input_config(
            "Controls camera's rotation in horizontal plane (y-axis). Negative indicates left rotation, positive indicates right rotation.",
        ),
        "camera_control_tilt": get_camera_control_input_config(
            "Controls camera's rolling amount (z-axis). Negative indicates counterclockwise, positive indicates clockwise.",
        ),
        "camera_control_zoom": get_camera_control_input_config(
            "Controls change in camera's focal length. Negative indicates narrower field of view, positive indicates wider field of view.",
        ),
    }


def download_url_to_video_output(video_url: str) -> tuple[VideoFromFile]:
    """Downloads a video from a URL and returns a VIDEO output."""
    video_io = download_url_to_bytesio(video_url)
    if video_io is None:
        error_msg = f"Failed to download video from {video_url}"
        logging.error(error_msg)
        raise KlingApiError(error_msg)
    return (VideoFromFile(video_io),)


class KlingNodeABC(ComfyNodeABC):
    """Base class for Kling nodes."""

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        prompt,
        negative_prompt,
        camera_control_horizontal,
        camera_control_vertical,
        camera_control_pan,
        camera_control_roll,
        camera_control_tilt,
        camera_control_zoom,
    ) -> Union[str, bool]:
        if not is_valid_prompt(prompt):
            return "Prompt is required"
        if len(prompt) >= 2500:
            return "Prompt must be less than 2500 characters"
        if negative_prompt and len(negative_prompt) >= 2500:
            return "Negative prompt must be less than 2500 characters"
        if not is_valid_camera_control_configs(
            [
                camera_control_horizontal,
                camera_control_vertical,
                camera_control_pan,
                camera_control_roll,
                camera_control_tilt,
                camera_control_zoom,
            ]
        ):
            return "Invalid camera control configs"
        return True

    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "api_call"
    CATEGORY = "api node/video/Kling"
    API_NODE = True


class KlingTextToVideoNode(KlingNodeABC):
    """
    Kling Text to Video Node.
    """

    @staticmethod
    def poll_for_task_status(task_id: str, auth_token: str) -> KlingText2VideoResponse:
        """Polls the Kling API endpoint until the task reaches a terminal state."""
        polling_operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"{PATH_TEXT_TO_VIDEO}/{task_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=KlingText2VideoResponse,
            ),
            completed_statuses=[
                TaskStatus.succeed.value,
            ],
            failed_statuses=[TaskStatus.failed.value],
            status_extractor=lambda response: (
                response.data.task_status.value
                if response.data and response.data.task_status
                else None
            ),
            auth_token=auth_token,
        )
        return polling_operation.execute()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": model_field_to_node_input(
                    IO.STRING, KlingText2VideoRequest, "prompt", multiline=True
                ),
                "negative_prompt": model_field_to_node_input(
                    IO.STRING, KlingText2VideoRequest, "negative_prompt", multiline=True
                ),
                "cfg_scale": model_field_to_node_input(
                    IO.FLOAT, KlingText2VideoRequest, "cfg_scale"
                ),
                "mode": model_field_to_node_input(
                    IO.COMBO, KlingText2VideoRequest, "mode", enum_type=Mode
                ),
                "duration": model_field_to_node_input(
                    IO.COMBO, KlingText2VideoRequest, "duration", enum_type=Duration
                ),
                "aspect_ratio": model_field_to_node_input(
                    IO.COMBO,
                    KlingText2VideoRequest,
                    "aspect_ratio",
                    enum_type=AspectRatio,
                ),
                **_get_camera_control_inputs(),
            },
            "hidden": {"auth_token": "AUTH_TOKEN_COMFY_ORG"},
        }

    RETURN_TYPES = ("VIDEO",)

    def api_call(
        self,
        prompt: str,
        negative_prompt: str,
        duration: int,
        mode: str,
        cfg_scale: float,
        aspect_ratio: str,
        camera_control_type: str,
        camera_control_horizontal: float,
        camera_control_vertical: float,
        camera_control_pan: float,
        camera_control_roll: float,
        camera_control_tilt: float,
        camera_control_zoom: float,
        auth_token: Optional[str] = None,
    ) -> tuple[VideoFromFile]:
        camera_control = None
        if is_camera_control_supported("kling-v1-6", duration, mode):
            camera_control = CameraControl(
                type=CameraType(camera_control_type),
                config=CameraConfig(
                    horizontal=camera_control_horizontal,
                    vertical=camera_control_vertical,
                    pan=camera_control_pan,
                    roll=camera_control_roll,
                    tilt=camera_control_tilt,
                    zoom=camera_control_zoom,
                ).model_dump(exclude_none=True),
            )

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
                duration=Duration(duration),
                mode=Mode(mode),
                cfg_scale=cfg_scale,
                aspect_ratio=AspectRatio(aspect_ratio),
                camera_control=camera_control,
            ),
            auth_token=auth_token,
        )

        initial_response = initial_operation.execute()
        if not is_valid_initial_response(initial_response):
            error_msg = f"Kling initial request failed. Code: {initial_response.code}, Message: {initial_response.message}, Data: {initial_response.data}"
            logging.error(error_msg)
            raise KlingApiError(error_msg)

        task_id = initial_response.data.task_id
        logging.debug("Kling task submitted. Task ID: %s", task_id)

        final_response = self.poll_for_task_status(task_id, auth_token)
        if not is_valid_video_response(final_response):
            error_msg = (
                f"Kling task {task_id} succeeded but no video data found in response."
            )
            logging.error(error_msg)
            raise KlingApiError(error_msg)

        video_url = str(final_response.data.task_result.videos[0].url)
        logging.debug("Kling task %s succeeded. Video URL: %s", task_id, video_url)

        return download_url_to_video_output(video_url)


class KlingImage2VideoNode(KlingNodeABC):
    """
    Kling Image to Video Node.
    """

    @staticmethod
    def poll_for_task_status(task_id: str, auth_token: str) -> KlingImage2VideoResponse:
        """Polls the Kling API endpoint until the task reaches a terminal state."""
        polling_operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"{PATH_IMAGE_TO_VIDEO}/{task_id}",
                method=HttpMethod.GET,
                request_model=KlingImage2VideoRequest,
                response_model=KlingImage2VideoResponse,
            ),
            completed_statuses=[TaskStatus.succeed.value],
            failed_statuses=[TaskStatus.failed.value],
            status_extractor=lambda response: (
                response.data.task_status.value
                if response.data and response.data.task_status
                else None
            ),
            auth_token=auth_token,
        )
        return polling_operation.execute()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": model_field_to_node_input(
                    IO.COMBO, KlingImage2VideoRequest, "model_name", enum_type=ModelName
                ),
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
                    IO.FLOAT, KlingImage2VideoRequest, "cfg_scale"
                ),
                "mode": model_field_to_node_input(
                    IO.COMBO, KlingImage2VideoRequest, "mode", enum_type=Mode
                ),
                "aspect_ratio": model_field_to_node_input(
                    IO.COMBO,
                    KlingImage2VideoRequest,
                    "aspect_ratio",
                    enum_type=AspectRatio,
                ),
                "duration": model_field_to_node_input(
                    IO.COMBO, KlingImage2VideoRequest, "duration", enum_type=Duration
                ),
                **_get_camera_control_inputs(),
            },
            "optional": {
                "end_frame": model_field_to_node_input(
                    IO.IMAGE, KlingImage2VideoRequest, "image_tail"
                ),
            },
            "hidden": {"auth_token": "AUTH_TOKEN_COMFY_ORG"},
        }

    RETURN_TYPES = ("VIDEO",)

    def api_call(
        self,
        model_name: str,
        start_frame: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        mode: str,
        aspect_ratio: str,
        duration: str,
        camera_control_type: str,
        camera_control_horizontal: float,
        camera_control_vertical: float,
        camera_control_pan: float,
        camera_control_roll: float,
        camera_control_tilt: float,
        camera_control_zoom: float,
        end_frame: Optional[torch.Tensor] = None,
        auth_token: Optional[str] = None,
    ) -> tuple[VideoFromFile]:
        camera_control = None
        if is_camera_control_supported(model_name, duration, mode):
            config = None
            if camera_control_type != "right_turn_forward":
                config = CameraConfig(
                    horizontal=camera_control_horizontal,
                    vertical=camera_control_vertical,
                    pan=camera_control_pan,
                    roll=camera_control_roll,
                    tilt=camera_control_tilt,
                    zoom=camera_control_zoom,
                )
            camera_control = CameraControl(
                type=CameraType(camera_control_type),
                config=config if config else None,
            )

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_IMAGE_TO_VIDEO,
                method=HttpMethod.POST,
                request_model=KlingImage2VideoRequest,
                response_model=KlingImage2VideoResponse,
            ),
            request=KlingImage2VideoRequest(
                model_name=ModelName(model_name),
                image=tensor_to_base64_string(start_frame),
                image_tail=(
                    tensor_to_base64_string(end_frame)
                    if end_frame is not None
                    else None
                ),
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                cfg_scale=cfg_scale,
                mode=Mode(mode),
                aspect_ratio=AspectRatio(aspect_ratio),
                duration=Duration(duration),
                camera_control=camera_control,
            ),
            auth_token=auth_token,
        )
        initial_response = initial_operation.execute()
        if not is_valid_initial_response(initial_response):
            error_msg = f"Kling initial request failed. Code: {initial_response.code}, Message: {initial_response.message}, Data: {initial_response.data}"
            logging.error(error_msg)
            raise KlingApiError(error_msg)

        task_id = initial_response.data.task_id
        logging.debug("Kling task submitted. Task ID: %s", task_id)

        final_response = KlingImage2VideoNode.poll_for_task_status(task_id, auth_token)
        if not is_valid_video_response(final_response):
            error_msg = (
                f"Kling task {task_id} succeeded but no video data found in response."
            )
            logging.error(error_msg)
            raise KlingApiError(error_msg)

        video_url = str(final_response.data.task_result.videos[0].url)
        logging.info("Attempting to download video from URL: %s", video_url)

        return download_url_to_video_output(video_url)


NODE_CLASS_MAPPINGS = {
    "KlingTextToVideoNode": KlingTextToVideoNode,
    "KlingImage2VideoNode": KlingImage2VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingTextToVideoNode": "Kling Text to Video",
    "KlingImage2VideoNode": "Kling Image to Video",
}

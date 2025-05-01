from typing import Optional
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
from comfy_api_nodes.apinode_utils import (
    tensor_to_base64_string,
    download_url_to_video_output,
)
from comfy_api_nodes.mapper_utils import model_field_to_node_input
from comfy.comfy_types.node_typing import IO, InputTypeOptions, ComfyNodeABC
from comfy_api.input_impl import VideoFromFile

KLING_API_VERSION = "v1"
PATH_TEXT_TO_VIDEO = f"/proxy/kling/{KLING_API_VERSION}/videos/text2video"
PATH_IMAGE_TO_VIDEO = f"/proxy/kling/{KLING_API_VERSION}/videos/image2video"
PATH_VIDEO_EXTEND = f"/proxy/kling/{KLING_API_VERSION}/videos/video-extend"
PATH_LIP_SYNC = f"/proxy/kling/{KLING_API_VERSION}/videos/lip-sync"
PATH_VIDEO_EFFECTS = f"/proxy/kling/{KLING_API_VERSION}/videos/effects"
PATH_CHARACTER_IMAGE = f"/proxy/kling/{KLING_API_VERSION}/images/generations"
PATH_VIRTUAL_TRY_ON = f"/proxy/kling/{KLING_API_VERSION}/images/kolors-virtual-try-on"

MAX_PROMPT_LENGTH_T2V = 2500
MAX_PROMPT_LENGTH_I2V = 500


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


class KlingCameraControls(ComfyNodeABC):
    """Kling Camera Controls Node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_control_type": (
                    IO.COMBO,
                    {
                        "options": [
                            camera_control_type.value
                            for camera_control_type in CameraType
                        ],
                        "default": "simple",
                        "tooltip": "Predefined camera movements type. simple: Customizable camera movement. down_back: Camera descends and moves backward. forward_up: Camera moves forward and tilts up. right_turn_forward: Rotate right and move forward. left_turn_forward: Rotate left and move forward.",
                    },
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

    DESCRIPTION = "Kling Camera Controls Node. Not all model and mode combinations support camera control. Please refer to the Kling API documentation for more information."
    RETURN_TYPES = ("CAMERA_CONTROL",)
    RETURN_NAMES = ("camera_control",)
    FUNCTION = "main"

    def main(
        self,
        camera_control_type: str,
        horizontal_movement: float,
        vertical_movement: float,
        pan: float,
        tilt: float,
        roll: float,
        zoom: float,
    ):
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

        return (
            CameraControl(
                type=CameraType(camera_control_type),
                config=CameraConfig(
                    horizontal=horizontal_movement,
                    vertical=vertical_movement,
                    pan=pan,
                    roll=roll,
                    tilt=tilt,
                    zoom=zoom,
                ),
            ),
        )


class KlingNodeBase(ComfyNodeABC):
    """Base class for Kling nodes."""

    FUNCTION = "api_call"
    CATEGORY = "api node/video/Kling"
    API_NODE = True


class KlingTextToVideoNode(KlingNodeBase):
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
                "model_name": model_field_to_node_input(
                    IO.COMBO,
                    KlingText2VideoRequest,
                    "model_name",
                    enum_type=ModelName,
                    default="kling-v2-master",
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
            },
            "optional": {
                "camera_control": ("CAMERA_CONTROL", {}),
            },
            "hidden": {"auth_token": "AUTH_TOKEN_COMFY_ORG"},
        }

    RETURN_TYPES = ("VIDEO",)
    DESCRIPTION = "Kling Text to Video Node"

    def api_call(
        self,
        prompt: str,
        negative_prompt: str,
        model_name: str,
        cfg_scale: float,
        mode: str,
        duration: int,
        aspect_ratio: str,
        camera_control: Optional[CameraControl] = None,
        auth_token: Optional[str] = None,
    ) -> tuple[VideoFromFile]:
        validate_prompts(prompt, negative_prompt, MAX_PROMPT_LENGTH_T2V)
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
                model_name=ModelName(model_name),
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


class KlingImage2VideoNode(KlingNodeBase):
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
                    enum_type=ModelName,
                    default="kling-v2-master",
                ),
                "start_frame": model_field_to_node_input(
                    IO.IMAGE, KlingImage2VideoRequest, "image"
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
            },
            "optional": {
                "camera_control": ("CAMERA_CONTROL", {}),
                "end_frame": model_field_to_node_input(
                    IO.IMAGE, KlingImage2VideoRequest, "image_tail"
                ),
            },
            "hidden": {"auth_token": "AUTH_TOKEN_COMFY_ORG"},
        }

    RETURN_TYPES = ("VIDEO",)
    DESCRIPTION = "Kling Image to Video Node"

    def api_call(
        self,
        prompt: str,
        negative_prompt: str,
        model_name: str,
        start_frame: torch.Tensor,
        cfg_scale: float,
        mode: str,
        aspect_ratio: str,
        duration: str,
        camera_control: Optional[CameraControl] = None,
        end_frame: Optional[torch.Tensor] = None,
        auth_token: Optional[str] = None,
    ) -> tuple[VideoFromFile]:
        validate_prompts(prompt, negative_prompt, MAX_PROMPT_LENGTH_I2V)
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
    "KlingCameraControls": KlingCameraControls,
    "KlingTextToVideoNode": KlingTextToVideoNode,
    "KlingImage2VideoNode": KlingImage2VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingCameraControls": "Kling Camera Controls",
    "KlingTextToVideoNode": "Kling Text to Video",
    "KlingImage2VideoNode": "Kling Image to Video",
}

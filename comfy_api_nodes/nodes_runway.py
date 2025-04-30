from inspect import cleandoc
from typing import Union, Optional
import logging

import torch
from comfy_api_nodes.apis import (
    RunwayImageToVideoRequest,
    RunwayImageToVideoResponse,
    RunwayTaskStatusResponse as TaskStatusResponse,
    RunwayTaskStatusEnum as TaskStatus,
    RunwayModelEnum as Model,
    RunwayDurationEnum as Duration,
    RunwayAspectRatioEnum as AspectRatio,
    RunwayPromptImageObject,
    RunwayPromptImageDetailedObject,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.nodes_api import (
    download_url_to_bytesio,
    upload_images_to_comfyapi,
)
from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from comfy_api.input_impl import VideoFromFile
from comfy_api_nodes.mapper_utils import model_field_to_node_input

PATH_IMAGE_TO_VIDEO = "/proxy/runway/image-to-video"
PATH_GET_TASK_STATUS = "/proxy/runway/tasks"


class RunwayApiError(Exception):
    """Base exception for Runway API errors."""

    pass


def extract_progress_from_task_status(response: TaskStatusResponse) -> float:
    if hasattr(response, "progress") and response.progress is not None:
        return response.progress * 100
    return None


class RunwayImageToVideoNode(ComfyNodeABC):
    """
    Runway Image to Video Node.
    """

    @staticmethod
    def is_ratio_supported(model: str, ratio: str) -> bool:
        """
        Checks if the chosen aspect ratio is supported by the chosen model.
        """
        if model != "gen3a_turbo" and ratio in [
            "1280:768",
            "768:1280",
        ]:
            return False
        return True

    @staticmethod
    def is_end_frame_supported(model: str) -> bool:
        """
        Checks if the chosen model supports the end frame input.
        """
        return model == "gen3a_turbo"

    @staticmethod
    def is_valid_prompt(prompt: str) -> bool:
        return bool(prompt)

    @staticmethod
    def is_valid_initial_response(response: RunwayImageToVideoResponse) -> bool:
        return bool(response.id)

    @staticmethod
    def is_valid_image(image: torch.Tensor) -> bool:
        """https://docs.dev.runwayml.com/assets/inputs/#common-error-reasons"""
        return image.shape[2] < 8000 and image.shape[1] < 8000

    @staticmethod
    def is_valid_video_response(response: RunwayImageToVideoResponse) -> bool:
        return response.output and len(response.output) > 0

    @staticmethod
    def poll_for_task_status(task_id: str, auth_token: str) -> TaskStatusResponse:
        """
        Polls the Runway API endpoint until the task reaches a terminal state.
        """
        polling_operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"{PATH_GET_TASK_STATUS}/{task_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=TaskStatusResponse,
            ),
            completed_statuses=[
                TaskStatus.SUCCEEDED.value,
            ],
            failed_statuses=[
                TaskStatus.FAILED.value,
                TaskStatus.CANCELLED.value,
            ],
            progress_extractor=extract_progress_from_task_status,
            status_extractor=lambda response: (response.status.value),
            auth_token=auth_token,
        )
        return polling_operation.execute()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": model_field_to_node_input(
                    IO.COMBO, RunwayImageToVideoRequest, "model", enum_type=Model
                ),
                "prompt": model_field_to_node_input(
                    IO.STRING, RunwayImageToVideoRequest, "promptText", multiline=True
                ),
                "duration": model_field_to_node_input(
                    IO.COMBO, RunwayImageToVideoRequest, "duration", enum_type=Duration
                ),
                "ratio": model_field_to_node_input(
                    IO.COMBO, RunwayImageToVideoRequest, "ratio", enum_type=AspectRatio
                ),
                "seed": model_field_to_node_input(
                    IO.INT, RunwayImageToVideoRequest, "seed"
                ),
            },
            "optional": {
                "start_frame": (
                    IO.IMAGE,
                    {"tooltip": "Start frame to be used for the video"},
                ),
                "end_frame": (
                    IO.IMAGE,
                    {
                        "tooltip": "End frame to be used for the video. Supported for gen3a_turbo only."
                    },
                ),
            },
            "hidden": {"auth_token": "AUTH_TOKEN_COMFY_ORG"},
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "api_call"
    CATEGORY = "api node/video/Runway"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model: str,
        ratio: str,
    ) -> Union[str, bool]:
        if not RunwayImageToVideoNode.is_ratio_supported(model, ratio):
            return "Invalid aspect ratio for the chosen model. 1280:768 and 768:1280 are only supported for gen3a_turbo."
        return True

    def api_call(
        self,
        model: str,
        prompt: str,
        duration: str,
        ratio: str,
        seed: int,
        start_frame: Optional[torch.Tensor] = None,
        end_frame: Optional[torch.Tensor] = None,
        auth_token: Optional[str] = None,
    ) -> tuple[VideoFromFile]:
        # Validate manually because optional inputs are not passed to VALIDATE_INPUTS.
        if start_frame is None and end_frame is None:
            message = "Start frame and end frame cannot both be empty."
            raise RunwayApiError(message)
        if end_frame is not None and not RunwayImageToVideoNode.is_end_frame_supported(
            model
        ):
            message = "End frame is only supported for gen3a_turbo model."
            raise RunwayApiError(message)

        prompt_images_tensors: list[torch.Tensor] = []
        if start_frame is not None:
            if not RunwayImageToVideoNode.is_valid_image(start_frame):
                message = "Start frame is not a valid image."
                raise RunwayApiError(message)
            prompt_images_tensors.append(start_frame)

        if end_frame != None:
            if not RunwayImageToVideoNode.is_valid_image(end_frame):
                message = "End frame is not a valid image."
                raise RunwayApiError(message)
            prompt_images_tensors.append(end_frame)

        # stack tensors
        prompt_images_tensor = torch.cat(prompt_images_tensors, dim=0)

        download_urls = upload_images_to_comfyapi(
            prompt_images_tensor,
            max_images=2,
            auth_token=auth_token,
            mime_type="image/png",
        )

        # Create a list of detailed image objects
        prompt_image_details: list[RunwayPromptImageDetailedObject] = [
            RunwayPromptImageDetailedObject(uri=str(download_urls[0]), position="first")
        ]
        if len(download_urls) > 1:
            prompt_image_details.append(
                RunwayPromptImageDetailedObject(
                    uri=str(download_urls[1]), position="last"
                )
            )

        # Wrap the list in the main object if details exist
        prompt_image_object: Optional[RunwayPromptImageObject] = None
        if prompt_image_details:
            prompt_image_object = RunwayPromptImageObject(root=prompt_image_details)

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_IMAGE_TO_VIDEO,
                method=HttpMethod.POST,
                request_model=RunwayImageToVideoRequest,
                response_model=RunwayImageToVideoResponse,
            ),
            request=RunwayImageToVideoRequest(
                promptText=prompt,
                seed=seed,
                model=Model(model),
                duration=Duration(duration),
                ratio=AspectRatio(ratio),
                promptImage=prompt_image_object,
            ),
            auth_token=auth_token,
        )

        initial_response = initial_operation.execute()
        if not RunwayImageToVideoNode.is_valid_initial_response(initial_response):
            error_message = "Invalid initial response from Runway API."
            logging.error(error_message)
            raise RunwayApiError(error_message)

        task_id = initial_response.id
        logging.debug("Runway task submitted. Task ID: %s", task_id)

        final_response = self.poll_for_task_status(task_id, auth_token)
        if not RunwayImageToVideoNode.is_valid_video_response(final_response):
            error_message = "Runway task succeeded but no video data found in response."
            logging.error(error_message)
            raise RunwayApiError(error_message)

        video_url = final_response.output[0]
        logging.debug("Attempting to download video from URL: %s", video_url)

        video_io = download_url_to_bytesio(video_url)
        if video_io is None:
            error_msg = f"Failed to download video from {video_url}"
            logging.error(error_msg)
            raise RunwayApiError(error_msg)
        return (VideoFromFile(video_io),)


NODE_CLASS_MAPPINGS = {
    "RunwayImageToVideoNode": RunwayImageToVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunwayImageToVideoNode": "Runway Image to Video",
}

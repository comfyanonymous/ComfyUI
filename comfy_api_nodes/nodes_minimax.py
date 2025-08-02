from typing import Union
import logging
import torch

from comfy.comfy_types.node_typing import IO
from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api_nodes.apis import (
    MinimaxVideoGenerationRequest,
    MinimaxVideoGenerationResponse,
    MinimaxFileRetrieveResponse,
    MinimaxTaskResultResponse,
    SubjectReferenceItem,
    Model
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import (
    download_url_to_bytesio,
    upload_images_to_comfyapi,
    validate_string,
)
from server import PromptServer


I2V_AVERAGE_DURATION = 114
T2V_AVERAGE_DURATION = 234

class MinimaxTextToVideoNode:
    """
    Generates videos synchronously based on a prompt, and optional parameters using MiniMax's API.
    """

    AVERAGE_DURATION = T2V_AVERAGE_DURATION

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt to guide the video generation",
                    },
                ),
                "model": (
                    [
                        "T2V-01",
                        "T2V-01-Director",
                    ],
                    {
                        "default": "T2V-01",
                        "tooltip": "Model to use for video generation",
                    },
                ),
            },
            "optional": {
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VIDEO",)
    DESCRIPTION = "Generates videos from prompts using MiniMax's API"
    FUNCTION = "generate_video"
    CATEGORY = "api node/video/MiniMax"
    API_NODE = True
    OUTPUT_NODE = True

    async def generate_video(
        self,
        prompt_text,
        seed=0,
        model="T2V-01",
        image: torch.Tensor=None, # used for ImageToVideo
        subject: torch.Tensor=None, # used for SubjectToVideo
        unique_id: Union[str, None]=None,
        **kwargs,
    ):
        '''
        Function used between MiniMax nodes - supports T2V, I2V, and S2V, based on provided arguments.
        '''
        if image is None:
            validate_string(prompt_text, field_name="prompt_text")
        # upload image, if passed in
        image_url = None
        if image is not None:
            image_url = (await upload_images_to_comfyapi(image, max_images=1, auth_kwargs=kwargs))[0]

        # TODO: figure out how to deal with subject properly, API returns invalid params when using S2V-01 model
        subject_reference = None
        if subject is not None:
            subject_url = (await upload_images_to_comfyapi(subject, max_images=1, auth_kwargs=kwargs))[0]
            subject_reference = [SubjectReferenceItem(image=subject_url)]


        video_generate_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/minimax/video_generation",
                method=HttpMethod.POST,
                request_model=MinimaxVideoGenerationRequest,
                response_model=MinimaxVideoGenerationResponse,
            ),
            request=MinimaxVideoGenerationRequest(
                model=Model(model),
                prompt=prompt_text,
                callback_url=None,
                first_frame_image=image_url,
                subject_reference=subject_reference,
                prompt_optimizer=None,
            ),
            auth_kwargs=kwargs,
        )
        response = await video_generate_operation.execute()

        task_id = response.task_id
        if not task_id:
            raise Exception(f"MiniMax generation failed: {response.base_resp}")

        video_generate_operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path="/proxy/minimax/query/video_generation",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=MinimaxTaskResultResponse,
                query_params={"task_id": task_id},
            ),
            completed_statuses=["Success"],
            failed_statuses=["Fail"],
            status_extractor=lambda x: x.status.value,
            estimated_duration=self.AVERAGE_DURATION,
            node_id=unique_id,
            auth_kwargs=kwargs,
        )
        task_result = await video_generate_operation.execute()

        file_id = task_result.file_id
        if file_id is None:
            raise Exception("Request was not successful. Missing file ID.")
        file_retrieve_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/minimax/files/retrieve",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=MinimaxFileRetrieveResponse,
                query_params={"file_id": int(file_id)},
            ),
            request=EmptyRequest(),
            auth_kwargs=kwargs,
        )
        file_result = await file_retrieve_operation.execute()

        file_url = file_result.file.download_url
        if file_url is None:
            raise Exception(
                f"No video was found in the response. Full response: {file_result.model_dump()}"
            )
        logging.info(f"Generated video URL: {file_url}")
        if unique_id:
            if hasattr(file_result.file, "backup_download_url"):
                message = f"Result URL: {file_url}\nBackup URL: {file_result.file.backup_download_url}"
            else:
                message = f"Result URL: {file_url}"
            PromptServer.instance.send_progress_text(message, unique_id)

        video_io = await download_url_to_bytesio(file_url)
        if video_io is None:
            error_msg = f"Failed to download video from {file_url}"
            logging.error(error_msg)
            raise Exception(error_msg)
        return (VideoFromFile(video_io),)


class MinimaxImageToVideoNode(MinimaxTextToVideoNode):
    """
    Generates videos synchronously based on an image and prompt, and optional parameters using MiniMax's API.
    """

    AVERAGE_DURATION = I2V_AVERAGE_DURATION

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (
                    IO.IMAGE,
                    {
                        "tooltip": "Image to use as first frame of video generation"
                    },
                ),
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt to guide the video generation",
                    },
                ),
                "model": (
                    [
                        "I2V-01-Director",
                        "I2V-01",
                        "I2V-01-live",
                    ],
                    {
                        "default": "I2V-01",
                        "tooltip": "Model to use for video generation",
                    },
                ),
            },
            "optional": {
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VIDEO",)
    DESCRIPTION = "Generates videos from an image and prompts using MiniMax's API"
    FUNCTION = "generate_video"
    CATEGORY = "api node/video/MiniMax"
    API_NODE = True
    OUTPUT_NODE = True


class MinimaxSubjectToVideoNode(MinimaxTextToVideoNode):
    """
    Generates videos synchronously based on an image and prompt, and optional parameters using MiniMax's API.
    """

    AVERAGE_DURATION = T2V_AVERAGE_DURATION

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "subject": (
                    IO.IMAGE,
                    {
                        "tooltip": "Image of subject to reference video generation"
                    },
                ),
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt to guide the video generation",
                    },
                ),
                "model": (
                    [
                        "S2V-01",
                    ],
                    {
                        "default": "S2V-01",
                        "tooltip": "Model to use for video generation",
                    },
                ),
            },
            "optional": {
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VIDEO",)
    DESCRIPTION = "Generates videos from an image and prompts using MiniMax's API"
    FUNCTION = "generate_video"
    CATEGORY = "api node/video/MiniMax"
    API_NODE = True
    OUTPUT_NODE = True


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MinimaxTextToVideoNode": MinimaxTextToVideoNode,
    "MinimaxImageToVideoNode": MinimaxImageToVideoNode,
    # "MinimaxSubjectToVideoNode": MinimaxSubjectToVideoNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxTextToVideoNode": "MiniMax Text to Video",
    "MinimaxImageToVideoNode": "MiniMax Image to Video",
    "MinimaxSubjectToVideoNode": "MiniMax Subject to Video",
}

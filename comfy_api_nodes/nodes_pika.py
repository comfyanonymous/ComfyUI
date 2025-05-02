"""Pika API docs: https://pika-827374fb.mintlify.app/api-reference"""

from typing import Optional, TypeVar
import logging
import torch
from comfy_api_nodes.apis import (
    PikaBodyGenerate22T2vGenerate22T2vPost,
    PikaGenerateResponse,
    PikaBodyGenerate22I2vGenerate22I2vPost,
    PikaVideoResponse,
    PikaBodyGenerate22C2vGenerate22PikascenesPost,
    IngredientsMode,
    PikaDurationEnum,
    PikaResolutionEnum,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import (
    tensor_to_bytesio,
    download_url_to_video_output,
)
from comfy_api_nodes.mapper_utils import model_field_to_node_input
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeOptions
from comfy_api.input_impl import VideoFromFile

R = TypeVar("R")

PIKA_API_VERSION = "2.2"
PATH_TEXT_TO_VIDEO = f"/proxy/pika/generate/{PIKA_API_VERSION}/t2v"
PATH_IMAGE_TO_VIDEO = f"/proxy/pika/generate/{PIKA_API_VERSION}/i2v"
PATH_PIKAFRAMES = f"/proxy/pika/generate/{PIKA_API_VERSION}/pikaframes"
PATH_PIKASCENES = f"/proxy/pika/generate/{PIKA_API_VERSION}/pikascenes"
PATH_VIDEO_GET = "/proxy/pika/videos"


class PikaApiError(Exception):
    """Exception for Pika API errors."""

    pass


def is_valid_video_response(response: PikaVideoResponse) -> bool:
    """Check if the video response is valid."""
    return hasattr(response, "url") and response.url is not None


def is_valid_initial_response(response: PikaGenerateResponse) -> bool:
    """Check if the initial response is valid."""
    return hasattr(response, "video_id") and response.video_id is not None


class PikaNodeBase(ComfyNodeABC):
    """Base class for Pika nodes."""

    @classmethod
    def get_base_inputs_types(
        cls, request_model
    ) -> dict[str, tuple[IO, InputTypeOptions]]:
        """Get the base required inputs types common to all Pika nodes."""
        return {
            "prompt_text": model_field_to_node_input(
                IO.STRING,
                request_model,
                "promptText",
                multiline=True,
            ),
            "negative_prompt": model_field_to_node_input(
                IO.STRING,
                request_model,
                "negativePrompt",
                multiline=True,
            ),
            "seed": model_field_to_node_input(
                IO.INT,
                request_model,
                "seed",
                min=0,
                max=0xFFFFFFFF,
                control_after_generate=True,
            ),
            "resolution": model_field_to_node_input(
                IO.COMBO,
                request_model,
                "resolution",
                enum_type=PikaResolutionEnum,
            ),
            "duration": model_field_to_node_input(
                IO.COMBO,
                request_model,
                "duration",
                enum_type=PikaDurationEnum,
            ),
        }

    CATEGORY = "api node/video/Pika"
    API_NODE = True
    FUNCTION = "api_call"

    def poll_for_task_status(
        self, task_id: str, auth_token: str
    ) -> PikaGenerateResponse:
        """Polls the Pika API endpoint until the task reaches a terminal state."""
        polling_operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"{PATH_VIDEO_GET}/{task_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=PikaVideoResponse,
            ),
            completed_statuses=[
                "finished",
            ],
            failed_statuses=["failed", "cancelled"],
            status_extractor=lambda response: (
                response.status.value if response.status else None
            ),
            progress_extractor=lambda response: (
                response.progress if hasattr(response, "progress") else None
            ),
            auth_token=auth_token,
        )
        return polling_operation.execute()

    def execute_task(
        self,
        initial_operation: SynchronousOperation[R, PikaGenerateResponse],
        auth_token: Optional[str] = None,
    ) -> tuple[VideoFromFile]:
        """Executes the initial operation then polls for the task status until it is completed.

        Args:
            initial_operation: The initial operation to execute.
            auth_token: The authentication token to use for the API call.

        Returns:
            A tuple containing the video file as a VIDEO output.
        """
        initial_response = initial_operation.execute()
        if not is_valid_initial_response(initial_response):
            error_msg = f"Pika initial request failed. Code: {initial_response.code}, Message: {initial_response.message}, Data: {initial_response.data}"
            logging.error(error_msg)
            raise PikaApiError(error_msg)

        task_id = initial_response.video_id
        final_response = self.poll_for_task_status(task_id, auth_token)
        if not is_valid_video_response(final_response):
            error_msg = (
                f"Pika task {task_id} succeeded but no video data found in response."
            )
            logging.error(error_msg)
            raise PikaApiError(error_msg)

        video_url = str(final_response.url)
        logging.debug("Pika task %s succeeded. Video URL: %s", task_id, video_url)

        return (download_url_to_video_output(video_url),)


class PikaImageToVideoV2_2(PikaNodeBase):
    """Pika 2.2 Image to Video Node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    IO.IMAGE,
                    {"tooltip": "The image to convert to video"},
                ),
                **cls.get_base_inputs_types(PikaBodyGenerate22I2vGenerate22I2vPost),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
            },
        }

    DESCRIPTION = "Sends an image and prompt to the Pika API v2.2 to generate a video."
    RETURN_TYPES = ("VIDEO",)

    def api_call(
        self,
        image: torch.Tensor,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
        auth_token: Optional[str] = None,
    ) -> tuple[VideoFromFile]:
        """API call for Pika 2.2 Image to Video."""
        # Convert image to BytesIO
        image_bytes_io = tensor_to_bytesio(image)
        image_bytes_io.seek(0)  # Reset stream position

        # Prepare file data for multipart upload
        pika_files = {"image": ("image.png", image_bytes_io, "image/png")}

        # Prepare non-file data using the Pydantic model
        pika_request_data = PikaBodyGenerate22I2vGenerate22I2vPost(
            promptText=prompt_text,
            negativePrompt=negative_prompt,
            seed=seed,
            resolution=resolution,
            duration=duration,
        )

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_IMAGE_TO_VIDEO,
                method=HttpMethod.POST,
                request_model=PikaBodyGenerate22I2vGenerate22I2vPost,
                response_model=PikaGenerateResponse,
            ),
            request=pika_request_data,
            files=pika_files,
            content_type="multipart/form-data",
            auth_token=auth_token,
        )

        return self.execute_task(initial_operation, auth_token)


class PikaTextToVideoNodeV2_2(PikaNodeBase):
    """Pika 2.2 Text to Video Node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **cls.get_base_inputs_types(PikaBodyGenerate22T2vGenerate22T2vPost),
                "aspect_ratio": model_field_to_node_input(
                    IO.FLOAT,
                    PikaBodyGenerate22T2vGenerate22T2vPost,
                    "aspectRatio",
                    step=0.001,
                    min=0.4,
                    max=2.5,
                    default=1.7777777777777777,
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
            },
        }

    RETURN_TYPES = ("VIDEO",)
    DESCRIPTION = "Sends a text prompt to the Pika API v2.2 to generate a video."

    def api_call(
        self,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
        aspect_ratio: float,
        auth_token: Optional[str] = None,
    ) -> tuple[VideoFromFile]:
        """API call for Pika 2.2 Text to Video."""
        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_TEXT_TO_VIDEO,
                method=HttpMethod.POST,
                request_model=PikaBodyGenerate22T2vGenerate22T2vPost,
                response_model=PikaGenerateResponse,
            ),
            request=PikaBodyGenerate22T2vGenerate22T2vPost(
                promptText=prompt_text,
                negativePrompt=negative_prompt,
                seed=seed,
                resolution=resolution,
                duration=duration,
                aspectRatio=aspect_ratio,
            ),
            auth_token=auth_token,
            content_type="application/x-www-form-urlencoded",
        )

        return self.execute_task(initial_operation, auth_token)


class PikaScenesV2_2(PikaNodeBase):
    """Pika 2.2 Scenes Node."""

    @classmethod
    def INPUT_TYPES(cls):
        image_ingredient_input = (
            IO.IMAGE,
            {"tooltip": "Image that will be used as ingredient to create a video."},
        )
        return {
            "required": {
                **cls.get_base_inputs_types(
                    PikaBodyGenerate22C2vGenerate22PikascenesPost,
                ),
                "ingredients_mode": model_field_to_node_input(
                    IO.COMBO,
                    PikaBodyGenerate22C2vGenerate22PikascenesPost,
                    "ingredientsMode",
                    enum_type=IngredientsMode,
                    default="creative",
                ),
                "aspect_ratio": model_field_to_node_input(
                    IO.FLOAT,
                    PikaBodyGenerate22C2vGenerate22PikascenesPost,
                    "aspectRatio",
                    step=0.001,
                    min=0.4,
                    max=2.5,
                    default=1.7777777777777777,
                ),
            },
            "optional": {
                "image_ingredient_1": image_ingredient_input,
                "image_ingredient_2": image_ingredient_input,
                "image_ingredient_3": image_ingredient_input,
                "image_ingredient_4": image_ingredient_input,
                "image_ingredient_5": image_ingredient_input,
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
            },
        }

    DESCRIPTION = "Combine your images to create a video with the objects in them. Upload multiple images as ingredients and generate a high-quality video that incorporates all of them."
    RETURN_TYPES = ("VIDEO",)

    def api_call(
        self,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
        ingredients_mode: str,
        aspect_ratio: float,
        image_ingredient_1: Optional[torch.Tensor] = None,
        image_ingredient_2: Optional[torch.Tensor] = None,
        image_ingredient_3: Optional[torch.Tensor] = None,
        image_ingredient_4: Optional[torch.Tensor] = None,
        image_ingredient_5: Optional[torch.Tensor] = None,
        auth_token: Optional[str] = None,
    ) -> tuple[VideoFromFile]:
        """API call for Pika Scenes 2.2."""
        all_image_bytes_io = []
        for image in [
            image_ingredient_1,
            image_ingredient_2,
            image_ingredient_3,
            image_ingredient_4,
            image_ingredient_5,
        ]:
            if image is not None:
                image_bytes_io = tensor_to_bytesio(image)
                image_bytes_io.seek(0)
                all_image_bytes_io.append(image_bytes_io)

        # Prepare files data for multipart upload
        pika_files = [
            ("images", (f"image_{i}.png", image_bytes_io, "image/png"))
            for i, image_bytes_io in enumerate(all_image_bytes_io)
        ]

        # Prepare non-file data using the Pydantic model
        pika_request_data = PikaBodyGenerate22C2vGenerate22PikascenesPost(
            ingredientsMode=ingredients_mode,
            promptText=prompt_text,
            negativePrompt=negative_prompt,
            seed=seed,
            resolution=resolution,
            duration=duration,
            aspectRatio=aspect_ratio,
        )

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_PIKASCENES,
                method=HttpMethod.POST,
                request_model=PikaBodyGenerate22C2vGenerate22PikascenesPost,
                response_model=PikaGenerateResponse,
            ),
            request=pika_request_data,
            files=pika_files,
            content_type="multipart/form-data",
            auth_token=auth_token,
        )

        return self.execute_task(initial_operation, auth_token)


NODE_CLASS_MAPPINGS = {
    "PikaImageToVideoNode2_2": PikaImageToVideoV2_2,
    "PikaTextToVideoNode2_2": PikaTextToVideoNodeV2_2,
    "PikaScenesV2_2": PikaScenesV2_2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PikaImageToVideoNode2_2": "Pika 2.2 Image to Video",
    "PikaTextToVideoNode2_2": "Pika 2.2 Text to Video",
    "PikaScenesV2_2": "Pika 2.2 Scenes",
}

"""
Pika x ComfyUI API Nodes

Pika API docs: https://pika-827374fb.mintlify.app/api-reference
"""
from __future__ import annotations

import io
import logging
from typing import Optional, TypeVar

import numpy as np
import torch

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeOptions
from comfy_api.input_impl import VideoFromFile
from comfy_api.input_impl.video_types import VideoCodec, VideoContainer, VideoInput
from comfy_api_nodes.apinode_utils import (
    download_url_to_video_output,
    tensor_to_bytesio,
)
from comfy_api_nodes.apis import (
    IngredientsMode,
    PikaBodyGenerate22C2vGenerate22PikascenesPost,
    PikaBodyGenerate22I2vGenerate22I2vPost,
    PikaBodyGenerate22KeyframeGenerate22PikaframesPost,
    PikaBodyGenerate22T2vGenerate22T2vPost,
    PikaBodyGeneratePikadditionsGeneratePikadditionsPost,
    PikaBodyGeneratePikaffectsGeneratePikaffectsPost,
    PikaBodyGeneratePikaswapsGeneratePikaswapsPost,
    PikaDurationEnum,
    Pikaffect,
    PikaGenerateResponse,
    PikaResolutionEnum,
    PikaVideoResponse,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    EmptyRequest,
    HttpMethod,
    PollingOperation,
    SynchronousOperation,
)
from comfy_api_nodes.mapper_utils import model_field_to_node_input

R = TypeVar("R")

PATH_PIKADDITIONS = "/proxy/pika/generate/pikadditions"
PATH_PIKASWAPS = "/proxy/pika/generate/pikaswaps"
PATH_PIKAFFECTS = "/proxy/pika/generate/pikaffects"

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
    RETURN_TYPES = ("VIDEO",)

    async def poll_for_task_status(
        self,
        task_id: str,
        auth_kwargs: Optional[dict[str, str]] = None,
        node_id: Optional[str] = None,
    ) -> PikaGenerateResponse:
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
            auth_kwargs=auth_kwargs,
            result_url_extractor=lambda response: (
                response.url if hasattr(response, "url") else None
            ),
            node_id=node_id,
            estimated_duration=60
        )
        return await polling_operation.execute()

    async def execute_task(
        self,
        initial_operation: SynchronousOperation[R, PikaGenerateResponse],
        auth_kwargs: Optional[dict[str, str]] = None,
        node_id: Optional[str] = None,
    ) -> tuple[VideoFromFile]:
        """Executes the initial operation then polls for the task status until it is completed.

        Args:
            initial_operation: The initial operation to execute.
            auth_kwargs: The authentication token(s) to use for the API call.

        Returns:
            A tuple containing the video file as a VIDEO output.
        """
        initial_response = await initial_operation.execute()
        if not is_valid_initial_response(initial_response):
            error_msg = f"Pika initial request failed. Code: {initial_response.code}, Message: {initial_response.message}, Data: {initial_response.data}"
            logging.error(error_msg)
            raise PikaApiError(error_msg)

        task_id = initial_response.video_id
        final_response = await self.poll_for_task_status(task_id, auth_kwargs)
        if not is_valid_video_response(final_response):
            error_msg = (
                f"Pika task {task_id} succeeded but no video data found in response."
            )
            logging.error(error_msg)
            raise PikaApiError(error_msg)

        video_url = str(final_response.url)
        logging.info("Pika task %s succeeded. Video URL: %s", task_id, video_url)

        return (await download_url_to_video_output(video_url),)


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
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Sends an image and prompt to the Pika API v2.2 to generate a video."

    async def api_call(
        self,
        image: torch.Tensor,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
        unique_id: str,
        **kwargs,
    ) -> tuple[VideoFromFile]:
        # Convert image to BytesIO
        image_bytes_io = tensor_to_bytesio(image)
        image_bytes_io.seek(0)

        pika_files = {"image": ("image.png", image_bytes_io, "image/png")}

        # Prepare non-file data
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
            auth_kwargs=kwargs,
        )

        return await self.execute_task(initial_operation, auth_kwargs=kwargs, node_id=unique_id)


class PikaTextToVideoNodeV2_2(PikaNodeBase):
    """Pika Text2Video v2.2 Node."""

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
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Sends a text prompt to the Pika API v2.2 to generate a video."

    async def api_call(
        self,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
        aspect_ratio: float,
        unique_id: str,
        **kwargs,
    ) -> tuple[VideoFromFile]:
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
            auth_kwargs=kwargs,
            content_type="application/x-www-form-urlencoded",
        )

        return await self.execute_task(initial_operation, auth_kwargs=kwargs, node_id=unique_id)


class PikaScenesV2_2(PikaNodeBase):
    """PikaScenes v2.2 Node."""

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
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Combine your images to create a video with the objects in them. Upload multiple images as ingredients and generate a high-quality video that incorporates all of them."

    async def api_call(
        self,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
        ingredients_mode: str,
        aspect_ratio: float,
        unique_id: str,
        image_ingredient_1: Optional[torch.Tensor] = None,
        image_ingredient_2: Optional[torch.Tensor] = None,
        image_ingredient_3: Optional[torch.Tensor] = None,
        image_ingredient_4: Optional[torch.Tensor] = None,
        image_ingredient_5: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[VideoFromFile]:
        # Convert all passed images to BytesIO
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

        pika_files = [
            ("images", (f"image_{i}.png", image_bytes_io, "image/png"))
            for i, image_bytes_io in enumerate(all_image_bytes_io)
        ]

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
            auth_kwargs=kwargs,
        )

        return await self.execute_task(initial_operation, auth_kwargs=kwargs, node_id=unique_id)


class PikAdditionsNode(PikaNodeBase):
    """Pika Pikadditions Node. Add an image into a video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to add an image to."}),
                "image": (IO.IMAGE, {"tooltip": "The image to add to the video."}),
                "prompt_text": model_field_to_node_input(
                    IO.STRING,
                    PikaBodyGeneratePikadditionsGeneratePikadditionsPost,
                    "promptText",
                    multiline=True,
                ),
                "negative_prompt": model_field_to_node_input(
                    IO.STRING,
                    PikaBodyGeneratePikadditionsGeneratePikadditionsPost,
                    "negativePrompt",
                    multiline=True,
                ),
                "seed": model_field_to_node_input(
                    IO.INT,
                    PikaBodyGeneratePikadditionsGeneratePikadditionsPost,
                    "seed",
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Add any object or image into your video. Upload a video and specify what you'd like to add to create a seamlessly integrated result."

    async def api_call(
        self,
        video: VideoInput,
        image: torch.Tensor,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        unique_id: str,
        **kwargs,
    ) -> tuple[VideoFromFile]:
        # Convert video to BytesIO
        video_bytes_io = io.BytesIO()
        video.save_to(video_bytes_io, format=VideoContainer.MP4, codec=VideoCodec.H264)
        video_bytes_io.seek(0)

        # Convert image to BytesIO
        image_bytes_io = tensor_to_bytesio(image)
        image_bytes_io.seek(0)

        pika_files = {
            "video": ("video.mp4", video_bytes_io, "video/mp4"),
            "image": ("image.png", image_bytes_io, "image/png"),
        }

        # Prepare non-file data
        pika_request_data = PikaBodyGeneratePikadditionsGeneratePikadditionsPost(
            promptText=prompt_text,
            negativePrompt=negative_prompt,
            seed=seed,
        )

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_PIKADDITIONS,
                method=HttpMethod.POST,
                request_model=PikaBodyGeneratePikadditionsGeneratePikadditionsPost,
                response_model=PikaGenerateResponse,
            ),
            request=pika_request_data,
            files=pika_files,
            content_type="multipart/form-data",
            auth_kwargs=kwargs,
        )

        return await self.execute_task(initial_operation, auth_kwargs=kwargs, node_id=unique_id)


class PikaSwapsNode(PikaNodeBase):
    """Pika Pikaswaps Node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to swap an object in."}),
                "image": (
                    IO.IMAGE,
                    {
                        "tooltip": "The image used to replace the masked object in the video."
                    },
                ),
                "mask": (
                    IO.MASK,
                    {"tooltip": "Use the mask to define areas in the video to replace"},
                ),
                "prompt_text": model_field_to_node_input(
                    IO.STRING,
                    PikaBodyGeneratePikaswapsGeneratePikaswapsPost,
                    "promptText",
                    multiline=True,
                ),
                "negative_prompt": model_field_to_node_input(
                    IO.STRING,
                    PikaBodyGeneratePikaswapsGeneratePikaswapsPost,
                    "negativePrompt",
                    multiline=True,
                ),
                "seed": model_field_to_node_input(
                    IO.INT,
                    PikaBodyGeneratePikaswapsGeneratePikaswapsPost,
                    "seed",
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Swap out any object or region of your video with a new image or object. Define areas to replace either with a mask or coordinates."
    RETURN_TYPES = ("VIDEO",)

    async def api_call(
        self,
        video: VideoInput,
        image: torch.Tensor,
        mask: torch.Tensor,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        unique_id: str,
        **kwargs,
    ) -> tuple[VideoFromFile]:
        # Convert video to BytesIO
        video_bytes_io = io.BytesIO()
        video.save_to(video_bytes_io, format=VideoContainer.MP4, codec=VideoCodec.H264)
        video_bytes_io.seek(0)

        # Convert mask to binary mask with three channels
        mask = torch.round(mask)
        mask = mask.repeat(1, 3, 1, 1)

        # Convert 3-channel binary mask to BytesIO
        mask_bytes_io = io.BytesIO()
        mask_bytes_io.write(mask.numpy().astype(np.uint8))
        mask_bytes_io.seek(0)

        # Convert image to BytesIO
        image_bytes_io = tensor_to_bytesio(image)
        image_bytes_io.seek(0)

        pika_files = {
            "video": ("video.mp4", video_bytes_io, "video/mp4"),
            "image": ("image.png", image_bytes_io, "image/png"),
            "modifyRegionMask": ("mask.png", mask_bytes_io, "image/png"),
        }

        # Prepare non-file data
        pika_request_data = PikaBodyGeneratePikaswapsGeneratePikaswapsPost(
            promptText=prompt_text,
            negativePrompt=negative_prompt,
            seed=seed,
        )

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_PIKADDITIONS,
                method=HttpMethod.POST,
                request_model=PikaBodyGeneratePikadditionsGeneratePikadditionsPost,
                response_model=PikaGenerateResponse,
            ),
            request=pika_request_data,
            files=pika_files,
            content_type="multipart/form-data",
            auth_kwargs=kwargs,
        )

        return await self.execute_task(initial_operation, auth_kwargs=kwargs, node_id=unique_id)


class PikaffectsNode(PikaNodeBase):
    """Pika Pikaffects Node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    IO.IMAGE,
                    {"tooltip": "The reference image to apply the Pikaffect to."},
                ),
                "pikaffect": model_field_to_node_input(
                    IO.COMBO,
                    PikaBodyGeneratePikaffectsGeneratePikaffectsPost,
                    "pikaffect",
                    enum_type=Pikaffect,
                    default="Cake-ify",
                ),
                "prompt_text": model_field_to_node_input(
                    IO.STRING,
                    PikaBodyGeneratePikaffectsGeneratePikaffectsPost,
                    "promptText",
                    multiline=True,
                ),
                "negative_prompt": model_field_to_node_input(
                    IO.STRING,
                    PikaBodyGeneratePikaffectsGeneratePikaffectsPost,
                    "negativePrompt",
                    multiline=True,
                ),
                "seed": model_field_to_node_input(
                    IO.INT,
                    PikaBodyGeneratePikaffectsGeneratePikaffectsPost,
                    "seed",
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Generate a video with a specific Pikaffect. Supported Pikaffects: Cake-ify, Crumble, Crush, Decapitate, Deflate, Dissolve, Explode, Eye-pop, Inflate, Levitate, Melt, Peel, Poke, Squish, Ta-da, Tear"

    async def api_call(
        self,
        image: torch.Tensor,
        pikaffect: str,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        unique_id: str,
        **kwargs,
    ) -> tuple[VideoFromFile]:

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_PIKAFFECTS,
                method=HttpMethod.POST,
                request_model=PikaBodyGeneratePikaffectsGeneratePikaffectsPost,
                response_model=PikaGenerateResponse,
            ),
            request=PikaBodyGeneratePikaffectsGeneratePikaffectsPost(
                pikaffect=pikaffect,
                promptText=prompt_text,
                negativePrompt=negative_prompt,
                seed=seed,
            ),
            files={"image": ("image.png", tensor_to_bytesio(image), "image/png")},
            content_type="multipart/form-data",
            auth_kwargs=kwargs,
        )

        return await self.execute_task(initial_operation, auth_kwargs=kwargs, node_id=unique_id)


class PikaStartEndFrameNode2_2(PikaNodeBase):
    """PikaFrames v2.2 Node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_start": (IO.IMAGE, {"tooltip": "The first image to combine."}),
                "image_end": (IO.IMAGE, {"tooltip": "The last image to combine."}),
                **cls.get_base_inputs_types(
                    PikaBodyGenerate22KeyframeGenerate22PikaframesPost
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Generate a video by combining your first and last frame. Upload two images to define the start and end points, and let the AI create a smooth transition between them."

    async def api_call(
        self,
        image_start: torch.Tensor,
        image_end: torch.Tensor,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
        unique_id: str,
        **kwargs,
    ) -> tuple[VideoFromFile]:

        pika_files = [
            ("keyFrames", ("image_start.png", tensor_to_bytesio(image_start), "image/png")),
            ("keyFrames", ("image_end.png", tensor_to_bytesio(image_end), "image/png")),
        ]

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_PIKAFRAMES,
                method=HttpMethod.POST,
                request_model=PikaBodyGenerate22KeyframeGenerate22PikaframesPost,
                response_model=PikaGenerateResponse,
            ),
            request=PikaBodyGenerate22KeyframeGenerate22PikaframesPost(
                promptText=prompt_text,
                negativePrompt=negative_prompt,
                seed=seed,
                resolution=resolution,
                duration=duration,
            ),
            files=pika_files,
            content_type="multipart/form-data",
            auth_kwargs=kwargs,
        )

        return await self.execute_task(initial_operation, auth_kwargs=kwargs, node_id=unique_id)


NODE_CLASS_MAPPINGS = {
    "PikaImageToVideoNode2_2": PikaImageToVideoV2_2,
    "PikaTextToVideoNode2_2": PikaTextToVideoNodeV2_2,
    "PikaScenesV2_2": PikaScenesV2_2,
    "Pikadditions": PikAdditionsNode,
    "Pikaswaps": PikaSwapsNode,
    "Pikaffects": PikaffectsNode,
    "PikaStartEndFrameNode2_2": PikaStartEndFrameNode2_2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PikaImageToVideoNode2_2": "Pika Image to Video",
    "PikaTextToVideoNode2_2": "Pika Text to Video",
    "PikaScenesV2_2": "Pika Scenes (Video Image Composition)",
    "Pikadditions": "Pikadditions (Video Object Insertion)",
    "Pikaswaps": "Pika Swaps (Video Object Replacement)",
    "Pikaffects": "Pikaffects (Video Effects)",
    "PikaStartEndFrameNode2_2": "Pika Start and End Frame to Video",
}

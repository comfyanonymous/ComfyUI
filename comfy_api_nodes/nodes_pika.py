"""
Pika x ComfyUI API Nodes

Pika API docs: https://pika-827374fb.mintlify.app/api-reference
"""
from __future__ import annotations

from io import BytesIO
import logging
from typing import Optional, TypeVar
from enum import Enum

import numpy as np
import torch

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io as comfy_io
from comfy_api.input_impl import VideoFromFile
from comfy_api.input_impl.video_types import VideoCodec, VideoContainer, VideoInput
from comfy_api_nodes.apinode_utils import (
    download_url_to_video_output,
    tensor_to_bytesio,
)
from comfy_api_nodes.apis import (
    PikaBodyGenerate22C2vGenerate22PikascenesPost,
    PikaBodyGenerate22I2vGenerate22I2vPost,
    PikaBodyGenerate22KeyframeGenerate22PikaframesPost,
    PikaBodyGenerate22T2vGenerate22T2vPost,
    PikaBodyGeneratePikadditionsGeneratePikadditionsPost,
    PikaBodyGeneratePikaffectsGeneratePikaffectsPost,
    PikaBodyGeneratePikaswapsGeneratePikaswapsPost,
    PikaGenerateResponse,
    PikaVideoResponse,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    EmptyRequest,
    HttpMethod,
    PollingOperation,
    SynchronousOperation,
)

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


class PikaDurationEnum(int, Enum):
    integer_5 = 5
    integer_10 = 10


class PikaResolutionEnum(str, Enum):
    field_1080p = "1080p"
    field_720p = "720p"


class Pikaffect(str, Enum):
    Cake_ify = "Cake-ify"
    Crumble = "Crumble"
    Crush = "Crush"
    Decapitate = "Decapitate"
    Deflate = "Deflate"
    Dissolve = "Dissolve"
    Explode = "Explode"
    Eye_pop = "Eye-pop"
    Inflate = "Inflate"
    Levitate = "Levitate"
    Melt = "Melt"
    Peel = "Peel"
    Poke = "Poke"
    Squish = "Squish"
    Ta_da = "Ta-da"
    Tear = "Tear"


class PikaApiError(Exception):
    """Exception for Pika API errors."""

    pass


def is_valid_video_response(response: PikaVideoResponse) -> bool:
    """Check if the video response is valid."""
    return hasattr(response, "url") and response.url is not None


def is_valid_initial_response(response: PikaGenerateResponse) -> bool:
    """Check if the initial response is valid."""
    return hasattr(response, "video_id") and response.video_id is not None


async def poll_for_task_status(
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
    final_response = await poll_for_task_status(task_id, auth_kwargs, node_id=node_id)
    if not is_valid_video_response(final_response):
        error_msg = (
            f"Pika task {task_id} succeeded but no video data found in response."
        )
        logging.error(error_msg)
        raise PikaApiError(error_msg)

    video_url = str(final_response.url)
    logging.info("Pika task %s succeeded. Video URL: %s", task_id, video_url)

    return (await download_url_to_video_output(video_url),)


def get_base_inputs_types() -> list[comfy_io.Input]:
    """Get the base required inputs types common to all Pika nodes."""
    return [
        comfy_io.String.Input("prompt_text", multiline=True),
        comfy_io.String.Input("negative_prompt", multiline=True),
        comfy_io.Int.Input("seed", min=0, max=0xFFFFFFFF, control_after_generate=True),
        comfy_io.Combo.Input(
            "resolution", options=PikaResolutionEnum, default=PikaResolutionEnum.field_1080p
        ),
        comfy_io.Combo.Input(
            "duration", options=PikaDurationEnum, default=PikaDurationEnum.integer_5
        ),
    ]


class PikaImageToVideoV2_2(comfy_io.ComfyNode):
    """Pika 2.2 Image to Video Node."""

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="PikaImageToVideoNode2_2",
            display_name="Pika Image to Video",
            description="Sends an image and prompt to the Pika API v2.2 to generate a video.",
            category="api node/video/Pika",
            inputs=[
                comfy_io.Image.Input("image", tooltip="The image to convert to video"),
                *get_base_inputs_types(),
            ],
            outputs=[comfy_io.Video.Output()],
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
        image: torch.Tensor,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
    ) -> comfy_io.NodeOutput:
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
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
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
            auth_kwargs=auth,
        )
        return await execute_task(initial_operation, auth_kwargs=auth, node_id=cls.hidden.unique_id)


class PikaTextToVideoNodeV2_2(comfy_io.ComfyNode):
    """Pika Text2Video v2.2 Node."""

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="PikaTextToVideoNode2_2",
            display_name="Pika Text to Video",
            description="Sends a text prompt to the Pika API v2.2 to generate a video.",
            category="api node/video/Pika",
            inputs=[
                *get_base_inputs_types(),
                comfy_io.Float.Input(
                    "aspect_ratio",
                    step=0.001,
                    min=0.4,
                    max=2.5,
                    default=1.7777777777777777,
                    tooltip="Aspect ratio (width / height)",
                )
            ],
            outputs=[comfy_io.Video.Output()],
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
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
        aspect_ratio: float,
    ) -> comfy_io.NodeOutput:
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
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
            auth_kwargs=auth,
            content_type="application/x-www-form-urlencoded",
        )
        return await execute_task(initial_operation, auth_kwargs=auth, node_id=cls.hidden.unique_id)


class PikaScenesV2_2(comfy_io.ComfyNode):
    """PikaScenes v2.2 Node."""

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="PikaScenesV2_2",
            display_name="Pika Scenes (Video Image Composition)",
            description="Combine your images to create a video with the objects in them. Upload multiple images as ingredients and generate a high-quality video that incorporates all of them.",
            category="api node/video/Pika",
            inputs=[
                *get_base_inputs_types(),
                comfy_io.Combo.Input(
                    "ingredients_mode",
                    options=["creative", "precise"],
                    default="creative",
                ),
                comfy_io.Float.Input(
                    "aspect_ratio",
                    step=0.001,
                    min=0.4,
                    max=2.5,
                    default=1.7777777777777777,
                    tooltip="Aspect ratio (width / height)",
                ),
                comfy_io.Image.Input(
                    "image_ingredient_1",
                    optional=True,
                    tooltip="Image that will be used as ingredient to create a video.",
                ),
                comfy_io.Image.Input(
                    "image_ingredient_2",
                    optional=True,
                    tooltip="Image that will be used as ingredient to create a video.",
                ),
                comfy_io.Image.Input(
                    "image_ingredient_3",
                    optional=True,
                    tooltip="Image that will be used as ingredient to create a video.",
                ),
                comfy_io.Image.Input(
                    "image_ingredient_4",
                    optional=True,
                    tooltip="Image that will be used as ingredient to create a video.",
                ),
                comfy_io.Image.Input(
                    "image_ingredient_5",
                    optional=True,
                    tooltip="Image that will be used as ingredient to create a video.",
                ),
            ],
            outputs=[comfy_io.Video.Output()],
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
    ) -> comfy_io.NodeOutput:
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
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
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
            auth_kwargs=auth,
        )

        return await execute_task(initial_operation, auth_kwargs=auth, node_id=cls.hidden.unique_id)


class PikAdditionsNode(comfy_io.ComfyNode):
    """Pika Pikadditions Node. Add an image into a video."""

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="Pikadditions",
            display_name="Pikadditions (Video Object Insertion)",
            description="Add any object or image into your video. Upload a video and specify what you'd like to add to create a seamlessly integrated result.",
            category="api node/video/Pika",
            inputs=[
                comfy_io.Video.Input("video", tooltip="The video to add an image to."),
                comfy_io.Image.Input("image", tooltip="The image to add to the video."),
                comfy_io.String.Input("prompt_text", multiline=True),
                comfy_io.String.Input("negative_prompt", multiline=True),
                comfy_io.Int.Input(
                    "seed",
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                ),
            ],
            outputs=[comfy_io.Video.Output()],
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
        video: VideoInput,
        image: torch.Tensor,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
    ) -> comfy_io.NodeOutput:
        # Convert video to BytesIO
        video_bytes_io = BytesIO()
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
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
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
            auth_kwargs=auth,
        )

        return await execute_task(initial_operation, auth_kwargs=auth, node_id=cls.hidden.unique_id)


class PikaSwapsNode(comfy_io.ComfyNode):
    """Pika Pikaswaps Node."""

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="Pikaswaps",
            display_name="Pika Swaps (Video Object Replacement)",
            description="Swap out any object or region of your video with a new image or object. Define areas to replace either with a mask or coordinates.",
            category="api node/video/Pika",
            inputs=[
                comfy_io.Video.Input("video", tooltip="The video to swap an object in."),
                comfy_io.Image.Input("image", tooltip="The image used to replace the masked object in the video."),
                comfy_io.Mask.Input("mask", tooltip="Use the mask to define areas in the video to replace"),
                comfy_io.String.Input("prompt_text", multiline=True),
                comfy_io.String.Input("negative_prompt", multiline=True),
                comfy_io.Int.Input("seed", min=0, max=0xFFFFFFFF, control_after_generate=True),
            ],
            outputs=[comfy_io.Video.Output()],
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
        video: VideoInput,
        image: torch.Tensor,
        mask: torch.Tensor,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
    ) -> comfy_io.NodeOutput:
        # Convert video to BytesIO
        video_bytes_io = BytesIO()
        video.save_to(video_bytes_io, format=VideoContainer.MP4, codec=VideoCodec.H264)
        video_bytes_io.seek(0)

        # Convert mask to binary mask with three channels
        mask = torch.round(mask)
        mask = mask.repeat(1, 3, 1, 1)

        # Convert 3-channel binary mask to BytesIO
        mask_bytes_io = BytesIO()
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
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
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
            auth_kwargs=auth,
        )
        return await execute_task(initial_operation, auth_kwargs=auth, node_id=cls.hidden.unique_id)


class PikaffectsNode(comfy_io.ComfyNode):
    """Pika Pikaffects Node."""

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="Pikaffects",
            display_name="Pikaffects (Video Effects)",
            description="Generate a video with a specific Pikaffect. Supported Pikaffects: Cake-ify, Crumble, Crush, Decapitate, Deflate, Dissolve, Explode, Eye-pop, Inflate, Levitate, Melt, Peel, Poke, Squish, Ta-da, Tear",
            category="api node/video/Pika",
            inputs=[
                comfy_io.Image.Input("image", tooltip="The reference image to apply the Pikaffect to."),
                comfy_io.Combo.Input(
                    "pikaffect", options=Pikaffect, default="Cake-ify"
                ),
                comfy_io.String.Input("prompt_text", multiline=True),
                comfy_io.String.Input("negative_prompt", multiline=True),
                comfy_io.Int.Input("seed", min=0, max=0xFFFFFFFF, control_after_generate=True),
            ],
            outputs=[comfy_io.Video.Output()],
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
        image: torch.Tensor,
        pikaffect: str,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
    ) -> comfy_io.NodeOutput:
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
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
            auth_kwargs=auth,
        )
        return await execute_task(initial_operation, auth_kwargs=auth, node_id=cls.hidden.unique_id)


class PikaStartEndFrameNode2_2(comfy_io.ComfyNode):
    """PikaFrames v2.2 Node."""

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="PikaStartEndFrameNode2_2",
            display_name="Pika Start and End Frame to Video",
            description="Generate a video by combining your first and last frame. Upload two images to define the start and end points, and let the AI create a smooth transition between them.",
            category="api node/video/Pika",
            inputs=[
                comfy_io.Image.Input("image_start", tooltip="The first image to combine."),
                comfy_io.Image.Input("image_end", tooltip="The last image to combine."),
                *get_base_inputs_types(),
            ],
            outputs=[comfy_io.Video.Output()],
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
        image_start: torch.Tensor,
        image_end: torch.Tensor,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
    ) -> comfy_io.NodeOutput:
        pika_files = [
            ("keyFrames", ("image_start.png", tensor_to_bytesio(image_start), "image/png")),
            ("keyFrames", ("image_end.png", tensor_to_bytesio(image_end), "image/png")),
        ]
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
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
            auth_kwargs=auth,
        )
        return await execute_task(initial_operation, auth_kwargs=auth, node_id=cls.hidden.unique_id)


class PikaApiNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[comfy_io.ComfyNode]]:
        return [
            PikaImageToVideoV2_2,
            PikaTextToVideoNodeV2_2,
            PikaScenesV2_2,
            PikAdditionsNode,
            PikaSwapsNode,
            PikaffectsNode,
            PikaStartEndFrameNode2_2,
        ]


async def comfy_entrypoint() -> PikaApiNodesExtension:
    return PikaApiNodesExtension()

"""
Pika x ComfyUI API Nodes

Pika API docs: https://pika-827374fb.mintlify.app/api-reference
"""
from __future__ import annotations

from io import BytesIO
import logging
from typing import Optional, TypeVar

import torch

from typing_extensions import override
from comfy_api.latest import ComfyExtension, comfy_io
from comfy_api.input_impl.video_types import VideoCodec, VideoContainer, VideoInput
from comfy_api_nodes.apinode_utils import (
    download_url_to_video_output,
    tensor_to_bytesio,
    validate_string,
)
from comfy_api_nodes.apis import pika_defs
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


async def execute_task(
    initial_operation: SynchronousOperation[R, pika_defs.PikaGenerateResponse],
    auth_kwargs: Optional[dict[str, str]] = None,
    node_id: Optional[str] = None,
) -> comfy_io.NodeOutput:
    task_id = (await initial_operation.execute()).video_id
    final_response: pika_defs.PikaVideoResponse = await PollingOperation(
        poll_endpoint=ApiEndpoint(
            path=f"{PATH_VIDEO_GET}/{task_id}",
            method=HttpMethod.GET,
            request_model=EmptyRequest,
            response_model=pika_defs.PikaVideoResponse,
        ),
        completed_statuses=["finished"],
        failed_statuses=["failed", "cancelled"],
        status_extractor=lambda response: (response.status.value if response.status else None),
        progress_extractor=lambda response: (response.progress if hasattr(response, "progress") else None),
        auth_kwargs=auth_kwargs,
        result_url_extractor=lambda response: (response.url if hasattr(response, "url") else None),
        node_id=node_id,
        estimated_duration=60,
        max_poll_attempts=240,
    ).execute()
    if not final_response.url:
        error_msg = f"Pika task {task_id} succeeded but no video data found in response:\n{final_response}"
        logging.error(error_msg)
        raise Exception(error_msg)
    video_url = final_response.url
    logging.info("Pika task %s succeeded. Video URL: %s", task_id, video_url)
    return comfy_io.NodeOutput(await download_url_to_video_output(video_url))


def get_base_inputs_types() -> list[comfy_io.Input]:
    """Get the base required inputs types common to all Pika nodes."""
    return [
        comfy_io.String.Input("prompt_text", multiline=True),
        comfy_io.String.Input("negative_prompt", multiline=True),
        comfy_io.Int.Input("seed", min=0, max=0xFFFFFFFF, control_after_generate=True),
        comfy_io.Combo.Input("resolution", options=["1080p", "720p"], default="1080p"),
        comfy_io.Combo.Input("duration", options=[5, 10], default=5),
    ]


class PikaImageToVideo(comfy_io.ComfyNode):
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
        image_bytes_io = tensor_to_bytesio(image)
        pika_files = {"image": ("image.png", image_bytes_io, "image/png")}
        pika_request_data = pika_defs.PikaBodyGenerate22I2vGenerate22I2vPost(
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
                request_model=pika_defs.PikaBodyGenerate22I2vGenerate22I2vPost,
                response_model=pika_defs.PikaGenerateResponse,
            ),
            request=pika_request_data,
            files=pika_files,
            content_type="multipart/form-data",
            auth_kwargs=auth,
        )
        return await execute_task(initial_operation, auth_kwargs=auth, node_id=cls.hidden.unique_id)


class PikaTextToVideoNode(comfy_io.ComfyNode):
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
                request_model=pika_defs.PikaBodyGenerate22T2vGenerate22T2vPost,
                response_model=pika_defs.PikaGenerateResponse,
            ),
            request=pika_defs.PikaBodyGenerate22T2vGenerate22T2vPost(
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


class PikaScenes(comfy_io.ComfyNode):
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
        all_image_bytes_io = []
        for image in [
            image_ingredient_1,
            image_ingredient_2,
            image_ingredient_3,
            image_ingredient_4,
            image_ingredient_5,
        ]:
            if image is not None:
                all_image_bytes_io.append(tensor_to_bytesio(image))

        pika_files = [
            ("images", (f"image_{i}.png", image_bytes_io, "image/png"))
            for i, image_bytes_io in enumerate(all_image_bytes_io)
        ]

        pika_request_data = pika_defs.PikaBodyGenerate22C2vGenerate22PikascenesPost(
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
                request_model=pika_defs.PikaBodyGenerate22C2vGenerate22PikascenesPost,
                response_model=pika_defs.PikaGenerateResponse,
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
        video_bytes_io = BytesIO()
        video.save_to(video_bytes_io, format=VideoContainer.MP4, codec=VideoCodec.H264)
        video_bytes_io.seek(0)

        image_bytes_io = tensor_to_bytesio(image)
        pika_files = {
            "video": ("video.mp4", video_bytes_io, "video/mp4"),
            "image": ("image.png", image_bytes_io, "image/png"),
        }
        pika_request_data = pika_defs.PikaBodyGeneratePikadditionsGeneratePikadditionsPost(
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
                request_model=pika_defs.PikaBodyGeneratePikadditionsGeneratePikadditionsPost,
                response_model=pika_defs.PikaGenerateResponse,
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
                comfy_io.Image.Input(
                    "image",
                    tooltip="The image used to replace the masked object in the video.",
                    optional=True,
                ),
                comfy_io.Mask.Input(
                    "mask",
                    tooltip="Use the mask to define areas in the video to replace.",
                    optional=True,
                ),
                comfy_io.String.Input("prompt_text", multiline=True, optional=True),
                comfy_io.String.Input("negative_prompt", multiline=True, optional=True),
                comfy_io.Int.Input("seed", min=0, max=0xFFFFFFFF, control_after_generate=True, optional=True),
                comfy_io.String.Input(
                    "region_to_modify",
                    multiline=True,
                    optional=True,
                    tooltip="Plaintext description of the object / region to modify.",
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
        image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        prompt_text: str = "",
        negative_prompt: str = "",
        seed: int = 0,
        region_to_modify: str = "",
    ) -> comfy_io.NodeOutput:
        video_bytes_io = BytesIO()
        video.save_to(video_bytes_io, format=VideoContainer.MP4, codec=VideoCodec.H264)
        video_bytes_io.seek(0)
        pika_files = {
            "video": ("video.mp4", video_bytes_io, "video/mp4"),
        }
        if mask is not None:
            pika_files["modifyRegionMask"] = ("mask.png", tensor_to_bytesio(mask), "image/png")
        if image is not None:
            pika_files["image"] = ("image.png", tensor_to_bytesio(image), "image/png")

        pika_request_data = pika_defs.PikaBodyGeneratePikaswapsGeneratePikaswapsPost(
            promptText=prompt_text,
            negativePrompt=negative_prompt,
            seed=seed,
            modifyRegionRoi=region_to_modify if region_to_modify else None,
        )
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_PIKASWAPS,
                method=HttpMethod.POST,
                request_model=pika_defs.PikaBodyGeneratePikaswapsGeneratePikaswapsPost,
                response_model=pika_defs.PikaGenerateResponse,
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
                    "pikaffect", options=pika_defs.Pikaffect, default="Cake-ify"
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
                request_model=pika_defs.PikaBodyGeneratePikaffectsGeneratePikaffectsPost,
                response_model=pika_defs.PikaGenerateResponse,
            ),
            request=pika_defs.PikaBodyGeneratePikaffectsGeneratePikaffectsPost(
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


class PikaStartEndFrameNode(comfy_io.ComfyNode):
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
        validate_string(prompt_text, field_name="prompt_text", min_length=1)
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
                request_model=pika_defs.PikaBodyGenerate22KeyframeGenerate22PikaframesPost,
                response_model=pika_defs.PikaGenerateResponse,
            ),
            request=pika_defs.PikaBodyGenerate22KeyframeGenerate22PikaframesPost(
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
            PikaImageToVideo,
            PikaTextToVideoNode,
            PikaScenes,
            PikAdditionsNode,
            PikaSwapsNode,
            PikaffectsNode,
            PikaStartEndFrameNode,
        ]


async def comfy_entrypoint() -> PikaApiNodesExtension:
    return PikaApiNodesExtension()

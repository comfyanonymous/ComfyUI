"""
Pika x ComfyUI API Nodes

Pika API docs: https://pika-827374fb.mintlify.app/api-reference
"""
from __future__ import annotations

from io import BytesIO
import logging
from typing import Optional

import torch

from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO
from comfy_api.input_impl.video_types import VideoCodec, VideoContainer, VideoInput
from comfy_api_nodes.apis import pika_api as pika_defs
from comfy_api_nodes.util import (
    validate_string,
    download_url_to_video_output,
    tensor_to_bytesio,
    ApiEndpoint,
    sync_op,
    poll_op,
)


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
    task_id: str,
    cls: type[IO.ComfyNode],
) -> IO.NodeOutput:
    final_response: pika_defs.PikaVideoResponse = await poll_op(
        cls,
        ApiEndpoint(path=f"{PATH_VIDEO_GET}/{task_id}"),
        response_model=pika_defs.PikaVideoResponse,
        status_extractor=lambda response: (response.status.value if response.status else None),
        progress_extractor=lambda response: (response.progress if hasattr(response, "progress") else None),
        estimated_duration=60,
        max_poll_attempts=240,
    )
    if not final_response.url:
        error_msg = f"Pika task {task_id} succeeded but no video data found in response:\n{final_response}"
        logging.error(error_msg)
        raise Exception(error_msg)
    video_url = final_response.url
    logging.info("Pika task %s succeeded. Video URL: %s", task_id, video_url)
    return IO.NodeOutput(await download_url_to_video_output(video_url))


def get_base_inputs_types() -> list[IO.Input]:
    """Get the base required inputs types common to all Pika nodes."""
    return [
        IO.String.Input("prompt_text", multiline=True),
        IO.String.Input("negative_prompt", multiline=True),
        IO.Int.Input("seed", min=0, max=0xFFFFFFFF, control_after_generate=True),
        IO.Combo.Input("resolution", options=["1080p", "720p"], default="1080p"),
        IO.Combo.Input("duration", options=[5, 10], default=5),
    ]


class PikaImageToVideo(IO.ComfyNode):
    """Pika 2.2 Image to Video Node."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PikaImageToVideoNode2_2",
            display_name="Pika Image to Video",
            description="Sends an image and prompt to the Pika API v2.2 to generate a video.",
            category="api node/video/Pika",
            inputs=[
                IO.Image.Input("image", tooltip="The image to convert to video"),
                *get_base_inputs_types(),
            ],
            outputs=[IO.Video.Output()],
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
        image: torch.Tensor,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
    ) -> IO.NodeOutput:
        image_bytes_io = tensor_to_bytesio(image)
        pika_files = {"image": ("image.png", image_bytes_io, "image/png")}
        pika_request_data = pika_defs.PikaBodyGenerate22I2vGenerate22I2vPost(
            promptText=prompt_text,
            negativePrompt=negative_prompt,
            seed=seed,
            resolution=resolution,
            duration=duration,
        )
        initial_operation = await sync_op(
            cls,
            ApiEndpoint(path=PATH_IMAGE_TO_VIDEO, method="POST"),
            response_model=pika_defs.PikaGenerateResponse,
            data=pika_request_data,
            files=pika_files,
            content_type="multipart/form-data",
        )
        return await execute_task(initial_operation.video_id, cls)


class PikaTextToVideoNode(IO.ComfyNode):
    """Pika Text2Video v2.2 Node."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PikaTextToVideoNode2_2",
            display_name="Pika Text to Video",
            description="Sends a text prompt to the Pika API v2.2 to generate a video.",
            category="api node/video/Pika",
            inputs=[
                *get_base_inputs_types(),
                IO.Float.Input(
                    "aspect_ratio",
                    step=0.001,
                    min=0.4,
                    max=2.5,
                    default=1.7777777777777777,
                    tooltip="Aspect ratio (width / height)",
                )
            ],
            outputs=[IO.Video.Output()],
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
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
        aspect_ratio: float,
    ) -> IO.NodeOutput:
        initial_operation = await sync_op(
            cls,
            ApiEndpoint(path=PATH_TEXT_TO_VIDEO, method="POST"),
            response_model=pika_defs.PikaGenerateResponse,
            data=pika_defs.PikaBodyGenerate22T2vGenerate22T2vPost(
                promptText=prompt_text,
                negativePrompt=negative_prompt,
                seed=seed,
                resolution=resolution,
                duration=duration,
                aspectRatio=aspect_ratio,
            ),
            content_type="application/x-www-form-urlencoded",
        )
        return await execute_task(initial_operation.video_id, cls)


class PikaScenes(IO.ComfyNode):
    """PikaScenes v2.2 Node."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PikaScenesV2_2",
            display_name="Pika Scenes (Video Image Composition)",
            description="Combine your images to create a video with the objects in them. Upload multiple images as ingredients and generate a high-quality video that incorporates all of them.",
            category="api node/video/Pika",
            inputs=[
                *get_base_inputs_types(),
                IO.Combo.Input(
                    "ingredients_mode",
                    options=["creative", "precise"],
                    default="creative",
                ),
                IO.Float.Input(
                    "aspect_ratio",
                    step=0.001,
                    min=0.4,
                    max=2.5,
                    default=1.7777777777777777,
                    tooltip="Aspect ratio (width / height)",
                ),
                IO.Image.Input(
                    "image_ingredient_1",
                    optional=True,
                    tooltip="Image that will be used as ingredient to create a video.",
                ),
                IO.Image.Input(
                    "image_ingredient_2",
                    optional=True,
                    tooltip="Image that will be used as ingredient to create a video.",
                ),
                IO.Image.Input(
                    "image_ingredient_3",
                    optional=True,
                    tooltip="Image that will be used as ingredient to create a video.",
                ),
                IO.Image.Input(
                    "image_ingredient_4",
                    optional=True,
                    tooltip="Image that will be used as ingredient to create a video.",
                ),
                IO.Image.Input(
                    "image_ingredient_5",
                    optional=True,
                    tooltip="Image that will be used as ingredient to create a video.",
                ),
            ],
            outputs=[IO.Video.Output()],
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
    ) -> IO.NodeOutput:
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
        initial_operation = await sync_op(
            cls,
            ApiEndpoint(path=PATH_PIKASCENES, method="POST"),
            response_model=pika_defs.PikaGenerateResponse,
            data=pika_request_data,
            files=pika_files,
            content_type="multipart/form-data",
        )

        return await execute_task(initial_operation.video_id, cls)


class PikAdditionsNode(IO.ComfyNode):
    """Pika Pikadditions Node. Add an image into a video."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Pikadditions",
            display_name="Pikadditions (Video Object Insertion)",
            description="Add any object or image into your video. Upload a video and specify what you'd like to add to create a seamlessly integrated result.",
            category="api node/video/Pika",
            inputs=[
                IO.Video.Input("video", tooltip="The video to add an image to."),
                IO.Image.Input("image", tooltip="The image to add to the video."),
                IO.String.Input("prompt_text", multiline=True),
                IO.String.Input("negative_prompt", multiline=True),
                IO.Int.Input(
                    "seed",
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                ),
            ],
            outputs=[IO.Video.Output()],
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
        video: VideoInput,
        image: torch.Tensor,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
    ) -> IO.NodeOutput:
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
        initial_operation = await sync_op(
            cls,
            ApiEndpoint(path=PATH_PIKADDITIONS, method="POST"),
            response_model=pika_defs.PikaGenerateResponse,
            data=pika_request_data,
            files=pika_files,
            content_type="multipart/form-data",
        )

        return await execute_task(initial_operation.video_id, cls)


class PikaSwapsNode(IO.ComfyNode):
    """Pika Pikaswaps Node."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Pikaswaps",
            display_name="Pika Swaps (Video Object Replacement)",
            description="Swap out any object or region of your video with a new image or object. Define areas to replace either with a mask or coordinates.",
            category="api node/video/Pika",
            inputs=[
                IO.Video.Input("video", tooltip="The video to swap an object in."),
                IO.Image.Input(
                    "image",
                    tooltip="The image used to replace the masked object in the video.",
                    optional=True,
                ),
                IO.Mask.Input(
                    "mask",
                    tooltip="Use the mask to define areas in the video to replace.",
                    optional=True,
                ),
                IO.String.Input("prompt_text", multiline=True, optional=True),
                IO.String.Input("negative_prompt", multiline=True, optional=True),
                IO.Int.Input("seed", min=0, max=0xFFFFFFFF, control_after_generate=True, optional=True),
                IO.String.Input(
                    "region_to_modify",
                    multiline=True,
                    optional=True,
                    tooltip="Plaintext description of the object / region to modify.",
                ),
            ],
            outputs=[IO.Video.Output()],
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
        video: VideoInput,
        image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        prompt_text: str = "",
        negative_prompt: str = "",
        seed: int = 0,
        region_to_modify: str = "",
    ) -> IO.NodeOutput:
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
        initial_operation = await sync_op(
            cls,
            ApiEndpoint(path=PATH_PIKASWAPS, method="POST"),
            response_model=pika_defs.PikaGenerateResponse,
            data=pika_request_data,
            files=pika_files,
            content_type="multipart/form-data",
        )
        return await execute_task(initial_operation.video_id, cls)


class PikaffectsNode(IO.ComfyNode):
    """Pika Pikaffects Node."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Pikaffects",
            display_name="Pikaffects (Video Effects)",
            description="Generate a video with a specific Pikaffect. Supported Pikaffects: Cake-ify, Crumble, Crush, Decapitate, Deflate, Dissolve, Explode, Eye-pop, Inflate, Levitate, Melt, Peel, Poke, Squish, Ta-da, Tear",
            category="api node/video/Pika",
            inputs=[
                IO.Image.Input("image", tooltip="The reference image to apply the Pikaffect to."),
                IO.Combo.Input(
                    "pikaffect", options=pika_defs.Pikaffect, default="Cake-ify"
                ),
                IO.String.Input("prompt_text", multiline=True),
                IO.String.Input("negative_prompt", multiline=True),
                IO.Int.Input("seed", min=0, max=0xFFFFFFFF, control_after_generate=True),
            ],
            outputs=[IO.Video.Output()],
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
        image: torch.Tensor,
        pikaffect: str,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
    ) -> IO.NodeOutput:
        initial_operation = await sync_op(
            cls,
            ApiEndpoint(path=PATH_PIKAFFECTS, method="POST"),
            response_model=pika_defs.PikaGenerateResponse,
            data=pika_defs.PikaBodyGeneratePikaffectsGeneratePikaffectsPost(
                pikaffect=pikaffect,
                promptText=prompt_text,
                negativePrompt=negative_prompt,
                seed=seed,
            ),
            files={"image": ("image.png", tensor_to_bytesio(image), "image/png")},
            content_type="multipart/form-data",
        )
        return await execute_task(initial_operation.video_id, cls)


class PikaStartEndFrameNode(IO.ComfyNode):
    """PikaFrames v2.2 Node."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PikaStartEndFrameNode2_2",
            display_name="Pika Start and End Frame to Video",
            description="Generate a video by combining your first and last frame. Upload two images to define the start and end points, and let the AI create a smooth transition between them.",
            category="api node/video/Pika",
            inputs=[
                IO.Image.Input("image_start", tooltip="The first image to combine."),
                IO.Image.Input("image_end", tooltip="The last image to combine."),
                *get_base_inputs_types(),
            ],
            outputs=[IO.Video.Output()],
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
        image_start: torch.Tensor,
        image_end: torch.Tensor,
        prompt_text: str,
        negative_prompt: str,
        seed: int,
        resolution: str,
        duration: int,
    ) -> IO.NodeOutput:
        validate_string(prompt_text, field_name="prompt_text", min_length=1)
        pika_files = [
            ("keyFrames", ("image_start.png", tensor_to_bytesio(image_start), "image/png")),
            ("keyFrames", ("image_end.png", tensor_to_bytesio(image_end), "image/png")),
        ]
        initial_operation = await sync_op(
            cls,
            ApiEndpoint(path=PATH_PIKAFRAMES, method="POST"),
            response_model=pika_defs.PikaGenerateResponse,
            data=pika_defs.PikaBodyGenerate22KeyframeGenerate22PikaframesPost(
                promptText=prompt_text,
                negativePrompt=negative_prompt,
                seed=seed,
                resolution=resolution,
                duration=duration,
            ),
            files=pika_files,
            content_type="multipart/form-data",
        )
        return await execute_task(initial_operation.video_id, cls)


class PikaApiNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
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

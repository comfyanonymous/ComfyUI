"""Runway API Nodes

API Docs:
  - https://docs.dev.runwayml.com/api/#tag/Task-management/paths/~1v1~1tasks~1%7Bid%7D/delete

User Guides:
  - https://help.runwayml.com/hc/en-us/sections/30265301423635-Gen-3-Alpha
  - https://help.runwayml.com/hc/en-us/articles/37327109429011-Creating-with-Gen-4-Video
  - https://help.runwayml.com/hc/en-us/articles/33927968552339-Creating-with-Act-One-on-Gen-3-Alpha-and-Turbo
  - https://help.runwayml.com/hc/en-us/articles/34170748696595-Creating-with-Keyframes-on-Gen-3

"""

from typing import Union, Optional
from typing_extensions import override
from enum import Enum

import torch

from comfy_api_nodes.apis import (
    RunwayImageToVideoRequest,
    RunwayImageToVideoResponse,
    RunwayTaskStatusResponse as TaskStatusResponse,
    RunwayModelEnum as Model,
    RunwayDurationEnum as Duration,
    RunwayAspectRatioEnum as AspectRatio,
    RunwayPromptImageObject,
    RunwayPromptImageDetailedObject,
    RunwayTextToImageRequest,
    RunwayTextToImageResponse,
    Model4,
    ReferenceImage,
    RunwayTextToImageAspectRatioEnum,
)
from comfy_api_nodes.util import (
    image_tensor_pair_to_batch,
    validate_string,
    validate_image_dimensions,
    validate_image_aspect_ratio,
    upload_images_to_comfyapi,
    download_url_to_video_output,
    download_url_to_image_tensor,
    ApiEndpoint,
    sync_op,
    poll_op,
)
from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import ComfyExtension, IO

PATH_IMAGE_TO_VIDEO = "/proxy/runway/image_to_video"
PATH_TEXT_TO_IMAGE = "/proxy/runway/text_to_image"
PATH_GET_TASK_STATUS = "/proxy/runway/tasks"

AVERAGE_DURATION_I2V_SECONDS = 64
AVERAGE_DURATION_FLF_SECONDS = 256
AVERAGE_DURATION_T2I_SECONDS = 41


class RunwayApiError(Exception):
    """Base exception for Runway API errors."""

    pass


class RunwayGen4TurboAspectRatio(str, Enum):
    """Aspect ratios supported for Image to Video API when using gen4_turbo model."""

    field_1280_720 = "1280:720"
    field_720_1280 = "720:1280"
    field_1104_832 = "1104:832"
    field_832_1104 = "832:1104"
    field_960_960 = "960:960"
    field_1584_672 = "1584:672"


class RunwayGen3aAspectRatio(str, Enum):
    """Aspect ratios supported for Image to Video API when using gen3a_turbo model."""

    field_768_1280 = "768:1280"
    field_1280_768 = "1280:768"


def get_video_url_from_task_status(response: TaskStatusResponse) -> Union[str, None]:
    """Returns the video URL from the task status response if it exists."""
    if hasattr(response, "output") and len(response.output) > 0:
        return response.output[0]
    return None


def extract_progress_from_task_status(
    response: TaskStatusResponse,
) -> Union[float, None]:
    if hasattr(response, "progress") and response.progress is not None:
        return response.progress * 100
    return None


def get_image_url_from_task_status(response: TaskStatusResponse) -> Union[str, None]:
    """Returns the image URL from the task status response if it exists."""
    if hasattr(response, "output") and len(response.output) > 0:
        return response.output[0]
    return None


async def get_response(
    cls: type[IO.ComfyNode], task_id: str, estimated_duration: Optional[int] = None
) -> TaskStatusResponse:
    """Poll the task status until it is finished then get the response."""
    return await poll_op(
        cls,
        ApiEndpoint(path=f"{PATH_GET_TASK_STATUS}/{task_id}"),
        response_model=TaskStatusResponse,
        status_extractor=lambda r: r.status.value,
        estimated_duration=estimated_duration,
        progress_extractor=extract_progress_from_task_status,
    )


async def generate_video(
    cls: type[IO.ComfyNode],
    request: RunwayImageToVideoRequest,
    estimated_duration: Optional[int] = None,
) -> VideoFromFile:
    initial_response = await sync_op(
        cls,
        endpoint=ApiEndpoint(path=PATH_IMAGE_TO_VIDEO, method="POST"),
        response_model=RunwayImageToVideoResponse,
        data=request,
    )

    final_response = await get_response(cls, initial_response.id, estimated_duration)
    if not final_response.output:
        raise RunwayApiError("Runway task succeeded but no video data found in response.")

    video_url = get_video_url_from_task_status(final_response)
    return await download_url_to_video_output(video_url)


class RunwayImageToVideoNodeGen3a(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RunwayImageToVideoNodeGen3a",
            display_name="Runway Image to Video (Gen3a Turbo)",
            category="api node/video/Runway",
            description="Generate a video from a single starting frame using Gen3a Turbo model. "
            "Before diving in, review these best practices to ensure that "
            "your input selections will set your generation up for success: "
            "https://help.runwayml.com/hc/en-us/articles/33927968552339-Creating-with-Act-One-on-Gen-3-Alpha-and-Turbo.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the generation",
                ),
                IO.Image.Input(
                    "start_frame",
                    tooltip="Start frame to be used for the video",
                ),
                IO.Combo.Input(
                    "duration",
                    options=Duration,
                ),
                IO.Combo.Input(
                    "ratio",
                    options=RunwayGen3aAspectRatio,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Random seed for generation",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
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
        prompt: str,
        start_frame: torch.Tensor,
        duration: str,
        ratio: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1)
        validate_image_dimensions(start_frame, max_width=7999, max_height=7999)
        validate_image_aspect_ratio(start_frame, (1, 2), (2, 1))

        download_urls = await upload_images_to_comfyapi(
            cls,
            start_frame,
            max_images=1,
            mime_type="image/png",
        )

        return IO.NodeOutput(
            await generate_video(
                cls,
                RunwayImageToVideoRequest(
                    promptText=prompt,
                    seed=seed,
                    model=Model("gen3a_turbo"),
                    duration=Duration(duration),
                    ratio=AspectRatio(ratio),
                    promptImage=RunwayPromptImageObject(
                        root=[RunwayPromptImageDetailedObject(uri=str(download_urls[0]), position="first")]
                    ),
                ),
            )
        )


class RunwayImageToVideoNodeGen4(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RunwayImageToVideoNodeGen4",
            display_name="Runway Image to Video (Gen4 Turbo)",
            category="api node/video/Runway",
            description="Generate a video from a single starting frame using Gen4 Turbo model. "
            "Before diving in, review these best practices to ensure that "
            "your input selections will set your generation up for success: "
            "https://help.runwayml.com/hc/en-us/articles/37327109429011-Creating-with-Gen-4-Video.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the generation",
                ),
                IO.Image.Input(
                    "start_frame",
                    tooltip="Start frame to be used for the video",
                ),
                IO.Combo.Input(
                    "duration",
                    options=Duration,
                ),
                IO.Combo.Input(
                    "ratio",
                    options=RunwayGen4TurboAspectRatio,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Random seed for generation",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
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
        prompt: str,
        start_frame: torch.Tensor,
        duration: str,
        ratio: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1)
        validate_image_dimensions(start_frame, max_width=7999, max_height=7999)
        validate_image_aspect_ratio(start_frame, (1, 2), (2, 1))

        download_urls = await upload_images_to_comfyapi(
            cls,
            start_frame,
            max_images=1,
            mime_type="image/png",
        )

        return IO.NodeOutput(
            await generate_video(
                cls,
                RunwayImageToVideoRequest(
                    promptText=prompt,
                    seed=seed,
                    model=Model("gen4_turbo"),
                    duration=Duration(duration),
                    ratio=AspectRatio(ratio),
                    promptImage=RunwayPromptImageObject(
                        root=[RunwayPromptImageDetailedObject(uri=str(download_urls[0]), position="first")]
                    ),
                ),
                estimated_duration=AVERAGE_DURATION_FLF_SECONDS,
            )
        )


class RunwayFirstLastFrameNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RunwayFirstLastFrameNode",
            display_name="Runway First-Last-Frame to Video",
            category="api node/video/Runway",
            description="Upload first and last keyframes, draft a prompt, and generate a video. "
            "More complex transitions, such as cases where the Last frame is completely different "
            "from the First frame, may benefit from the longer 10s duration. "
            "This would give the generation more time to smoothly transition between the two inputs. "
            "Before diving in, review these best practices to ensure that your input selections "
            "will set your generation up for success: "
            "https://help.runwayml.com/hc/en-us/articles/34170748696595-Creating-with-Keyframes-on-Gen-3.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the generation",
                ),
                IO.Image.Input(
                    "start_frame",
                    tooltip="Start frame to be used for the video",
                ),
                IO.Image.Input(
                    "end_frame",
                    tooltip="End frame to be used for the video. Supported for gen3a_turbo only.",
                ),
                IO.Combo.Input(
                    "duration",
                    options=Duration,
                ),
                IO.Combo.Input(
                    "ratio",
                    options=RunwayGen3aAspectRatio,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Random seed for generation",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
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
        prompt: str,
        start_frame: torch.Tensor,
        end_frame: torch.Tensor,
        duration: str,
        ratio: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1)
        validate_image_dimensions(start_frame, max_width=7999, max_height=7999)
        validate_image_dimensions(end_frame, max_width=7999, max_height=7999)
        validate_image_aspect_ratio(start_frame, (1, 2), (2, 1))
        validate_image_aspect_ratio(end_frame, (1, 2), (2, 1))

        stacked_input_images = image_tensor_pair_to_batch(start_frame, end_frame)
        download_urls = await upload_images_to_comfyapi(
            cls,
            stacked_input_images,
            max_images=2,
            mime_type="image/png",
        )
        if len(download_urls) != 2:
            raise RunwayApiError("Failed to upload one or more images to comfy api.")

        return IO.NodeOutput(
            await generate_video(
                cls,
                RunwayImageToVideoRequest(
                    promptText=prompt,
                    seed=seed,
                    model=Model("gen3a_turbo"),
                    duration=Duration(duration),
                    ratio=AspectRatio(ratio),
                    promptImage=RunwayPromptImageObject(
                        root=[
                            RunwayPromptImageDetailedObject(uri=str(download_urls[0]), position="first"),
                            RunwayPromptImageDetailedObject(uri=str(download_urls[1]), position="last"),
                        ]
                    ),
                ),
                estimated_duration=AVERAGE_DURATION_FLF_SECONDS,
            )
        )


class RunwayTextToImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RunwayTextToImageNode",
            display_name="Runway Text to Image",
            category="api node/image/Runway",
            description="Generate an image from a text prompt using Runway's Gen 4 model. "
            "You can also include reference image to guide the generation.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the generation",
                ),
                IO.Combo.Input(
                    "ratio",
                    options=[model.value for model in RunwayTextToImageAspectRatioEnum],
                ),
                IO.Image.Input(
                    "reference_image",
                    tooltip="Optional reference image to guide the generation",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
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
        prompt: str,
        ratio: str,
        reference_image: Optional[torch.Tensor] = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1)

        # Prepare reference images if provided
        reference_images = None
        if reference_image is not None:
            validate_image_dimensions(reference_image, max_width=7999, max_height=7999)
            validate_image_aspect_ratio(reference_image, (1, 2), (2, 1))
            download_urls = await upload_images_to_comfyapi(
                cls,
                reference_image,
                max_images=1,
                mime_type="image/png",
            )
            reference_images = [ReferenceImage(uri=str(download_urls[0]))]

        initial_response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path=PATH_TEXT_TO_IMAGE, method="POST"),
            response_model=RunwayTextToImageResponse,
            data=RunwayTextToImageRequest(
                promptText=prompt,
                model=Model4.gen4_image,
                ratio=ratio,
                referenceImages=reference_images,
            ),
        )

        final_response = await get_response(
            cls,
            initial_response.id,
            estimated_duration=AVERAGE_DURATION_T2I_SECONDS,
        )
        if not final_response.output:
            raise RunwayApiError("Runway task succeeded but no image data found in response.")

        return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_task_status(final_response)))


class RunwayExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            RunwayFirstLastFrameNode,
            RunwayImageToVideoNodeGen3a,
            RunwayImageToVideoNodeGen4,
            RunwayTextToImageNode,
        ]


async def comfy_entrypoint() -> RunwayExtension:
    return RunwayExtension()

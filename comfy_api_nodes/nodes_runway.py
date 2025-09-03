"""Runway API Nodes

API Docs:
  - https://docs.dev.runwayml.com/api/#tag/Task-management/paths/~1v1~1tasks~1%7Bid%7D/delete

User Guides:
  - https://help.runwayml.com/hc/en-us/sections/30265301423635-Gen-3-Alpha
  - https://help.runwayml.com/hc/en-us/articles/37327109429011-Creating-with-Gen-4-Video
  - https://help.runwayml.com/hc/en-us/articles/33927968552339-Creating-with-Act-One-on-Gen-3-Alpha-and-Turbo
  - https://help.runwayml.com/hc/en-us/articles/34170748696595-Creating-with-Keyframes-on-Gen-3

"""

from typing import Union, Optional, Any
from typing_extensions import override
from enum import Enum

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
    RunwayTextToImageRequest,
    RunwayTextToImageResponse,
    Model4,
    ReferenceImage,
    RunwayTextToImageAspectRatioEnum,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import (
    upload_images_to_comfyapi,
    download_url_to_video_output,
    image_tensor_pair_to_batch,
    validate_string,
    download_url_to_image_tensor,
)
from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import ComfyExtension, io as comfy_io
from comfy_api_nodes.util.validation_utils import validate_image_dimensions, validate_image_aspect_ratio

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


async def poll_until_finished(
    auth_kwargs: dict[str, str],
    api_endpoint: ApiEndpoint[Any, TaskStatusResponse],
    estimated_duration: Optional[int] = None,
    node_id: Optional[str] = None,
) -> TaskStatusResponse:
    """Polls the Runway API endpoint until the task reaches a terminal state, then returns the response."""
    return await PollingOperation(
        poll_endpoint=api_endpoint,
        completed_statuses=[
            TaskStatus.SUCCEEDED.value,
        ],
        failed_statuses=[
            TaskStatus.FAILED.value,
            TaskStatus.CANCELLED.value,
        ],
        status_extractor=lambda response: response.status.value,
        auth_kwargs=auth_kwargs,
        result_url_extractor=get_video_url_from_task_status,
        estimated_duration=estimated_duration,
        node_id=node_id,
        progress_extractor=extract_progress_from_task_status,
    ).execute()


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
    task_id: str, auth_kwargs: dict[str, str], node_id: Optional[str] = None, estimated_duration: Optional[int] = None
) -> TaskStatusResponse:
    """Poll the task status until it is finished then get the response."""
    return await poll_until_finished(
        auth_kwargs,
        ApiEndpoint(
            path=f"{PATH_GET_TASK_STATUS}/{task_id}",
            method=HttpMethod.GET,
            request_model=EmptyRequest,
            response_model=TaskStatusResponse,
        ),
        estimated_duration=estimated_duration,
        node_id=node_id,
    )


async def generate_video(
    request: RunwayImageToVideoRequest,
    auth_kwargs: dict[str, str],
    node_id: Optional[str] = None,
    estimated_duration: Optional[int] = None,
) -> VideoFromFile:
    initial_operation = SynchronousOperation(
        endpoint=ApiEndpoint(
            path=PATH_IMAGE_TO_VIDEO,
            method=HttpMethod.POST,
            request_model=RunwayImageToVideoRequest,
            response_model=RunwayImageToVideoResponse,
        ),
        request=request,
        auth_kwargs=auth_kwargs,
    )

    initial_response = await initial_operation.execute()

    final_response = await get_response(initial_response.id, auth_kwargs, node_id, estimated_duration)
    if not final_response.output:
        raise RunwayApiError("Runway task succeeded but no video data found in response.")

    video_url = get_video_url_from_task_status(final_response)
    return await download_url_to_video_output(video_url)


class RunwayImageToVideoNodeGen3a(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="RunwayImageToVideoNodeGen3a",
            display_name="Runway Image to Video (Gen3a Turbo)",
            category="api node/video/Runway",
            description="Generate a video from a single starting frame using Gen3a Turbo model. "
                        "Before diving in, review these best practices to ensure that "
                        "your input selections will set your generation up for success: "
                        "https://help.runwayml.com/hc/en-us/articles/33927968552339-Creating-with-Act-One-on-Gen-3-Alpha-and-Turbo.",
            inputs=[
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the generation",
                ),
                comfy_io.Image.Input(
                    "start_frame",
                    tooltip="Start frame to be used for the video",
                ),
                comfy_io.Combo.Input(
                    "duration",
                    options=[model.value for model in Duration],
                ),
                comfy_io.Combo.Input(
                    "ratio",
                    options=[model.value for model in RunwayGen3aAspectRatio],
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    control_after_generate=True,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Random seed for generation",
                ),
            ],
            outputs=[
                comfy_io.Video.Output(),
            ],
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
        prompt: str,
        start_frame: torch.Tensor,
        duration: str,
        ratio: str,
        seed: int,
    ) -> comfy_io.NodeOutput:
        validate_string(prompt, min_length=1)
        validate_image_dimensions(start_frame, max_width=7999, max_height=7999)
        validate_image_aspect_ratio(start_frame, min_aspect_ratio=0.5, max_aspect_ratio=2.0)

        auth_kwargs = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }

        download_urls = await upload_images_to_comfyapi(
            start_frame,
            max_images=1,
            mime_type="image/png",
            auth_kwargs=auth_kwargs,
        )

        return comfy_io.NodeOutput(
            await generate_video(
                RunwayImageToVideoRequest(
                    promptText=prompt,
                    seed=seed,
                    model=Model("gen3a_turbo"),
                    duration=Duration(duration),
                    ratio=AspectRatio(ratio),
                    promptImage=RunwayPromptImageObject(
                        root=[
                            RunwayPromptImageDetailedObject(
                                uri=str(download_urls[0]), position="first"
                            )
                        ]
                    ),
                ),
                auth_kwargs=auth_kwargs,
                node_id=cls.hidden.unique_id,
            )
        )


class RunwayImageToVideoNodeGen4(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="RunwayImageToVideoNodeGen4",
            display_name="Runway Image to Video (Gen4 Turbo)",
            category="api node/video/Runway",
            description="Generate a video from a single starting frame using Gen4 Turbo model. "
                        "Before diving in, review these best practices to ensure that "
                        "your input selections will set your generation up for success: "
                        "https://help.runwayml.com/hc/en-us/articles/37327109429011-Creating-with-Gen-4-Video.",
            inputs=[
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the generation",
                ),
                comfy_io.Image.Input(
                    "start_frame",
                    tooltip="Start frame to be used for the video",
                ),
                comfy_io.Combo.Input(
                    "duration",
                    options=[model.value for model in Duration],
                ),
                comfy_io.Combo.Input(
                    "ratio",
                    options=[model.value for model in RunwayGen4TurboAspectRatio],
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    control_after_generate=True,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Random seed for generation",
                ),
            ],
            outputs=[
                comfy_io.Video.Output(),
            ],
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
        prompt: str,
        start_frame: torch.Tensor,
        duration: str,
        ratio: str,
        seed: int,
    ) -> comfy_io.NodeOutput:
        validate_string(prompt, min_length=1)
        validate_image_dimensions(start_frame, max_width=7999, max_height=7999)
        validate_image_aspect_ratio(start_frame, min_aspect_ratio=0.5, max_aspect_ratio=2.0)

        auth_kwargs = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }

        download_urls = await upload_images_to_comfyapi(
            start_frame,
            max_images=1,
            mime_type="image/png",
            auth_kwargs=auth_kwargs,
        )

        return comfy_io.NodeOutput(
            await generate_video(
                RunwayImageToVideoRequest(
                    promptText=prompt,
                    seed=seed,
                    model=Model("gen4_turbo"),
                    duration=Duration(duration),
                    ratio=AspectRatio(ratio),
                    promptImage=RunwayPromptImageObject(
                        root=[
                            RunwayPromptImageDetailedObject(
                                uri=str(download_urls[0]), position="first"
                            )
                        ]
                    ),
                ),
                auth_kwargs=auth_kwargs,
                node_id=cls.hidden.unique_id,
                estimated_duration=AVERAGE_DURATION_FLF_SECONDS,
            )
        )


class RunwayFirstLastFrameNode(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
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
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the generation",
                ),
                comfy_io.Image.Input(
                    "start_frame",
                    tooltip="Start frame to be used for the video",
                ),
                comfy_io.Image.Input(
                    "end_frame",
                    tooltip="End frame to be used for the video. Supported for gen3a_turbo only.",
                ),
                comfy_io.Combo.Input(
                    "duration",
                    options=[model.value for model in Duration],
                ),
                comfy_io.Combo.Input(
                    "ratio",
                    options=[model.value for model in RunwayGen3aAspectRatio],
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    control_after_generate=True,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Random seed for generation",
                ),
            ],
            outputs=[
                comfy_io.Video.Output(),
            ],
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
        prompt: str,
        start_frame: torch.Tensor,
        end_frame: torch.Tensor,
        duration: str,
        ratio: str,
        seed: int,
    ) -> comfy_io.NodeOutput:
        validate_string(prompt, min_length=1)
        validate_image_dimensions(start_frame, max_width=7999, max_height=7999)
        validate_image_dimensions(end_frame, max_width=7999, max_height=7999)
        validate_image_aspect_ratio(start_frame, min_aspect_ratio=0.5, max_aspect_ratio=2.0)
        validate_image_aspect_ratio(end_frame, min_aspect_ratio=0.5, max_aspect_ratio=2.0)

        auth_kwargs = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }

        stacked_input_images = image_tensor_pair_to_batch(start_frame, end_frame)
        download_urls = await upload_images_to_comfyapi(
            stacked_input_images,
            max_images=2,
            mime_type="image/png",
            auth_kwargs=auth_kwargs,
        )
        if len(download_urls) != 2:
            raise RunwayApiError("Failed to upload one or more images to comfy api.")

        return comfy_io.NodeOutput(
            await generate_video(
                RunwayImageToVideoRequest(
                    promptText=prompt,
                    seed=seed,
                    model=Model("gen3a_turbo"),
                    duration=Duration(duration),
                    ratio=AspectRatio(ratio),
                    promptImage=RunwayPromptImageObject(
                        root=[
                            RunwayPromptImageDetailedObject(
                                uri=str(download_urls[0]), position="first"
                            ),
                            RunwayPromptImageDetailedObject(
                                uri=str(download_urls[1]), position="last"
                            ),
                        ]
                    ),
                ),
                auth_kwargs=auth_kwargs,
                node_id=cls.hidden.unique_id,
                estimated_duration=AVERAGE_DURATION_FLF_SECONDS,
            )
        )


class RunwayTextToImageNode(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="RunwayTextToImageNode",
            display_name="Runway Text to Image",
            category="api node/image/Runway",
            description="Generate an image from a text prompt using Runway's Gen 4 model. "
                        "You can also include reference image to guide the generation.",
            inputs=[
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the generation",
                ),
                comfy_io.Combo.Input(
                    "ratio",
                    options=[model.value for model in RunwayTextToImageAspectRatioEnum],
                ),
                comfy_io.Image.Input(
                    "reference_image",
                    tooltip="Optional reference image to guide the generation",
                    optional=True,
                ),
            ],
            outputs=[
                comfy_io.Image.Output(),
            ],
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
        prompt: str,
        ratio: str,
        reference_image: Optional[torch.Tensor] = None,
    ) -> comfy_io.NodeOutput:
        validate_string(prompt, min_length=1)

        auth_kwargs = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }

        # Prepare reference images if provided
        reference_images = None
        if reference_image is not None:
            validate_image_dimensions(reference_image, max_width=7999, max_height=7999)
            validate_image_aspect_ratio(reference_image, min_aspect_ratio=0.5, max_aspect_ratio=2.0)
            download_urls = await upload_images_to_comfyapi(
                reference_image,
                max_images=1,
                mime_type="image/png",
                auth_kwargs=auth_kwargs,
            )
            reference_images = [ReferenceImage(uri=str(download_urls[0]))]

        request = RunwayTextToImageRequest(
            promptText=prompt,
            model=Model4.gen4_image,
            ratio=ratio,
            referenceImages=reference_images,
        )

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=PATH_TEXT_TO_IMAGE,
                method=HttpMethod.POST,
                request_model=RunwayTextToImageRequest,
                response_model=RunwayTextToImageResponse,
            ),
            request=request,
            auth_kwargs=auth_kwargs,
        )

        initial_response = await initial_operation.execute()

        # Poll for completion
        final_response = await get_response(
            initial_response.id,
            auth_kwargs=auth_kwargs,
            node_id=cls.hidden.unique_id,
            estimated_duration=AVERAGE_DURATION_T2I_SECONDS,
        )
        if not final_response.output:
            raise RunwayApiError("Runway task succeeded but no image data found in response.")

        return comfy_io.NodeOutput(await download_url_to_image_tensor(get_image_url_from_task_status(final_response)))


class RunwayExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[comfy_io.ComfyNode]]:
        return [
            RunwayFirstLastFrameNode,
            RunwayImageToVideoNodeGen3a,
            RunwayImageToVideoNodeGen4,
            RunwayTextToImageNode,
        ]

async def comfy_entrypoint() -> RunwayExtension:
    return RunwayExtension()

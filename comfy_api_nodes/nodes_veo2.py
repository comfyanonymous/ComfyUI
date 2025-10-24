import base64
from io import BytesIO

from typing_extensions import override

from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis.veo_api import (
    VeoGenVidPollRequest,
    VeoGenVidPollResponse,
    VeoGenVidRequest,
    VeoGenVidResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    poll_op,
    sync_op,
    tensor_to_base64_string,
)

AVERAGE_DURATION_VIDEO_GEN = 32
MODELS_MAP = {
    "veo-2.0-generate-001": "veo-2.0-generate-001",
    "veo-3.1-generate": "veo-3.1-generate-preview",
    "veo-3.1-fast-generate": "veo-3.1-fast-generate-preview",
    "veo-3.0-generate-001": "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001": "veo-3.0-fast-generate-001",
}


class VeoVideoGenerationNode(IO.ComfyNode):
    """
    Generates videos from text prompts using Google's Veo API.

    This node can create videos from text descriptions and optional image inputs,
    with control over parameters like aspect ratio, duration, and more.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VeoVideoGenerationNode",
            display_name="Google Veo 2 Video Generation",
            category="api node/video/Veo",
            description="Generates videos from text prompts using Google's Veo 2 API",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text description of the video",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "9:16"],
                    default="16:9",
                    tooltip="Aspect ratio of the output video",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative text prompt to guide what to avoid in the video",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration_seconds",
                    default=5,
                    min=5,
                    max=8,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "enhance_prompt",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance",
                    optional=True,
                ),
                IO.Combo.Input(
                    "person_generation",
                    options=["ALLOW", "BLOCK"],
                    default="ALLOW",
                    tooltip="Whether to allow generating people in the video",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Optional reference image to guide video generation",
                    optional=True,
                ),
                IO.Combo.Input(
                    "model",
                    options=["veo-2.0-generate-001"],
                    default="veo-2.0-generate-001",
                    tooltip="Veo 2 model to use for video generation",
                    optional=True,
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
        prompt,
        aspect_ratio="16:9",
        negative_prompt="",
        duration_seconds=5,
        enhance_prompt=True,
        person_generation="ALLOW",
        seed=0,
        image=None,
        model="veo-2.0-generate-001",
        generate_audio=False,
    ):
        model = MODELS_MAP[model]
        # Prepare the instances for the request
        instances = []

        instance = {"prompt": prompt}

        # Add image if provided
        if image is not None:
            image_base64 = tensor_to_base64_string(image)
            if image_base64:
                instance["image"] = {"bytesBase64Encoded": image_base64, "mimeType": "image/png"}

        instances.append(instance)

        # Create parameters dictionary
        parameters = {
            "aspectRatio": aspect_ratio,
            "personGeneration": person_generation,
            "durationSeconds": duration_seconds,
            "enhancePrompt": enhance_prompt,
        }

        # Add optional parameters if provided
        if negative_prompt:
            parameters["negativePrompt"] = negative_prompt
        if seed > 0:
            parameters["seed"] = seed
        # Only add generateAudio for Veo 3 models
        if model.find("veo-2.0") == -1:
            parameters["generateAudio"] = generate_audio

        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/veo/{model}/generate", method="POST"),
            response_model=VeoGenVidResponse,
            data=VeoGenVidRequest(
                instances=instances,
                parameters=parameters,
            ),
        )

        def status_extractor(response):
            # Only return "completed" if the operation is done, regardless of success or failure
            # We'll check for errors after polling completes
            return "completed" if response.done else "pending"

        poll_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/veo/{model}/poll", method="POST"),
            response_model=VeoGenVidPollResponse,
            status_extractor=status_extractor,
            data=VeoGenVidPollRequest(
                operationName=initial_response.name,
            ),
            poll_interval=5.0,
            estimated_duration=AVERAGE_DURATION_VIDEO_GEN,
        )

        # Now check for errors in the final response
        # Check for error in poll response
        if poll_response.error:
            raise Exception(f"Veo API error: {poll_response.error.message} (code: {poll_response.error.code})")

        # Check for RAI filtered content
        if (
            hasattr(poll_response.response, "raiMediaFilteredCount")
            and poll_response.response.raiMediaFilteredCount > 0
        ):

            # Extract reason message if available
            if (
                hasattr(poll_response.response, "raiMediaFilteredReasons")
                and poll_response.response.raiMediaFilteredReasons
            ):
                reason = poll_response.response.raiMediaFilteredReasons[0]
                error_message = f"Content filtered by Google's Responsible AI practices: {reason} ({poll_response.response.raiMediaFilteredCount} videos filtered.)"
            else:
                error_message = f"Content filtered by Google's Responsible AI practices ({poll_response.response.raiMediaFilteredCount} videos filtered.)"

            raise Exception(error_message)

        # Extract video data
        if (
            poll_response.response
            and hasattr(poll_response.response, "videos")
            and poll_response.response.videos
            and len(poll_response.response.videos) > 0
        ):
            video = poll_response.response.videos[0]

            # Check if video is provided as base64 or URL
            if hasattr(video, "bytesBase64Encoded") and video.bytesBase64Encoded:
                return IO.NodeOutput(VideoFromFile(BytesIO(base64.b64decode(video.bytesBase64Encoded))))

            if hasattr(video, "gcsUri") and video.gcsUri:
                return IO.NodeOutput(await download_url_to_video_output(video.gcsUri))

            raise Exception("Video returned but no data or URL was provided")
        raise Exception("Video generation completed but no video was returned")


class Veo3VideoGenerationNode(VeoVideoGenerationNode):
    """
    Generates videos from text prompts using Google's Veo 3 API.

    Supported models:
    - veo-3.0-generate-001
    - veo-3.0-fast-generate-001

    This node extends the base Veo node with Veo 3 specific features including
    audio generation and fixed 8-second duration.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Veo3VideoGenerationNode",
            display_name="Google Veo 3 Video Generation",
            category="api node/video/Veo",
            description="Generates videos from text prompts using Google's Veo 3 API",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text description of the video",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "9:16"],
                    default="16:9",
                    tooltip="Aspect ratio of the output video",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative text prompt to guide what to avoid in the video",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration_seconds",
                    default=8,
                    min=8,
                    max=8,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds (Veo 3 only supports 8 seconds)",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "enhance_prompt",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance",
                    optional=True,
                ),
                IO.Combo.Input(
                    "person_generation",
                    options=["ALLOW", "BLOCK"],
                    default="ALLOW",
                    tooltip="Whether to allow generating people in the video",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Optional reference image to guide video generation",
                    optional=True,
                ),
                IO.Combo.Input(
                    "model",
                    options=[
                        "veo-3.1-generate",
                        "veo-3.1-fast-generate",
                        "veo-3.0-generate-001",
                        "veo-3.0-fast-generate-001",
                    ],
                    default="veo-3.0-generate-001",
                    tooltip="Veo 3 model to use for video generation",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    tooltip="Generate audio for the video. Supported by all Veo 3 models.",
                    optional=True,
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


class VeoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            VeoVideoGenerationNode,
            Veo3VideoGenerationNode,
        ]


async def comfy_entrypoint() -> VeoExtension:
    return VeoExtension()

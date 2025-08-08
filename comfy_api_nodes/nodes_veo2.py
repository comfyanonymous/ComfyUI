import io
import logging
import base64
import aiohttp
import torch
from typing import Optional

from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api_nodes.apis import (
    VeoGenVidRequest,
    VeoGenVidResponse,
    VeoGenVidPollRequest,
    VeoGenVidPollResponse
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
)

from comfy_api_nodes.apinode_utils import (
    downscale_image_tensor,
    tensor_to_base64_string
)

AVERAGE_DURATION_VIDEO_GEN = 32

def convert_image_to_base64(image: torch.Tensor):
    if image is None:
        return None

    scaled_image = downscale_image_tensor(image, total_pixels=2048*2048)
    return tensor_to_base64_string(scaled_image)


def get_video_url_from_response(poll_response: VeoGenVidPollResponse) -> Optional[str]:
    if (
        poll_response.response
        and hasattr(poll_response.response, "videos")
        and poll_response.response.videos
        and len(poll_response.response.videos) > 0
    ):
        video = poll_response.response.videos[0]
    else:
        return None
    if hasattr(video, "gcsUri") and video.gcsUri:
        return str(video.gcsUri)
    return None


class VeoVideoGenerationNode(ComfyNodeABC):
    """
    Generates videos from text prompts using Google's Veo API.

    This node can create videos from text descriptions and optional image inputs,
    with control over parameters like aspect ratio, duration, and more.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text description of the video",
                    },
                ),
                "aspect_ratio": (
                    IO.COMBO,
                    {
                        "options": ["16:9", "9:16"],
                        "default": "16:9",
                        "tooltip": "Aspect ratio of the output video",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Negative text prompt to guide what to avoid in the video",
                    },
                ),
                "duration_seconds": (
                    IO.INT,
                    {
                        "default": 5,
                        "min": 5,
                        "max": 8,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Duration of the output video in seconds",
                    },
                ),
                "enhance_prompt": (
                    IO.BOOLEAN,
                    {
                        "default": True,
                        "tooltip": "Whether to enhance the prompt with AI assistance",
                    }
                ),
                "person_generation": (
                    IO.COMBO,
                    {
                        "options": ["ALLOW", "BLOCK"],
                        "default": "ALLOW",
                        "tooltip": "Whether to allow generating people in the video",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "step": 1,
                        "display": "number",
                        "control_after_generate": True,
                        "tooltip": "Seed for video generation (0 for random)",
                    },
                ),
                "image": (IO.IMAGE, {
                    "default": None,
                    "tooltip": "Optional reference image to guide video generation",
                }),
                "model": (
                    IO.COMBO,
                    {
                        "options": ["veo-2.0-generate-001"],
                        "default": "veo-2.0-generate-001",
                        "tooltip": "Veo 2 model to use for video generation",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.VIDEO,)
    FUNCTION = "generate_video"
    CATEGORY = "api node/video/Veo"
    DESCRIPTION = "Generates videos from text prompts using Google's Veo 2 API"
    API_NODE = True

    async def generate_video(
        self,
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
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        # Prepare the instances for the request
        instances = []

        instance = {
            "prompt": prompt
        }

        # Add image if provided
        if image is not None:
            image_base64 = convert_image_to_base64(image)
            if image_base64:
                instance["image"] = {
                    "bytesBase64Encoded": image_base64,
                    "mimeType": "image/png"
                }

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
        if "veo-3.0" in model:
            parameters["generateAudio"] = generate_audio

        # Initial request to start video generation
        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=f"/proxy/veo/{model}/generate",
                method=HttpMethod.POST,
                request_model=VeoGenVidRequest,
                response_model=VeoGenVidResponse
            ),
            request=VeoGenVidRequest(
                instances=instances,
                parameters=parameters
            ),
            auth_kwargs=kwargs,
        )

        initial_response = await initial_operation.execute()
        operation_name = initial_response.name

        logging.info(f"Veo generation started with operation name: {operation_name}")

        # Define status extractor function
        def status_extractor(response):
            # Only return "completed" if the operation is done, regardless of success or failure
            # We'll check for errors after polling completes
            return "completed" if response.done else "pending"

        # Define progress extractor function
        def progress_extractor(response):
            # Could be enhanced if the API provides progress information
            return None

        # Define the polling operation
        poll_operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/veo/{model}/poll",
                method=HttpMethod.POST,
                request_model=VeoGenVidPollRequest,
                response_model=VeoGenVidPollResponse
            ),
            completed_statuses=["completed"],
            failed_statuses=[],  # No failed statuses, we'll handle errors after polling
            status_extractor=status_extractor,
            progress_extractor=progress_extractor,
            request=VeoGenVidPollRequest(
                operationName=operation_name
            ),
            auth_kwargs=kwargs,
            poll_interval=5.0,
            result_url_extractor=get_video_url_from_response,
            node_id=unique_id,
            estimated_duration=AVERAGE_DURATION_VIDEO_GEN,
        )

        # Execute the polling operation
        poll_response = await poll_operation.execute()

        # Now check for errors in the final response
        # Check for error in poll response
        if hasattr(poll_response, 'error') and poll_response.error:
            error_message = f"Veo API error: {poll_response.error.message} (code: {poll_response.error.code})"
            logging.error(error_message)
            raise Exception(error_message)

        # Check for RAI filtered content
        if (hasattr(poll_response.response, 'raiMediaFilteredCount') and
            poll_response.response.raiMediaFilteredCount > 0):

            # Extract reason message if available
            if (hasattr(poll_response.response, 'raiMediaFilteredReasons') and
                poll_response.response.raiMediaFilteredReasons):
                reason = poll_response.response.raiMediaFilteredReasons[0]
                error_message = f"Content filtered by Google's Responsible AI practices: {reason} ({poll_response.response.raiMediaFilteredCount} videos filtered.)"
            else:
                error_message = f"Content filtered by Google's Responsible AI practices ({poll_response.response.raiMediaFilteredCount} videos filtered.)"

            logging.error(error_message)
            raise Exception(error_message)

        # Extract video data
        if poll_response.response and hasattr(poll_response.response, 'videos') and poll_response.response.videos and len(poll_response.response.videos) > 0:
            video = poll_response.response.videos[0]

            # Check if video is provided as base64 or URL
            if hasattr(video, 'bytesBase64Encoded') and video.bytesBase64Encoded:
                # Decode base64 string to bytes
                video_data = base64.b64decode(video.bytesBase64Encoded)
            elif hasattr(video, 'gcsUri') and video.gcsUri:
                # Download from URL
                async with aiohttp.ClientSession() as session:
                    async with session.get(video.gcsUri) as video_response:
                        video_data = await video_response.content.read()
            else:
                raise Exception("Video returned but no data or URL was provided")
        else:
            raise Exception("Video generation completed but no video was returned")

        if not video_data:
            raise Exception("No video data was returned")

        logging.info("Video generation completed successfully")

        # Convert video data to BytesIO object
        video_io = io.BytesIO(video_data)

        # Return VideoFromFile object
        return (VideoFromFile(video_io),)


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
    def INPUT_TYPES(s):
        parent_input = super().INPUT_TYPES()

        # Update model options for Veo 3
        parent_input["optional"]["model"] = (
            IO.COMBO,
            {
                "options": ["veo-3.0-generate-001", "veo-3.0-fast-generate-001"],
                "default": "veo-3.0-generate-001",
                "tooltip": "Veo 3 model to use for video generation",
            },
        )

        # Add generateAudio parameter
        parent_input["optional"]["generate_audio"] = (
            IO.BOOLEAN,
            {
                "default": False,
                "tooltip": "Generate audio for the video. Supported by all Veo 3 models.",
            }
        )

        # Update duration constraints for Veo 3 (only 8 seconds supported)
        parent_input["optional"]["duration_seconds"] = (
            IO.INT,
            {
                "default": 8,
                "min": 8,
                "max": 8,
                "step": 1,
                "display": "number",
                "tooltip": "Duration of the output video in seconds (Veo 3 only supports 8 seconds)",
            },
        )

        return parent_input


# Register the nodes
NODE_CLASS_MAPPINGS = {
    "VeoVideoGenerationNode": VeoVideoGenerationNode,
    "Veo3VideoGenerationNode": Veo3VideoGenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VeoVideoGenerationNode": "Google Veo 2 Video Generation",
    "Veo3VideoGenerationNode": "Google Veo 3 Video Generation",
}

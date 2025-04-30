import io
import logging
import base64
import requests
import math
import torch
import numpy as np
from PIL import Image

from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from comfy.utils import common_upscale
from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api_nodes.apis import (
    Veo2GenVidRequest,
    Veo2GenVidResponse,
    Veo2GenVidPollRequest,
    Veo2GenVidPollResponse
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
)

def downscale_input(image, total_pixels=1536*1024):
    samples = image.movedim(-1,1)
    # Downscaling input images to roughly the same size as the outputs
    total = int(total_pixels)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1,-1)
    return s

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
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
            },
        }

    RETURN_TYPES = (IO.VIDEO,)
    FUNCTION = "generate_video"
    CATEGORY = "api node/video/Veo"
    DESCRIPTION = "Generates videos from text prompts using Google's Veo API"
    API_NODE = True

    def _convert_image_to_base64(self, image: torch.Tensor):
        if image is None:
            return None

        scaled_image = downscale_input(image, total_pixels=2048*2048)

        # Remove batch dimension if present
        if len(scaled_image.shape) > 3:
            scaled_image = scaled_image[0]

        # Convert to numpy array and then to PIL Image
        image_np = (scaled_image.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(image_np)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def generate_video(
        self,
        prompt,
        aspect_ratio="16:9",
        negative_prompt="",
        duration_seconds=5,
        enhance_prompt=True,
        person_generation="ALLOW",
        seed=0,
        image=None,
        auth_token=None,
    ):
        # Prepare the instances for the request
        instances = []

        instance = {
            "prompt": prompt
        }

        # Add image if provided
        if image is not None:
            image_base64 = self._convert_image_to_base64(image)
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

        # Initial request to start video generation
        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/veo/generate",
                method=HttpMethod.POST,
                request_model=Veo2GenVidRequest,
                response_model=Veo2GenVidResponse
            ),
            request=Veo2GenVidRequest(
                instances=instances,
                parameters=parameters
            ),
            auth_token=auth_token
        )

        initial_response = initial_operation.execute()
        operation_name = initial_response.name

        logging.info(f"Veo generation started with operation name: {operation_name}")

        # Poll until operation is complete
        video_data = None
        while True:
            poll_operation = SynchronousOperation(
                endpoint=ApiEndpoint(
                    path="/proxy/veo/poll",
                    method=HttpMethod.POST,
                    request_model=Veo2GenVidPollRequest,
                    response_model=Veo2GenVidPollResponse
                ),
                request=Veo2GenVidPollRequest(
                    operationName=operation_name
                ),
                auth_token=auth_token
            )

            poll_response = poll_operation.execute()

            # Check for error in poll response
            if hasattr(poll_response, 'error') and poll_response.error:
                error_message = f"Veo API error: {poll_response.error.message} (code: {poll_response.error.code})"
                logging.error(error_message)
                raise Exception(error_message)

            if poll_response.done:
                # Check for RAI filtered content
                if (hasattr(poll_response.response, 'raiMediaFilteredCount') and
                    poll_response.response.raiMediaFilteredCount > 0):

                    # Extract reason message if available
                    if (hasattr(poll_response.response, 'raiMediaFilteredReasons') and
                        poll_response.response.raiMediaFilteredReasons):
                        reason = poll_response.response.raiMediaFilteredReasons[0]
                        error_message = f"Content filtered by Google's Responsible AI practices: {reason} ({poll_response.response.raiMediaFilteredCount} videos filtered.)"

                    logging.error(error_message)
                    raise Exception(error_message)

                # Process successful response
                if poll_response.response and hasattr(poll_response.response, 'videos') and poll_response.response.videos and len(poll_response.response.videos) > 0:
                    video = poll_response.response.videos[0]

                    # Check if video is provided as base64 or URL
                    if hasattr(video, 'bytesBase64Encoded') and video.bytesBase64Encoded:
                        # Decode base64 string to bytes
                        video_data = base64.b64decode(video.bytesBase64Encoded)
                        break
                    elif hasattr(video, 'gcsUri') and video.gcsUri:
                        # Download from URL
                        video_url = video.gcsUri
                        video_response = requests.get(video_url)
                        video_data = video_response.content
                        break
                    else:
                        raise Exception("Video returned but no data or URL was provided")
                else:
                    raise Exception("Video generation completed but no video was returned")

            # Wait before polling again
            import time
            time.sleep(5)

        if not video_data:
            raise Exception("No video data was returned")

        logging.info("Video generation completed successfully")

        # Convert video data to BytesIO object
        video_io = io.BytesIO(video_data)

        # Return VideoFromFile object
        return (VideoFromFile(video_io),)


# Register the node
NODE_CLASS_MAPPINGS = {
    "VeoVideoGenerationNode": VeoVideoGenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VeoVideoGenerationNode": "Google Veo2 Video Generation",
}

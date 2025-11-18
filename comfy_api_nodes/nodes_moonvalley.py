import logging
from typing import Optional

import torch
from typing_extensions import override

from comfy_api.input import VideoInput
from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis import (
    MoonvalleyPromptResponse,
    MoonvalleyTextToVideoInferenceParams,
    MoonvalleyTextToVideoRequest,
    MoonvalleyVideoToVideoInferenceParams,
    MoonvalleyVideoToVideoRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    poll_op,
    sync_op,
    trim_video,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_container_format_is_mp4,
    validate_image_dimensions,
    validate_string,
)

API_UPLOADS_ENDPOINT = "/proxy/moonvalley/uploads"
API_PROMPTS_ENDPOINT = "/proxy/moonvalley/prompts"
API_VIDEO2VIDEO_ENDPOINT = "/proxy/moonvalley/prompts/video-to-video"
API_TXT2VIDEO_ENDPOINT = "/proxy/moonvalley/prompts/text-to-video"
API_IMG2VIDEO_ENDPOINT = "/proxy/moonvalley/prompts/image-to-video"

MIN_WIDTH = 300
MIN_HEIGHT = 300

MAX_WIDTH = 10000
MAX_HEIGHT = 10000

MIN_VID_WIDTH = 300
MIN_VID_HEIGHT = 300

MAX_VID_WIDTH = 10000
MAX_VID_HEIGHT = 10000

MAX_VIDEO_SIZE = 1024 * 1024 * 1024  # 1 GB max for in-memory video processing

MOONVALLEY_MAREY_MAX_PROMPT_LENGTH = 5000


def is_valid_task_creation_response(response: MoonvalleyPromptResponse) -> bool:
    """Verifies that the initial response contains a task ID."""
    return bool(response.id)


def validate_task_creation_response(response) -> None:
    if not is_valid_task_creation_response(response):
        error_msg = f"Moonvalley Marey API: Initial request failed. Code: {response.code}, Message: {response.message}, Data: {response}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)


def validate_video_to_video_input(video: VideoInput) -> VideoInput:
    """
    Validates and processes video input for Moonvalley Video-to-Video generation.

    Args:
        video: Input video to validate

    Returns:
        Validated and potentially trimmed video

    Raises:
        ValueError: If video doesn't meet requirements
        MoonvalleyApiError: If video duration is too short
    """
    width, height = _get_video_dimensions(video)
    _validate_video_dimensions(width, height)
    validate_container_format_is_mp4(video)

    return _validate_and_trim_duration(video)


def _get_video_dimensions(video: VideoInput) -> tuple[int, int]:
    """Extracts video dimensions with error handling."""
    try:
        return video.get_dimensions()
    except Exception as e:
        logging.error("Error getting dimensions of video: %s", e)
        raise ValueError(f"Cannot get video dimensions: {e}") from e


def _validate_video_dimensions(width: int, height: int) -> None:
    """Validates video dimensions meet Moonvalley V2V requirements."""
    supported_resolutions = {
        (1920, 1080),
        (1080, 1920),
        (1152, 1152),
        (1536, 1152),
        (1152, 1536),
    }

    if (width, height) not in supported_resolutions:
        supported_list = ", ".join([f"{w}x{h}" for w, h in sorted(supported_resolutions)])
        raise ValueError(f"Resolution {width}x{height} not supported. Supported: {supported_list}")


def _validate_and_trim_duration(video: VideoInput) -> VideoInput:
    """Validates video duration and trims to 5 seconds if needed."""
    duration = video.get_duration()
    _validate_minimum_duration(duration)
    return _trim_if_too_long(video, duration)


def _validate_minimum_duration(duration: float) -> None:
    """Ensures video is at least 5 seconds long."""
    if duration < 5:
        raise ValueError("Input video must be at least 5 seconds long.")


def _trim_if_too_long(video: VideoInput, duration: float) -> VideoInput:
    """Trims video to 5 seconds if longer."""
    if duration > 5:
        return trim_video(video, 5)
    return video


def parse_width_height_from_res(resolution: str):
    # Accepts a string like "16:9 (1920 x 1080)" and returns width, height as a dict
    res_map = {
        "16:9 (1920 x 1080)": {"width": 1920, "height": 1080},
        "9:16 (1080 x 1920)": {"width": 1080, "height": 1920},
        "1:1 (1152 x 1152)": {"width": 1152, "height": 1152},
        "4:3 (1536 x 1152)": {"width": 1536, "height": 1152},
        "3:4 (1152 x 1536)": {"width": 1152, "height": 1536},
        # "21:9 (2560 x 1080)": {"width": 2560, "height": 1080},
    }
    return res_map.get(resolution, {"width": 1920, "height": 1080})


def parse_control_parameter(value):
    control_map = {
        "Motion Transfer": "motion_control",
        "Canny": "canny_control",
        "Pose Transfer": "pose_control",
        "Depth": "depth_control",
    }
    return control_map.get(value, control_map["Motion Transfer"])


async def get_response(cls: type[IO.ComfyNode], task_id: str) -> MoonvalleyPromptResponse:
    return await poll_op(
        cls,
        ApiEndpoint(path=f"{API_PROMPTS_ENDPOINT}/{task_id}"),
        response_model=MoonvalleyPromptResponse,
        status_extractor=lambda r: (r.status if r and r.status else None),
        poll_interval=16.0,
        max_poll_attempts=240,
    )


class MoonvalleyImg2VideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="MoonvalleyImg2VideoNode",
            display_name="Moonvalley Marey Image to Video",
            category="api node/video/Moonvalley Marey",
            description="Moonvalley Marey Image to Video Node",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="The reference image used to generate the video",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="<synthetic> <scene cut> gopro, bright, contrast, static, overexposed, vignette, "
                    "artifacts, still, noise, texture, scanlines, videogame, 360 camera, VR, transition, "
                    "flare, saturation, distorted, warped, wide angle, saturated, vibrant, glowing, "
                    "cross dissolve, cheesy, ugly hands, mutated hands, mutant, disfigured, extra fingers, "
                    "blown out, horrible, blurry, worst quality, bad, dissolve, melt, fade in, fade out, "
                    "wobbly, weird, low quality, plastic, stock footage, video camera, boring",
                    tooltip="Negative prompt text",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=[
                        "16:9 (1920 x 1080)",
                        "9:16 (1080 x 1920)",
                        "1:1 (1152 x 1152)",
                        "4:3 (1536 x 1152)",
                        "3:4 (1152 x 1536)",
                        # "21:9 (2560 x 1080)",
                    ],
                    default="16:9 (1920 x 1080)",
                    tooltip="Resolution of the output video",
                ),
                IO.Float.Input(
                    "prompt_adherence",
                    default=4.5,
                    min=1.0,
                    max=20.0,
                    step=1.0,
                    tooltip="Guidance scale for generation control",
                ),
                IO.Int.Input(
                    "seed",
                    default=9,
                    min=0,
                    max=4294967295,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Random seed value",
                    control_after_generate=True,
                ),
                IO.Int.Input(
                    "steps",
                    default=33,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Number of denoising steps",
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
        image: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        resolution: str,
        prompt_adherence: float,
        seed: int,
        steps: int,
    ) -> IO.NodeOutput:
        validate_image_dimensions(image, min_width=300, min_height=300, max_height=MAX_HEIGHT, max_width=MAX_WIDTH)
        validate_string(prompt, min_length=1, max_length=MOONVALLEY_MAREY_MAX_PROMPT_LENGTH)
        validate_string(negative_prompt, field_name="negative_prompt", max_length=MOONVALLEY_MAREY_MAX_PROMPT_LENGTH)
        width_height = parse_width_height_from_res(resolution)

        inference_params = MoonvalleyTextToVideoInferenceParams(
            negative_prompt=negative_prompt,
            steps=steps,
            seed=seed,
            guidance_scale=prompt_adherence,
            width=width_height["width"],
            height=width_height["height"],
            use_negative_prompts=True,
        )

        # Get MIME type from tensor - assuming PNG format for image tensors
        mime_type = "image/png"
        image_url = (await upload_images_to_comfyapi(cls, image, max_images=1, mime_type=mime_type))[0]
        task_creation_response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path=API_IMG2VIDEO_ENDPOINT, method="POST"),
            response_model=MoonvalleyPromptResponse,
            data=MoonvalleyTextToVideoRequest(
                image_url=image_url, prompt_text=prompt, inference_params=inference_params
            ),
        )
        validate_task_creation_response(task_creation_response)
        final_response = await get_response(cls, task_creation_response.id)
        video = await download_url_to_video_output(final_response.output_url)
        return IO.NodeOutput(video)


class MoonvalleyVideo2VideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="MoonvalleyVideo2VideoNode",
            display_name="Moonvalley Marey Video to Video",
            category="api node/video/Moonvalley Marey",
            description="",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Describes the video to generate",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="<synthetic> <scene cut> gopro, bright, contrast, static, overexposed, vignette, "
                    "artifacts, still, noise, texture, scanlines, videogame, 360 camera, VR, transition, "
                    "flare, saturation, distorted, warped, wide angle, saturated, vibrant, glowing, "
                    "cross dissolve, cheesy, ugly hands, mutated hands, mutant, disfigured, extra fingers, "
                    "blown out, horrible, blurry, worst quality, bad, dissolve, melt, fade in, fade out, "
                    "wobbly, weird, low quality, plastic, stock footage, video camera, boring",
                    tooltip="Negative prompt text",
                ),
                IO.Int.Input(
                    "seed",
                    default=9,
                    min=0,
                    max=4294967295,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Random seed value",
                    control_after_generate=False,
                ),
                IO.Video.Input(
                    "video",
                    tooltip="The reference video used to generate the output video. Must be at least 5 seconds long. "
                    "Videos longer than 5s will be automatically trimmed. Only MP4 format supported.",
                ),
                IO.Combo.Input(
                    "control_type",
                    options=["Motion Transfer", "Pose Transfer"],
                    default="Motion Transfer",
                    optional=True,
                ),
                IO.Int.Input(
                    "motion_intensity",
                    default=100,
                    min=0,
                    max=100,
                    step=1,
                    tooltip="Only used if control_type is 'Motion Transfer'",
                    optional=True,
                ),
                IO.Int.Input(
                    "steps",
                    default=33,
                    min=1,
                    max=100,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Number of inference steps",
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
        prompt: str,
        negative_prompt: str,
        seed: int,
        video: Optional[VideoInput] = None,
        control_type: str = "Motion Transfer",
        motion_intensity: Optional[int] = 100,
        steps=33,
        prompt_adherence=4.5,
    ) -> IO.NodeOutput:
        validated_video = validate_video_to_video_input(video)
        video_url = await upload_video_to_comfyapi(cls, validated_video)
        validate_string(prompt, min_length=1, max_length=MOONVALLEY_MAREY_MAX_PROMPT_LENGTH)
        validate_string(negative_prompt, field_name="negative_prompt", max_length=MOONVALLEY_MAREY_MAX_PROMPT_LENGTH)

        # Only include motion_intensity for Motion Transfer
        control_params = {}
        if control_type == "Motion Transfer" and motion_intensity is not None:
            control_params["motion_intensity"] = motion_intensity

        inference_params = MoonvalleyVideoToVideoInferenceParams(
            negative_prompt=negative_prompt,
            seed=seed,
            control_params=control_params,
            steps=steps,
            guidance_scale=prompt_adherence,
        )

        task_creation_response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path=API_VIDEO2VIDEO_ENDPOINT, method="POST"),
            response_model=MoonvalleyPromptResponse,
            data=MoonvalleyVideoToVideoRequest(
                control_type=parse_control_parameter(control_type),
                video_url=video_url,
                prompt_text=prompt,
                inference_params=inference_params,
            ),
        )
        validate_task_creation_response(task_creation_response)
        final_response = await get_response(cls, task_creation_response.id)
        return IO.NodeOutput(await download_url_to_video_output(final_response.output_url))


class MoonvalleyTxt2VideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="MoonvalleyTxt2VideoNode",
            display_name="Moonvalley Marey Text to Video",
            category="api node/video/Moonvalley Marey",
            description="",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="<synthetic> <scene cut> gopro, bright, contrast, static, overexposed, vignette, "
                    "artifacts, still, noise, texture, scanlines, videogame, 360 camera, VR, transition, "
                    "flare, saturation, distorted, warped, wide angle, saturated, vibrant, glowing, "
                    "cross dissolve, cheesy, ugly hands, mutated hands, mutant, disfigured, extra fingers, "
                    "blown out, horrible, blurry, worst quality, bad, dissolve, melt, fade in, fade out, "
                    "wobbly, weird, low quality, plastic, stock footage, video camera, boring",
                    tooltip="Negative prompt text",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=[
                        "16:9 (1920 x 1080)",
                        "9:16 (1080 x 1920)",
                        "1:1 (1152 x 1152)",
                        "4:3 (1536 x 1152)",
                        "3:4 (1152 x 1536)",
                        "21:9 (2560 x 1080)",
                    ],
                    default="16:9 (1920 x 1080)",
                    tooltip="Resolution of the output video",
                ),
                IO.Float.Input(
                    "prompt_adherence",
                    default=4.0,
                    min=1.0,
                    max=20.0,
                    step=1.0,
                    tooltip="Guidance scale for generation control",
                ),
                IO.Int.Input(
                    "seed",
                    default=9,
                    min=0,
                    max=4294967295,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Random seed value",
                ),
                IO.Int.Input(
                    "steps",
                    default=33,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Inference steps",
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
        prompt: str,
        negative_prompt: str,
        resolution: str,
        prompt_adherence: float,
        seed: int,
        steps: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=MOONVALLEY_MAREY_MAX_PROMPT_LENGTH)
        validate_string(negative_prompt, field_name="negative_prompt", max_length=MOONVALLEY_MAREY_MAX_PROMPT_LENGTH)
        width_height = parse_width_height_from_res(resolution)

        inference_params = MoonvalleyTextToVideoInferenceParams(
            negative_prompt=negative_prompt,
            steps=steps,
            seed=seed,
            guidance_scale=prompt_adherence,
            num_frames=128,
            width=width_height["width"],
            height=width_height["height"],
        )

        task_creation_response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path=API_TXT2VIDEO_ENDPOINT, method="POST"),
            response_model=MoonvalleyPromptResponse,
            data=MoonvalleyTextToVideoRequest(prompt_text=prompt, inference_params=inference_params),
        )
        validate_task_creation_response(task_creation_response)
        final_response = await get_response(cls, task_creation_response.id)
        return IO.NodeOutput(await download_url_to_video_output(final_response.output_url))


class MoonvalleyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            MoonvalleyImg2VideoNode,
            MoonvalleyTxt2VideoNode,
            MoonvalleyVideo2VideoNode,
        ]


async def comfy_entrypoint() -> MoonvalleyExtension:
    return MoonvalleyExtension()

import logging
from typing import Any, Callable, Optional, TypeVar
import torch
from typing_extensions import override
from comfy_api_nodes.util.validation_utils import (
    get_image_dimensions,
    validate_image_dimensions,
)


from comfy_api_nodes.apis import (
    MoonvalleyTextToVideoRequest,
    MoonvalleyTextToVideoInferenceParams,
    MoonvalleyVideoToVideoInferenceParams,
    MoonvalleyVideoToVideoRequest,
    MoonvalleyPromptResponse,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import (
    download_url_to_video_output,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
)

from comfy_api.input import VideoInput
from comfy_api.latest import ComfyExtension, InputImpl, io as comfy_io
import av
import io

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
R = TypeVar("R")


class MoonvalleyApiError(Exception):
    """Base exception for Moonvalley API errors."""

    pass


def is_valid_task_creation_response(response: MoonvalleyPromptResponse) -> bool:
    """Verifies that the initial response contains a task ID."""
    return bool(response.id)


def validate_task_creation_response(response) -> None:
    if not is_valid_task_creation_response(response):
        error_msg = f"Moonvalley Marey API: Initial request failed. Code: {response.code}, Message: {response.message}, Data: {response}"
        logging.error(error_msg)
        raise MoonvalleyApiError(error_msg)


def get_video_from_response(response):
    video = response.output_url
    logging.info(
        "Moonvalley Marey API: Task %s succeeded. Video URL: %s", response.id, video
    )
    return video


def get_video_url_from_response(response) -> Optional[str]:
    """Returns the first video url from the Moonvalley video generation task result.
    Will not raise an error if the response is not valid.
    """
    if response:
        return str(get_video_from_response(response))
    else:
        return None


async def poll_until_finished(
    auth_kwargs: dict[str, str],
    api_endpoint: ApiEndpoint[Any, R],
    result_url_extractor: Optional[Callable[[R], str]] = None,
    node_id: Optional[str] = None,
) -> R:
    """Polls the Moonvalley API endpoint until the task reaches a terminal state, then returns the response."""
    return await PollingOperation(
        poll_endpoint=api_endpoint,
        completed_statuses=[
            "completed",
        ],
        max_poll_attempts=240,  # 64 minutes with 16s interval
        poll_interval=16.0,
        failed_statuses=["error"],
        status_extractor=lambda response: (
            response.status if response and response.status else None
        ),
        auth_kwargs=auth_kwargs,
        result_url_extractor=result_url_extractor,
        node_id=node_id,
    ).execute()


def validate_prompts(
    prompt: str, negative_prompt: str, max_length=MOONVALLEY_MAREY_MAX_PROMPT_LENGTH
):
    """Verifies that the prompt isn't empty and that neither prompt is too long."""
    if not prompt:
        raise ValueError("Positive prompt is empty")
    if len(prompt) > max_length:
        raise ValueError(f"Positive prompt is too long: {len(prompt)} characters")
    if negative_prompt and len(negative_prompt) > max_length:
        raise ValueError(
            f"Negative prompt is too long: {len(negative_prompt)} characters"
        )
    return True


def validate_input_media(width, height, with_frame_conditioning, num_frames_in=None):
    # inference validation
    # T = num_frames
    # in all cases, the following must be true: T divisible by 16 and H,W by 8. in addition...
    # with image conditioning: H*W must be divisible by 8192
    # without image conditioning: T divisible by 32
    if num_frames_in and not num_frames_in % 16 == 0:
        return False, ("The input video total frame count must be divisible by 16!")

    if height % 8 != 0 or width % 8 != 0:
        return False, (
            f"Height ({height}) and width ({width}) must be " "divisible by 8"
        )

    if with_frame_conditioning:
        if (height * width) % 8192 != 0:
            return False, (
                f"Height * width ({height * width}) must be "
                "divisible by 8192 for frame conditioning"
            )
    else:
        if num_frames_in and not num_frames_in % 32 == 0:
            return False, ("The input video total frame count must be divisible by 32!")


def validate_input_image(
    image: torch.Tensor, with_frame_conditioning: bool = False
) -> None:
    """
    Validates the input image adheres to the expectations of the API:
    - The image resolution should not be less than 300*300px
    - The aspect ratio of the image should be between 1:2.5 ~ 2.5:1

    """
    height, width = get_image_dimensions(image)
    validate_input_media(width, height, with_frame_conditioning)
    validate_image_dimensions(
        image, min_width=300, min_height=300, max_height=MAX_HEIGHT, max_width=MAX_WIDTH
    )


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
    _validate_container_format(video)

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
        supported_list = ", ".join(
            [f"{w}x{h}" for w, h in sorted(supported_resolutions)]
        )
        raise ValueError(
            f"Resolution {width}x{height} not supported. Supported: {supported_list}"
        )


def _validate_container_format(video: VideoInput) -> None:
    """Validates video container format is MP4."""
    container_format = video.get_container_format()
    if container_format not in ["mp4", "mov,mp4,m4a,3gp,3g2,mj2"]:
        raise ValueError(
            f"Only MP4 container format supported. Got: {container_format}"
        )


def _validate_and_trim_duration(video: VideoInput) -> VideoInput:
    """Validates video duration and trims to 5 seconds if needed."""
    duration = video.get_duration()
    _validate_minimum_duration(duration)
    return _trim_if_too_long(video, duration)


def _validate_minimum_duration(duration: float) -> None:
    """Ensures video is at least 5 seconds long."""
    if duration < 5:
        raise MoonvalleyApiError("Input video must be at least 5 seconds long.")


def _trim_if_too_long(video: VideoInput, duration: float) -> VideoInput:
    """Trims video to 5 seconds if longer."""
    if duration > 5:
        return trim_video(video, 5)
    return video


def trim_video(video: VideoInput, duration_sec: float) -> VideoInput:
    """
    Returns a new VideoInput object trimmed from the beginning to the specified duration,
    using av to avoid loading entire video into memory.

    Args:
        video: Input video to trim
        duration_sec: Duration in seconds to keep from the beginning

    Returns:
        VideoFromFile object that owns the output buffer
    """
    output_buffer = io.BytesIO()

    input_container = None
    output_container = None

    try:
        # Get the stream source - this avoids loading entire video into memory
        # when the source is already a file path
        input_source = video.get_stream_source()

        # Open containers
        input_container = av.open(input_source, mode="r")
        output_container = av.open(output_buffer, mode="w", format="mp4")

        # Set up output streams for re-encoding
        video_stream = None
        audio_stream = None

        for stream in input_container.streams:
            logging.info(f"Found stream: type={stream.type}, class={type(stream)}")
            if isinstance(stream, av.VideoStream):
                # Create output video stream with same parameters
                video_stream = output_container.add_stream(
                    "h264", rate=stream.average_rate
                )
                video_stream.width = stream.width
                video_stream.height = stream.height
                video_stream.pix_fmt = "yuv420p"
                logging.info(
                    f"Added video stream: {stream.width}x{stream.height} @ {stream.average_rate}fps"
                )
            elif isinstance(stream, av.AudioStream):
                # Create output audio stream with same parameters
                audio_stream = output_container.add_stream(
                    "aac", rate=stream.sample_rate
                )
                audio_stream.sample_rate = stream.sample_rate
                audio_stream.layout = stream.layout
                logging.info(
                    f"Added audio stream: {stream.sample_rate}Hz, {stream.channels} channels"
                )

        # Calculate target frame count that's divisible by 16
        fps = input_container.streams.video[0].average_rate
        estimated_frames = int(duration_sec * fps)
        target_frames = (
            estimated_frames // 16
        ) * 16  # Round down to nearest multiple of 16

        if target_frames == 0:
            raise ValueError("Video too short: need at least 16 frames for Moonvalley")

        frame_count = 0
        audio_frame_count = 0

        # Decode and re-encode video frames
        if video_stream:
            for frame in input_container.decode(video=0):
                if frame_count >= target_frames:
                    break

                # Re-encode frame
                for packet in video_stream.encode(frame):
                    output_container.mux(packet)
                frame_count += 1

            # Flush encoder
            for packet in video_stream.encode():
                output_container.mux(packet)

            logging.info(
                f"Encoded {frame_count} video frames (target: {target_frames})"
            )

        # Decode and re-encode audio frames
        if audio_stream:
            input_container.seek(0)  # Reset to beginning for audio
            for frame in input_container.decode(audio=0):
                if frame.time >= duration_sec:
                    break

                # Re-encode frame
                for packet in audio_stream.encode(frame):
                    output_container.mux(packet)
                audio_frame_count += 1

            # Flush encoder
            for packet in audio_stream.encode():
                output_container.mux(packet)

            logging.info(f"Encoded {audio_frame_count} audio frames")

        # Close containers
        output_container.close()
        input_container.close()

        # Return as VideoFromFile using the buffer
        output_buffer.seek(0)
        return InputImpl.VideoFromFile(output_buffer)

    except Exception as e:
        # Clean up on error
        if input_container is not None:
            input_container.close()
        if output_container is not None:
            output_container.close()
        raise RuntimeError(f"Failed to trim video: {str(e)}") from e


def parse_width_height_from_res(resolution: str):
    # Accepts a string like "16:9 (1920 x 1080)" and returns width, height as a dict
    res_map = {
        "16:9 (1920 x 1080)": {"width": 1920, "height": 1080},
        "9:16 (1080 x 1920)": {"width": 1080, "height": 1920},
        "1:1 (1152 x 1152)": {"width": 1152, "height": 1152},
        "4:3 (1536 x 1152)": {"width": 1536, "height": 1152},
        "3:4 (1152 x 1536)": {"width": 1152, "height": 1536},
        "21:9 (2560 x 1080)": {"width": 2560, "height": 1080},
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


async def get_response(
    task_id: str, auth_kwargs: dict[str, str], node_id: Optional[str] = None
) -> MoonvalleyPromptResponse:
    return await poll_until_finished(
        auth_kwargs,
        ApiEndpoint(
            path=f"{API_PROMPTS_ENDPOINT}/{task_id}",
            method=HttpMethod.GET,
            request_model=EmptyRequest,
            response_model=MoonvalleyPromptResponse,
        ),
        result_url_extractor=get_video_url_from_response,
        node_id=node_id,
    )


class MoonvalleyImg2VideoNode(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="MoonvalleyImg2VideoNode",
            display_name="Moonvalley Marey Image to Video",
            category="api node/video/Moonvalley Marey",
            description="Moonvalley Marey Image to Video Node",
            inputs=[
                comfy_io.Image.Input(
                    "image",
                    tooltip="The reference image used to generate the video",
                ),
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                ),
                comfy_io.String.Input(
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
                comfy_io.Combo.Input(
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
                comfy_io.Float.Input(
                    "prompt_adherence",
                    default=10.0,
                    min=1.0,
                    max=20.0,
                    step=1.0,
                    tooltip="Guidance scale for generation control",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=9,
                    min=0,
                    max=4294967295,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Random seed value",
                ),
                comfy_io.Int.Input(
                    "steps",
                    default=100,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Number of denoising steps",
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
        image: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        resolution: str,
        prompt_adherence: float,
        seed: int,
        steps: int,
    ) -> comfy_io.NodeOutput:
        validate_input_image(image, True)
        validate_prompts(prompt, negative_prompt, MOONVALLEY_MAREY_MAX_PROMPT_LENGTH)
        width_height = parse_width_height_from_res(resolution)

        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }

        inference_params = MoonvalleyTextToVideoInferenceParams(
            negative_prompt=negative_prompt,
            steps=steps,
            seed=seed,
            guidance_scale=prompt_adherence,
            num_frames=128,
            width=width_height["width"],
            height=width_height["height"],
            use_negative_prompts=True,
        )
        """Upload image to comfy backend to have a URL available for further processing"""
        # Get MIME type from tensor - assuming PNG format for image tensors
        mime_type = "image/png"

        image_url = (
            await upload_images_to_comfyapi(
                image, max_images=1, auth_kwargs=auth, mime_type=mime_type
            )
        )[0]

        request = MoonvalleyTextToVideoRequest(
            image_url=image_url, prompt_text=prompt, inference_params=inference_params
        )
        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=API_IMG2VIDEO_ENDPOINT,
                method=HttpMethod.POST,
                request_model=MoonvalleyTextToVideoRequest,
                response_model=MoonvalleyPromptResponse,
            ),
            request=request,
            auth_kwargs=auth,
        )
        task_creation_response = await initial_operation.execute()
        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.id

        final_response = await get_response(
            task_id, auth_kwargs=auth, node_id=cls.hidden.unique_id
        )
        video = await download_url_to_video_output(final_response.output_url)
        return comfy_io.NodeOutput(video)


class MoonvalleyVideo2VideoNode(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="MoonvalleyVideo2VideoNode",
            display_name="Moonvalley Marey Video to Video",
            category="api node/video/Moonvalley Marey",
            description="",
            inputs=[
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Describes the video to generate",
                ),
                comfy_io.String.Input(
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
                comfy_io.Int.Input(
                    "seed",
                    default=9,
                    min=0,
                    max=4294967295,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Random seed value",
                    control_after_generate=False,
                ),
                comfy_io.Video.Input(
                    "video",
                    tooltip="The reference video used to generate the output video. Must be at least 5 seconds long. "
                            "Videos longer than 5s will be automatically trimmed. Only MP4 format supported.",
                ),
                comfy_io.Combo.Input(
                    "control_type",
                    options=["Motion Transfer", "Pose Transfer"],
                    default="Motion Transfer",
                    optional=True,
                ),
                comfy_io.Int.Input(
                    "motion_intensity",
                    default=100,
                    min=0,
                    max=100,
                    step=1,
                    tooltip="Only used if control_type is 'Motion Transfer'",
                    optional=True,
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
        prompt: str,
        negative_prompt: str,
        seed: int,
        video: Optional[VideoInput] = None,
        control_type: str = "Motion Transfer",
        motion_intensity: Optional[int] = 100,
    ) -> comfy_io.NodeOutput:
        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }

        validated_video = validate_video_to_video_input(video)
        video_url = await upload_video_to_comfyapi(validated_video, auth_kwargs=auth)

        """Validate prompts and inference input"""
        validate_prompts(prompt, negative_prompt)

        # Only include motion_intensity for Motion Transfer
        control_params = {}
        if control_type == "Motion Transfer" and motion_intensity is not None:
            control_params["motion_intensity"] = motion_intensity

        inference_params = MoonvalleyVideoToVideoInferenceParams(
            negative_prompt=negative_prompt,
            seed=seed,
            control_params=control_params,
        )

        control = parse_control_parameter(control_type)

        request = MoonvalleyVideoToVideoRequest(
            control_type=control,
            video_url=video_url,
            prompt_text=prompt,
            inference_params=inference_params,
        )

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=API_VIDEO2VIDEO_ENDPOINT,
                method=HttpMethod.POST,
                request_model=MoonvalleyVideoToVideoRequest,
                response_model=MoonvalleyPromptResponse,
            ),
            request=request,
            auth_kwargs=auth,
        )
        task_creation_response = await initial_operation.execute()
        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.id

        final_response = await get_response(
            task_id, auth_kwargs=auth, node_id=cls.hidden.unique_id
        )

        video = await download_url_to_video_output(final_response.output_url)
        return comfy_io.NodeOutput(video)


class MoonvalleyTxt2VideoNode(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="MoonvalleyTxt2VideoNode",
            display_name="Moonvalley Marey Text to Video",
            category="api node/video/Moonvalley Marey",
            description="",
            inputs=[
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                ),
                comfy_io.String.Input(
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
                comfy_io.Combo.Input(
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
                comfy_io.Float.Input(
                    "prompt_adherence",
                    default=10.0,
                    min=1.0,
                    max=20.0,
                    step=1.0,
                    tooltip="Guidance scale for generation control",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=9,
                    min=0,
                    max=4294967295,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Random seed value",
                ),
                comfy_io.Int.Input(
                    "steps",
                    default=100,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Inference steps",
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
        prompt: str,
        negative_prompt: str,
        resolution: str,
        prompt_adherence: float,
        seed: int,
        steps: int,
    ) -> comfy_io.NodeOutput:
        validate_prompts(prompt, negative_prompt, MOONVALLEY_MAREY_MAX_PROMPT_LENGTH)
        width_height = parse_width_height_from_res(resolution)

        auth = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }

        inference_params = MoonvalleyTextToVideoInferenceParams(
            negative_prompt=negative_prompt,
            steps=steps,
            seed=seed,
            guidance_scale=prompt_adherence,
            num_frames=128,
            width=width_height["width"],
            height=width_height["height"],
        )
        request = MoonvalleyTextToVideoRequest(
            prompt_text=prompt, inference_params=inference_params
        )

        init_op = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=API_TXT2VIDEO_ENDPOINT,
                method=HttpMethod.POST,
                request_model=MoonvalleyTextToVideoRequest,
                response_model=MoonvalleyPromptResponse,
            ),
            request=request,
            auth_kwargs=auth,
        )
        task_creation_response = await init_op.execute()
        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.id

        final_response = await get_response(
            task_id, auth_kwargs=auth, node_id=cls.hidden.unique_id
        )

        video = await download_url_to_video_output(final_response.output_url)
        return comfy_io.NodeOutput(video)


class MoonvalleyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[comfy_io.ComfyNode]]:
        return [
            MoonvalleyImg2VideoNode,
            MoonvalleyTxt2VideoNode,
            MoonvalleyVideo2VideoNode,
        ]


async def comfy_entrypoint() -> MoonvalleyExtension:
    return MoonvalleyExtension()

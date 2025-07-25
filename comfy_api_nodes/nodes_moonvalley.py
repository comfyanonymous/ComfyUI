import logging
from typing import Any, Callable, Optional, TypeVar
import random
import torch
from comfy_api_nodes.util.validation_utils import (
    get_image_dimensions,
    validate_image_dimensions,
    validate_video_dimensions,
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
from comfy_api_nodes.mapper_utils import model_field_to_node_input

from comfy_api.input.video_types import VideoInput
from comfy.comfy_types.node_typing import IO
from comfy_api.input_impl import VideoFromFile
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


def poll_until_finished(
    auth_kwargs: dict[str, str],
    api_endpoint: ApiEndpoint[Any, R],
    result_url_extractor: Optional[Callable[[R], str]] = None,
    node_id: Optional[str] = None,
) -> R:
    """Polls the Moonvalley API endpoint until the task reaches a terminal state, then returns the response."""
    return PollingOperation(
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


def validate_input_video(
    video: VideoInput, num_frames_out: int, with_frame_conditioning: bool = False
):
    try:
        width, height = video.get_dimensions()
    except Exception as e:
        logging.error("Error getting dimensions of video: %s", e)
        raise ValueError(f"Cannot get video dimensions: {e}") from e

    validate_input_media(width, height, with_frame_conditioning)
    validate_video_dimensions(
        video,
        min_width=MIN_VID_WIDTH,
        min_height=MIN_VID_HEIGHT,
        max_width=MAX_VID_WIDTH,
        max_height=MAX_VID_HEIGHT,
    )

    trimmed_video = validate_input_video_length(video, num_frames_out)
    return trimmed_video


def validate_input_video_length(video: VideoInput, num_frames: int):

    if video.get_duration() > 60:
        raise MoonvalleyApiError(
            "Input Video lenth should be less than 1min. Please trim."
        )

    if num_frames == 128:
        if video.get_duration() < 5:
            raise MoonvalleyApiError(
                "Input Video length is less than 5s. Please use a video longer than or equal to 5s."
            )
        if video.get_duration() > 5:
            #    trim video to 5s
            video = trim_video(video, 5)
    if num_frames == 256:
        if video.get_duration() < 10:
            raise MoonvalleyApiError(
                "Input Video length is less than 10s. Please use a video longer than or equal to 10s."
            )
        if video.get_duration() > 10:
            # trim video to 10s
            video = trim_video(video, 10)
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

        # Calculate target frame count that's divisible by 32
        fps = input_container.streams.video[0].average_rate
        estimated_frames = int(duration_sec * fps)
        target_frames = (
            estimated_frames // 32
        ) * 32  # Round down to nearest multiple of 32

        if target_frames == 0:
            raise ValueError("Video too short: need at least 32 frames for Moonvalley")

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
        return VideoFromFile(output_buffer)

    except Exception as e:
        # Clean up on error
        if input_container is not None:
            input_container.close()
        if output_container is not None:
            output_container.close()
        raise RuntimeError(f"Failed to trim video: {str(e)}") from e


# --- BaseMoonvalleyVideoNode ---
class BaseMoonvalleyVideoNode:
    def parseWidthHeightFromRes(self, resolution: str):
        # Accepts a string like "16:9 (1920 x 1080)" and returns width, height as a dict
        res_map = {
            "16:9 (1920 x 1080)": {"width": 1920, "height": 1080},
            "9:16 (1080 x 1920)": {"width": 1080, "height": 1920},
            "1:1 (1152 x 1152)": {"width": 1152, "height": 1152},
            "4:3 (1440 x 1080)": {"width": 1440, "height": 1080},
            "3:4 (1080 x 1440)": {"width": 1080, "height": 1440},
            "21:9 (2560 x 1080)": {"width": 2560, "height": 1080},
        }
        if resolution in res_map:
            return res_map[resolution]
        else:
            # Default to 1920x1080 if unknown
            return {"width": 1920, "height": 1080}

    def parseControlParameter(self, value):
        control_map = {
            "Motion Transfer": "motion_control",
            "Canny": "canny_control",
            "Pose Transfer": "pose_control",
            "Depth": "depth_control",
        }
        if value in control_map:
            return control_map[value]
        else:
            return control_map["Motion Transfer"]

    def get_response(
        self, task_id: str, auth_kwargs: dict[str, str], node_id: Optional[str] = None
    ) -> MoonvalleyPromptResponse:
        return poll_until_finished(
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": model_field_to_node_input(
                    IO.STRING,
                    MoonvalleyTextToVideoRequest,
                    "prompt_text",
                    multiline=True,
                ),
                "negative_prompt": model_field_to_node_input(
                    IO.STRING,
                    MoonvalleyTextToVideoInferenceParams,
                    "negative_prompt",
                    multiline=True,
                    default="low-poly, flat shader, bad rigging, stiff animation, uncanny eyes, low-quality textures, looping glitch, cheap effect, overbloom, bloom spam, default lighting, game asset, stiff face, ugly specular, AI artifacts",
                ),
                "resolution": (
                    IO.COMBO,
                    {
                        "options": [
                            "16:9 (1920 x 1080)",
                            "9:16 (1080 x 1920)",
                            "1:1 (1152 x 1152)",
                            "4:3 (1440 x 1080)",
                            "3:4 (1080 x 1440)",
                            "21:9 (2560 x 1080)",
                        ],
                        "default": "16:9 (1920 x 1080)",
                        "tooltip": "Resolution of the output video",
                    },
                ),
                # "length": (IO.COMBO,{"options":['5s','10s'], "default": '5s'}),
                "prompt_adherence": model_field_to_node_input(
                    IO.FLOAT,
                    MoonvalleyTextToVideoInferenceParams,
                    "guidance_scale",
                    default=7.0,
                    step=1,
                    min=1,
                    max=20,
                ),
                "seed": model_field_to_node_input(
                    IO.INT,
                    MoonvalleyTextToVideoInferenceParams,
                    "seed",
                    default=random.randint(0, 2**32 - 1),
                    min=0,
                    max=4294967295,
                    step=1,
                    display="number",
                    tooltip="Random seed value",
                    control_after_generate=True,
                ),
                "steps": model_field_to_node_input(
                    IO.INT,
                    MoonvalleyTextToVideoInferenceParams,
                    "steps",
                    default=100,
                    min=1,
                    max=100,
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "image": model_field_to_node_input(
                    IO.IMAGE,
                    MoonvalleyTextToVideoRequest,
                    "image_url",
                    tooltip="The reference image used to generate the video",
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "api node/video/Moonvalley Marey"
    API_NODE = True

    def generate(self, **kwargs):
        return None


# --- MoonvalleyImg2VideoNode ---
class MoonvalleyImg2VideoNode(BaseMoonvalleyVideoNode):

    @classmethod
    def INPUT_TYPES(cls):
        return super().INPUT_TYPES()

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    DESCRIPTION = "Moonvalley Marey Image to Video Node"

    def generate(
        self, prompt, negative_prompt, unique_id: Optional[str] = None, **kwargs
    ):
        image = kwargs.get("image", None)
        if image is None:
            raise MoonvalleyApiError("image is required")
        total_frames = get_total_frames_from_length()

        validate_input_image(image, True)
        validate_prompts(prompt, negative_prompt, MOONVALLEY_MAREY_MAX_PROMPT_LENGTH)
        width_height = self.parseWidthHeightFromRes(kwargs.get("resolution"))

        inference_params = MoonvalleyTextToVideoInferenceParams(
            negative_prompt=negative_prompt,
            steps=kwargs.get("steps"),
            seed=kwargs.get("seed"),
            guidance_scale=kwargs.get("prompt_adherence"),
            num_frames=total_frames,
            width=width_height.get("width"),
            height=width_height.get("height"),
            use_negative_prompts=True,
        )
        """Upload image to comfy backend to have a URL available for further processing"""
        # Get MIME type from tensor - assuming PNG format for image tensors
        mime_type = "image/png"

        image_url = upload_images_to_comfyapi(
            image, max_images=1, auth_kwargs=kwargs, mime_type=mime_type
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
            auth_kwargs=kwargs,
        )
        task_creation_response = initial_operation.execute()
        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.id

        final_response = self.get_response(
            task_id, auth_kwargs=kwargs, node_id=unique_id
        )
        video = download_url_to_video_output(final_response.output_url)
        return (video,)


# --- MoonvalleyVid2VidNode ---
class MoonvalleyVideo2VideoNode(BaseMoonvalleyVideoNode):
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        input_types = super().INPUT_TYPES()
        for param in ["resolution", "image"]:
            if param in input_types["required"]:
                del input_types["required"][param]
            if param in input_types["optional"]:
                del input_types["optional"][param]
        input_types["optional"] = {
            "video": (
                IO.VIDEO,
                {
                    "default": "",
                    "multiline": False,
                    "tooltip": "The reference video used to generate the output video. Input a 5s video for 128 frames and a 10s video for 256 frames. Longer videos will be trimmed automatically.",
                },
            ),
            "control_type": (
                ["Motion Transfer", "Pose Transfer"],
                {"default": "Motion Transfer"},
            ),
            "motion_intensity": (
                "INT",
                {
                    "default": 100,
                    "step": 1,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Only used if control_type is 'Motion Transfer'",
                },
            ),
        }

        return input_types

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)

    def generate(
        self, prompt, negative_prompt, unique_id: Optional[str] = None, **kwargs
    ):
        video = kwargs.get("video")
        num_frames = get_total_frames_from_length()

        if not video:
            raise MoonvalleyApiError("video is required")

        """Validate video input"""
        video_url = ""
        if video:
            validated_video = validate_input_video(video, num_frames, False)
            video_url = upload_video_to_comfyapi(validated_video, auth_kwargs=kwargs)

        control_type = kwargs.get("control_type")
        motion_intensity = kwargs.get("motion_intensity")

        """Validate prompts and inference input"""
        validate_prompts(prompt, negative_prompt)
        inference_params = MoonvalleyVideoToVideoInferenceParams(
            negative_prompt=negative_prompt,
            steps=kwargs.get("steps"),
            seed=kwargs.get("seed"),
            guidance_scale=kwargs.get("prompt_adherence"),
            control_params={"motion_intensity": motion_intensity},
        )

        control = self.parseControlParameter(control_type)

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
            auth_kwargs=kwargs,
        )
        task_creation_response = initial_operation.execute()
        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.id

        final_response = self.get_response(
            task_id, auth_kwargs=kwargs, node_id=unique_id
        )

        video = download_url_to_video_output(final_response.output_url)

        return (video,)


# --- MoonvalleyTxt2VideoNode ---
class MoonvalleyTxt2VideoNode(BaseMoonvalleyVideoNode):
    def __init__(self):
        super().__init__()

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)

    @classmethod
    def INPUT_TYPES(cls):
        input_types = super().INPUT_TYPES()
        # Remove image-specific parameters
        for param in ["image"]:
            if param in input_types["optional"]:
                del input_types["optional"][param]
        return input_types

    def generate(
        self, prompt, negative_prompt, unique_id: Optional[str] = None, **kwargs
    ):
        validate_prompts(prompt, negative_prompt, MOONVALLEY_MAREY_MAX_PROMPT_LENGTH)
        width_height = self.parseWidthHeightFromRes(kwargs.get("resolution"))
        num_frames = get_total_frames_from_length()

        inference_params = MoonvalleyTextToVideoInferenceParams(
            negative_prompt=negative_prompt,
            steps=kwargs.get("steps"),
            seed=kwargs.get("seed"),
            guidance_scale=kwargs.get("prompt_adherence"),
            num_frames=num_frames,
            width=width_height.get("width"),
            height=width_height.get("height"),
        )
        request = MoonvalleyTextToVideoRequest(
            prompt_text=prompt, inference_params=inference_params
        )

        initial_operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=API_TXT2VIDEO_ENDPOINT,
                method=HttpMethod.POST,
                request_model=MoonvalleyTextToVideoRequest,
                response_model=MoonvalleyPromptResponse,
            ),
            request=request,
            auth_kwargs=kwargs,
        )
        task_creation_response = initial_operation.execute()
        validate_task_creation_response(task_creation_response)
        task_id = task_creation_response.id

        final_response = self.get_response(
            task_id, auth_kwargs=kwargs, node_id=unique_id
        )

        video = download_url_to_video_output(final_response.output_url)
        return (video,)


NODE_CLASS_MAPPINGS = {
    "MoonvalleyImg2VideoNode": MoonvalleyImg2VideoNode,
    "MoonvalleyTxt2VideoNode": MoonvalleyTxt2VideoNode,
    # "MoonvalleyVideo2VideoNode": MoonvalleyVideo2VideoNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonvalleyImg2VideoNode": "Moonvalley Marey Image to Video",
    "MoonvalleyTxt2VideoNode": "Moonvalley Marey Text to Video",
    # "MoonvalleyVideo2VideoNode": "Moonvalley Marey Video to Video",
}


def get_total_frames_from_length(length="5s"):
    # if length == '5s':
    #     return 128
    # elif length == '10s':
    #     return 256
    return 128
    # else:
    #     raise MoonvalleyApiError("length is required")

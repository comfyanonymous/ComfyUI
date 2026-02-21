import re

from pydantic import BaseModel, Field
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_to_base64_string,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    tensor_to_base64_string,
    upload_video_to_comfyapi,
    validate_audio_duration,
    validate_video_duration,
)


class Text2ImageInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)


class Image2ImageInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)
    images: list[str] = Field(..., min_length=1, max_length=2)


class Text2VideoInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)
    audio_url: str | None = Field(None)


class Image2VideoInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)
    img_url: str = Field(...)
    audio_url: str | None = Field(None)


class Reference2VideoInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)
    reference_video_urls: list[str] = Field(...)


class Txt2ImageParametersField(BaseModel):
    size: str = Field(...)
    n: int = Field(1, description="Number of images to generate.")  # we support only value=1
    seed: int = Field(..., ge=0, le=2147483647)
    prompt_extend: bool = Field(True)
    watermark: bool = Field(False)


class Image2ImageParametersField(BaseModel):
    size: str | None = Field(None)
    n: int = Field(1, description="Number of images to generate.")  # we support only value=1
    seed: int = Field(..., ge=0, le=2147483647)
    watermark: bool = Field(False)


class Text2VideoParametersField(BaseModel):
    size: str = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    duration: int = Field(5, ge=5, le=15)
    prompt_extend: bool = Field(True)
    watermark: bool = Field(False)
    audio: bool = Field(False, description="Whether to generate audio automatically.")
    shot_type: str = Field("single")


class Image2VideoParametersField(BaseModel):
    resolution: str = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    duration: int = Field(5, ge=5, le=15)
    prompt_extend: bool = Field(True)
    watermark: bool = Field(False)
    audio: bool = Field(False, description="Whether to generate audio automatically.")
    shot_type: str = Field("single")


class Reference2VideoParametersField(BaseModel):
    size: str = Field(...)
    duration: int = Field(5, ge=5, le=15)
    shot_type: str = Field("single")
    seed: int = Field(..., ge=0, le=2147483647)
    watermark: bool = Field(False)


class Text2ImageTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Text2ImageInputField = Field(...)
    parameters: Txt2ImageParametersField = Field(...)


class Image2ImageTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Image2ImageInputField = Field(...)
    parameters: Image2ImageParametersField = Field(...)


class Text2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Text2VideoInputField = Field(...)
    parameters: Text2VideoParametersField = Field(...)


class Image2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Image2VideoInputField = Field(...)
    parameters: Image2VideoParametersField = Field(...)


class Reference2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Reference2VideoInputField = Field(...)
    parameters: Reference2VideoParametersField = Field(...)


class TaskCreationOutputField(BaseModel):
    task_id: str = Field(...)
    task_status: str = Field(...)


class TaskCreationResponse(BaseModel):
    output: TaskCreationOutputField | None = Field(None)
    request_id: str = Field(...)
    code: str | None = Field(None, description="Error code for the failed request.")
    message: str | None = Field(None, description="Details about the failed request.")


class TaskResult(BaseModel):
    url: str | None = Field(None)
    code: str | None = Field(None)
    message: str | None = Field(None)


class ImageTaskStatusOutputField(TaskCreationOutputField):
    task_id: str = Field(...)
    task_status: str = Field(...)
    results: list[TaskResult] | None = Field(None)


class VideoTaskStatusOutputField(TaskCreationOutputField):
    task_id: str = Field(...)
    task_status: str = Field(...)
    video_url: str | None = Field(None)
    code: str | None = Field(None)
    message: str | None = Field(None)


class ImageTaskStatusResponse(BaseModel):
    output: ImageTaskStatusOutputField | None = Field(None)
    request_id: str = Field(...)


class VideoTaskStatusResponse(BaseModel):
    output: VideoTaskStatusOutputField | None = Field(None)
    request_id: str = Field(...)


RES_IN_PARENS = re.compile(r"\((\d+)\s*[x×]\s*(\d+)\)")


class WanTextToImageApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WanTextToImageApi",
            display_name="Wan Text to Image",
            category="api node/image/Wan",
            description="Generates an image based on a text prompt.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["wan2.5-t2i-preview"],
                    default="wan2.5-t2i-preview",
                    tooltip="Model to use.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt describing the elements and visual features. Supports English and Chinese.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative prompt describing what to avoid.",
                    optional=True,
                ),
                IO.Int.Input(
                    "width",
                    default=1024,
                    min=768,
                    max=1440,
                    step=32,
                    optional=True,
                ),
                IO.Int.Input(
                    "height",
                    default=1024,
                    min=768,
                    max=1440,
                    step=32,
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
                    optional=True,
                    advanced=True,
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
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.03}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        prompt_extend: bool = True,
        watermark: bool = False,
    ):
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/wan/api/v1/services/aigc/text2image/image-synthesis", method="POST"),
            response_model=TaskCreationResponse,
            data=Text2ImageTaskCreationRequest(
                model=model,
                input=Text2ImageInputField(prompt=prompt, negative_prompt=negative_prompt),
                parameters=Txt2ImageParametersField(
                    size=f"{width}*{height}",
                    seed=seed,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=ImageTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            estimated_duration=9,
            poll_interval=3,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(str(response.output.results[0].url)))


class WanImageToImageApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WanImageToImageApi",
            display_name="Wan Image to Image",
            category="api node/image/Wan",
            description="Generates an image from one or two input images and a text prompt. "
            "The output image is currently fixed at 1.6 MP, and its aspect ratio matches the input image(s).",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["wan2.5-i2i-preview"],
                    default="wan2.5-i2i-preview",
                    tooltip="Model to use.",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Single-image editing or multi-image fusion. Maximum 2 images.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt describing the elements and visual features. Supports English and Chinese.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative prompt describing what to avoid.",
                    optional=True,
                ),
                # redo this later as an optional combo of recommended resolutions
                # IO.Int.Input(
                #     "width",
                #     default=1280,
                #     min=384,
                #     max=1440,
                #     step=16,
                #     optional=True,
                # ),
                # IO.Int.Input(
                #     "height",
                #     default=1280,
                #     min=384,
                #     max=1440,
                #     step=16,
                #     optional=True,
                # ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
                    optional=True,
                    advanced=True,
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
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.03}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        prompt: str,
        negative_prompt: str = "",
        # width: int = 1024,
        # height: int = 1024,
        seed: int = 0,
        watermark: bool = False,
    ):
        n_images = get_number_of_images(image)
        if n_images not in (1, 2):
            raise ValueError(f"Expected 1 or 2 input images, but got {n_images}.")
        images = []
        for i in image:
            images.append("data:image/png;base64," + tensor_to_base64_string(i, total_pixels=4096 * 4096))
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/wan/api/v1/services/aigc/image2image/image-synthesis", method="POST"),
            response_model=TaskCreationResponse,
            data=Image2ImageTaskCreationRequest(
                model=model,
                input=Image2ImageInputField(prompt=prompt, negative_prompt=negative_prompt, images=images),
                parameters=Image2ImageParametersField(
                    # size=f"{width}*{height}",
                    seed=seed,
                    watermark=watermark,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=ImageTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            estimated_duration=42,
            poll_interval=4,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(str(response.output.results[0].url)))


class WanTextToVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WanTextToVideoApi",
            display_name="Wan Text to Video",
            category="api node/video/Wan",
            description="Generates a video based on a text prompt.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["wan2.5-t2v-preview", "wan2.6-t2v"],
                    default="wan2.6-t2v",
                    tooltip="Model to use.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt describing the elements and visual features. Supports English and Chinese.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative prompt describing what to avoid.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "size",
                    options=[
                        "480p: 1:1 (624x624)",
                        "480p: 16:9 (832x480)",
                        "480p: 9:16 (480x832)",
                        "720p: 1:1 (960x960)",
                        "720p: 16:9 (1280x720)",
                        "720p: 9:16 (720x1280)",
                        "720p: 4:3 (1088x832)",
                        "720p: 3:4 (832x1088)",
                        "1080p: 1:1 (1440x1440)",
                        "1080p: 16:9 (1920x1080)",
                        "1080p: 9:16 (1080x1920)",
                        "1080p: 4:3 (1632x1248)",
                        "1080p: 3:4 (1248x1632)",
                    ],
                    default="720p: 1:1 (960x960)",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=15,
                    step=5,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="A 15-second duration is available only for the Wan 2.6 model.",
                    optional=True,
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Audio must contain a clear, loud voice, without extraneous noise or background music.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    optional=True,
                    tooltip="If no audio input is provided, generate audio automatically.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "shot_type",
                    options=["single", "multi"],
                    tooltip="Specifies the shot type for the generated video, that is, whether the video is a "
                    "single continuous shot or multiple shots with cuts. "
                    "This parameter takes effect only when prompt_extend is True.",
                    optional=True,
                    advanced=True,
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
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["duration", "size"]),
                expr="""
                (
                  $ppsTable := { "480p": 0.05, "720p": 0.1, "1080p": 0.15 };
                  $resKey := $substringBefore(widgets.size, ":");
                  $pps := $lookup($ppsTable, $resKey);
                  { "type": "usd", "usd": $round($pps * widgets.duration, 2) }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        negative_prompt: str = "",
        size: str = "720p: 1:1 (960x960)",
        duration: int = 5,
        audio: Input.Audio | None = None,
        seed: int = 0,
        generate_audio: bool = False,
        prompt_extend: bool = True,
        watermark: bool = False,
        shot_type: str = "single",
    ):
        if "480p" in size and model == "wan2.6-t2v":
            raise ValueError("The Wan 2.6 model does not support 480p.")
        if duration == 15 and model == "wan2.5-t2v-preview":
            raise ValueError("A 15-second duration is supported only by the Wan 2.6 model.")
        width, height = RES_IN_PARENS.search(size).groups()
        audio_url = None
        if audio is not None:
            validate_audio_duration(audio, 3.0, 29.0)
            audio_url = "data:audio/mp3;base64," + audio_to_base64_string(audio, "mp3", "libmp3lame")

        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis", method="POST"),
            response_model=TaskCreationResponse,
            data=Text2VideoTaskCreationRequest(
                model=model,
                input=Text2VideoInputField(prompt=prompt, negative_prompt=negative_prompt, audio_url=audio_url),
                parameters=Text2VideoParametersField(
                    size=f"{width}*{height}",
                    duration=duration,
                    seed=seed,
                    audio=generate_audio,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                    shot_type=shot_type,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            estimated_duration=120 * int(duration / 5),
            poll_interval=6,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class WanImageToVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WanImageToVideoApi",
            display_name="Wan Image to Video",
            category="api node/video/Wan",
            description="Generates a video from the first frame and a text prompt.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["wan2.5-i2v-preview", "wan2.6-i2v"],
                    default="wan2.6-i2v",
                    tooltip="Model to use.",
                ),
                IO.Image.Input(
                    "image",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt describing the elements and visual features. Supports English and Chinese.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative prompt describing what to avoid.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=[
                        "480P",
                        "720P",
                        "1080P",
                    ],
                    default="720P",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=15,
                    step=5,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration 15 available only for WAN2.6 model.",
                    optional=True,
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Audio must contain a clear, loud voice, without extraneous noise or background music.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    optional=True,
                    tooltip="If no audio input is provided, generate audio automatically.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "shot_type",
                    options=["single", "multi"],
                    tooltip="Specifies the shot type for the generated video, that is, whether the video is a "
                    "single continuous shot or multiple shots with cuts. "
                    "This parameter takes effect only when prompt_extend is True.",
                    optional=True,
                    advanced=True,
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
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["duration", "resolution"]),
                expr="""
                (
                  $ppsTable := { "480p": 0.05, "720p": 0.1, "1080p": 0.15 };
                  $pps := $lookup($ppsTable, widgets.resolution);
                  { "type": "usd", "usd": $round($pps * widgets.duration, 2) }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        prompt: str,
        negative_prompt: str = "",
        resolution: str = "720P",
        duration: int = 5,
        audio: Input.Audio | None = None,
        seed: int = 0,
        generate_audio: bool = False,
        prompt_extend: bool = True,
        watermark: bool = False,
        shot_type: str = "single",
    ):
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        if "480P" in resolution and model == "wan2.6-i2v":
            raise ValueError("The Wan 2.6 model does not support 480P.")
        if duration == 15 and model == "wan2.5-i2v-preview":
            raise ValueError("A 15-second duration is supported only by the Wan 2.6 model.")
        image_url = "data:image/png;base64," + tensor_to_base64_string(image, total_pixels=2000 * 2000)
        audio_url = None
        if audio is not None:
            validate_audio_duration(audio, 3.0, 29.0)
            audio_url = "data:audio/mp3;base64," + audio_to_base64_string(audio, "mp3", "libmp3lame")
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis", method="POST"),
            response_model=TaskCreationResponse,
            data=Image2VideoTaskCreationRequest(
                model=model,
                input=Image2VideoInputField(
                    prompt=prompt, negative_prompt=negative_prompt, img_url=image_url, audio_url=audio_url
                ),
                parameters=Image2VideoParametersField(
                    resolution=resolution,
                    duration=duration,
                    seed=seed,
                    audio=generate_audio,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                    shot_type=shot_type,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            estimated_duration=120 * int(duration / 5),
            poll_interval=6,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class WanReferenceVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WanReferenceVideoApi",
            display_name="Wan Reference to Video",
            category="api node/video/Wan",
            description="Use the character and voice from input videos, combined with a prompt, "
            "to generate a new video that maintains character consistency.",
            inputs=[
                IO.Combo.Input("model", options=["wan2.6-r2v"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt describing the elements and visual features. Supports English and Chinese. "
                    "Use identifiers such as `character1` and `character2` to refer to the reference characters.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative prompt describing what to avoid.",
                ),
                IO.Autogrow.Input(
                    "reference_videos",
                    template=IO.Autogrow.TemplateNames(
                        IO.Video.Input("reference_video"),
                        names=["character1", "character2", "character3"],
                        min=1,
                    ),
                ),
                IO.Combo.Input(
                    "size",
                    options=[
                        "720p: 1:1 (960x960)",
                        "720p: 16:9 (1280x720)",
                        "720p: 9:16 (720x1280)",
                        "720p: 4:3 (1088x832)",
                        "720p: 3:4 (832x1088)",
                        "1080p: 1:1 (1440x1440)",
                        "1080p: 16:9 (1920x1080)",
                        "1080p: 9:16 (1080x1920)",
                        "1080p: 4:3 (1632x1248)",
                        "1080p: 3:4 (1248x1632)",
                    ],
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=10,
                    step=5,
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Combo.Input(
                    "shot_type",
                    options=["single", "multi"],
                    tooltip="Specifies the shot type for the generated video, that is, whether the video is a "
                    "single continuous shot or multiple shots with cuts.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
                    advanced=True,
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
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["size", "duration"]),
                expr="""
                (
                  $rate := $contains(widgets.size, "1080p") ? 0.15 : 0.10;
                  $inputMin := 2 * $rate;
                  $inputMax := 5 * $rate;
                  $outputPrice := widgets.duration * $rate;
                  {
                    "type": "range_usd",
                    "min_usd": $inputMin + $outputPrice,
                    "max_usd": $inputMax + $outputPrice
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        negative_prompt: str,
        reference_videos: IO.Autogrow.Type,
        size: str,
        duration: int,
        seed: int,
        shot_type: str,
        watermark: bool,
    ):
        reference_video_urls = []
        for i in reference_videos:
            validate_video_duration(reference_videos[i], min_duration=2, max_duration=30)
        for i in reference_videos:
            reference_video_urls.append(await upload_video_to_comfyapi(cls, reference_videos[i]))
        width, height = RES_IN_PARENS.search(size).groups()
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis", method="POST"),
            response_model=TaskCreationResponse,
            data=Reference2VideoTaskCreationRequest(
                model=model,
                input=Reference2VideoInputField(
                    prompt=prompt, negative_prompt=negative_prompt, reference_video_urls=reference_video_urls
                ),
                parameters=Reference2VideoParametersField(
                    size=f"{width}*{height}",
                    duration=duration,
                    shot_type=shot_type,
                    watermark=watermark,
                    seed=seed,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            poll_interval=6,
            max_poll_attempts=280,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class WanApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            WanTextToImageApi,
            WanImageToImageApi,
            WanTextToVideoApi,
            WanImageToVideoApi,
            WanReferenceVideoApi,
        ]


async def comfy_entrypoint() -> WanApiExtension:
    return WanApiExtension()

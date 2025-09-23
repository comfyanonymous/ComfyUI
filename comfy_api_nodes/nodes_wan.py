import re
from typing import Optional, Type, Union
from typing_extensions import override

import torch
from pydantic import BaseModel, Field
from comfy_api.latest import ComfyExtension, Input, io as comfy_io
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
    R,
    T,
)
from comfy_api_nodes.util.validation_utils import get_number_of_images, validate_audio_duration

from comfy_api_nodes.apinode_utils import (
    download_url_to_image_tensor,
    download_url_to_video_output,
    tensor_to_base64_string,
    audio_to_base64_string,
)

class Text2ImageInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: Optional[str] = Field(None)


class Text2VideoInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: Optional[str] = Field(None)
    audio_url: Optional[str] = Field(None)


class Image2VideoInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: Optional[str] = Field(None)
    img_url: str = Field(...)
    audio_url: Optional[str] = Field(None)


class Txt2ImageParametersField(BaseModel):
    size: str = Field(...)
    n: int = Field(1, description="Number of images to generate.")  # we support only value=1
    seed: int = Field(..., ge=0, le=2147483647)
    prompt_extend: bool = Field(True)
    watermark: bool = Field(True)


class Text2VideoParametersField(BaseModel):
    size: str = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    duration: int = Field(5, ge=5, le=10)
    prompt_extend: bool = Field(True)
    watermark: bool = Field(True)
    audio: bool = Field(False, description="Should be audio generated automatically")


class Image2VideoParametersField(BaseModel):
    resolution: str = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    duration: int = Field(5, ge=5, le=10)
    prompt_extend: bool = Field(True)
    watermark: bool = Field(True)
    audio: bool = Field(False, description="Should be audio generated automatically")


class Text2ImageTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Text2ImageInputField = Field(...)
    parameters: Txt2ImageParametersField = Field(...)


class Text2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Text2VideoInputField = Field(...)
    parameters: Text2VideoParametersField = Field(...)


class Image2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Image2VideoInputField = Field(...)
    parameters: Image2VideoParametersField = Field(...)


class TaskCreationOutputField(BaseModel):
    task_id: str = Field(...)
    task_status: str = Field(...)


class TaskCreationResponse(BaseModel):
    output: Optional[TaskCreationOutputField] = Field(None)
    request_id: str = Field(...)
    code: Optional[str] = Field(None, description="The error code of the failed request.")
    message: Optional[str] = Field(None, description="Details of the failed request.")


class TaskResult(BaseModel):
    url: Optional[str] = Field(None)
    code: Optional[str] = Field(None)
    message: Optional[str] = Field(None)


class ImageTaskStatusOutputField(TaskCreationOutputField):
    task_id: str = Field(...)
    task_status: str = Field(...)
    results: Optional[list[TaskResult]] = Field(None)


class VideoTaskStatusOutputField(TaskCreationOutputField):
    task_id: str = Field(...)
    task_status: str = Field(...)
    video_url: Optional[str] = Field(None)
    code: Optional[str] = Field(None)
    message: Optional[str] = Field(None)


class ImageTaskStatusResponse(BaseModel):
    output: Optional[ImageTaskStatusOutputField] = Field(None)
    request_id: str = Field(...)


class VideoTaskStatusResponse(BaseModel):
    output: Optional[VideoTaskStatusOutputField] = Field(None)
    request_id: str = Field(...)


RES_IN_PARENS = re.compile(r'\((\d+)\s*[x×]\s*(\d+)\)')


async def process_task(
    auth_kwargs: dict[str, str],
    url: str,
    request_model: Type[T],
    response_model: Type[R],
    payload: Union[Text2ImageTaskCreationRequest, Text2VideoTaskCreationRequest, Image2VideoTaskCreationRequest],
    node_id: str,
    estimated_duration: int,
    poll_interval: int,
) -> Type[R]:
    initial_response = await SynchronousOperation(
        endpoint=ApiEndpoint(
            path=url,
            method=HttpMethod.POST,
            request_model=request_model,
            response_model=TaskCreationResponse,
        ),
        request=payload,
        auth_kwargs=auth_kwargs,
    ).execute()

    if not initial_response.output:
        raise Exception(f"Unknown error occurred: {initial_response.code} - {initial_response.message}")

    return await PollingOperation(
        poll_endpoint=ApiEndpoint(
            path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}",
            method=HttpMethod.GET,
            request_model=EmptyRequest,
            response_model=response_model,
        ),
        completed_statuses=["SUCCEEDED"],
        failed_statuses=["FAILED", "CANCELED", "UNKNOWN"],
        status_extractor=lambda x: x.output.task_status,
        estimated_duration=estimated_duration,
        poll_interval=poll_interval,
        node_id=node_id,
        auth_kwargs=auth_kwargs,
    ).execute()


class WanTextToImageApi(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="WanTextToImageApi",
            display_name="Wan Text to Image",
            category="api node/image/Wan",
            description="Generates image based on text prompt.",
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=["wan2.5-t2i-preview"],
                    default="wan2.5-t2i-preview",
                    tooltip="Model to use.",
                ),
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt used to describe the elements and visual features, supports English/Chinese.",
                ),
                comfy_io.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative text prompt to guide what to avoid.",
                    optional=True,
                ),
                comfy_io.Int.Input(
                    "width",
                    default=1024,
                    min=768,
                    max=1440,
                    step=32,
                    optional=True,
                ),
                comfy_io.Int.Input(
                    "height",
                    default=1024,
                    min=768,
                    max=1440,
                    step=32,
                    optional=True,
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                comfy_io.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    optional=True,
                ),
                comfy_io.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip="Whether to add an \"AI generated\" watermark to the result.",
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
        model: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        prompt_extend: bool = True,
        watermark: bool = True,
    ):
        payload = Text2ImageTaskCreationRequest(
            model=model,
            input=Text2ImageInputField(prompt=prompt, negative_prompt=negative_prompt),
            parameters=Txt2ImageParametersField(
                size=f"{width}*{height}",
                seed=seed,
                prompt_extend=prompt_extend,
                watermark=watermark,
            ),
        )
        response = await process_task(
            {
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
            "/proxy/wan/api/v1/services/aigc/text2image/image-synthesis",
            request_model=Text2ImageTaskCreationRequest,
            response_model=ImageTaskStatusResponse,
            payload=payload,
            node_id=cls.hidden.unique_id,
            estimated_duration=9,
            poll_interval=3,
        )
        return comfy_io.NodeOutput(await download_url_to_image_tensor(str(response.output.results[0].url)))


class WanTextToVideoApi(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="WanTextToVideoApi",
            display_name="Wan Text to Video",
            category="api node/video/Wan",
            description="Generates video based on text prompt.",
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=["wan2.5-t2v-preview"],
                    default="wan2.5-t2v-preview",
                    tooltip="Model to use.",
                ),
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt used to describe the elements and visual features, supports English/Chinese.",
                ),
                comfy_io.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative text prompt to guide what to avoid.",
                    optional=True,
                ),
                comfy_io.Combo.Input(
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
                    default="480p: 1:1 (624x624)",
                    optional=True,
                ),
                comfy_io.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=10,
                    step=5,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Available durations: 5 and 10 seconds",
                    optional=True,
                ),
                comfy_io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Audio must contain a clear, loud voice, without extraneous noise, background music.",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                comfy_io.Boolean.Input(
                    "generate_audio",
                    default=False,
                    optional=True,
                    tooltip="If there is no audio input, generate audio automatically.",
                ),
                comfy_io.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    optional=True,
                ),
                comfy_io.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip="Whether to add an \"AI generated\" watermark to the result.",
                    optional=True,
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
        model: str,
        prompt: str,
        negative_prompt: str = "",
        size: str = "480p: 1:1 (624x624)",
        duration: int = 5,
        audio: Optional[Input.Audio] = None,
        seed: int = 0,
        generate_audio: bool = False,
        prompt_extend: bool = True,
        watermark: bool = True,
    ):
        width, height = RES_IN_PARENS.search(size).groups()
        audio_url = None
        if audio is not None:
            validate_audio_duration(audio, 3.0, 29.0)
            audio_url = "data:audio/mp3;base64," + audio_to_base64_string(audio, "mp3", "libmp3lame")
        payload = Text2VideoTaskCreationRequest(
            model=model,
            input=Text2VideoInputField(prompt=prompt, negative_prompt=negative_prompt, audio_url=audio_url),
            parameters=Text2VideoParametersField(
                size=f"{width}*{height}",
                duration=duration,
                seed=seed,
                audio=generate_audio,
                prompt_extend=prompt_extend,
                watermark=watermark,
            ),
        )
        response = await process_task(
            {
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
            "/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
            request_model=Text2VideoTaskCreationRequest,
            response_model=VideoTaskStatusResponse,
            payload=payload,
            node_id=cls.hidden.unique_id,
            estimated_duration=120 * int(duration / 5),
            poll_interval=6,
        )
        return comfy_io.NodeOutput(await download_url_to_video_output(response.output.video_url))


class WanImageToVideoApi(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="WanImageToVideoApi",
            display_name="Wan Image to Video",
            category="api node/video/Wan",
            description="Generates video based on the first frame and text prompt.",
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=["wan2.5-i2v-preview"],
                    default="wan2.5-i2v-preview",
                    tooltip="Model to use.",
                ),
                comfy_io.Image.Input(
                    "image",
                ),
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt used to describe the elements and visual features, supports English/Chinese.",
                ),
                comfy_io.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative text prompt to guide what to avoid.",
                    optional=True,
                ),
                comfy_io.Combo.Input(
                    "resolution",
                    options=[
                        "480P",
                        "720P",
                        "1080P",
                    ],
                    default="480P",
                    optional=True,
                ),
                comfy_io.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=10,
                    step=5,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Available durations: 5 and 10 seconds",
                    optional=True,
                ),
                comfy_io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Audio must contain a clear, loud voice, without extraneous noise, background music.",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                comfy_io.Boolean.Input(
                    "generate_audio",
                    default=False,
                    optional=True,
                    tooltip="If there is no audio input, generate audio automatically.",
                ),
                comfy_io.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    optional=True,
                ),
                comfy_io.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip="Whether to add an \"AI generated\" watermark to the result.",
                    optional=True,
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
        model: str,
        image: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        resolution: str = "480P",
        duration: int = 5,
        audio: Optional[Input.Audio] = None,
        seed: int = 0,
        generate_audio: bool = False,
        prompt_extend: bool = True,
        watermark: bool = True,
    ):
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        image_url = "data:image/png;base64," + tensor_to_base64_string(image, total_pixels=2000*2000)
        audio_url = None
        if audio is not None:
            validate_audio_duration(audio, 3.0, 29.0)
            audio_url = "data:audio/mp3;base64," + audio_to_base64_string(audio, "mp3", "libmp3lame")
        payload = Image2VideoTaskCreationRequest(
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
            ),
        )
        response = await process_task(
            {
                "auth_token": cls.hidden.auth_token_comfy_org,
                "comfy_api_key": cls.hidden.api_key_comfy_org,
            },
            "/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
            request_model=Image2VideoTaskCreationRequest,
            response_model=VideoTaskStatusResponse,
            payload=payload,
            node_id=cls.hidden.unique_id,
            estimated_duration=120 * int(duration / 5),
            poll_interval=6,
        )
        return comfy_io.NodeOutput(await download_url_to_video_output(response.output.video_url))


class WanApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[comfy_io.ComfyNode]]:
        return [
            WanTextToImageApi,
            WanTextToVideoApi,
            WanImageToVideoApi,
        ]


async def comfy_entrypoint() -> WanApiExtension:
    return WanApiExtension()

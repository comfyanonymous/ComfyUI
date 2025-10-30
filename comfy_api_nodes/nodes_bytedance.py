import logging
import math
from enum import Enum
from typing import Literal, Optional, Union

import torch
from pydantic import BaseModel, Field
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_number_of_images,
    image_tensor_pair_to_batch,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    validate_image_aspect_ratio,
    validate_image_dimensions,
    validate_string,
)

BYTEPLUS_IMAGE_ENDPOINT = "/proxy/byteplus/api/v3/images/generations"

# Long-running tasks endpoints(e.g., video)
BYTEPLUS_TASK_ENDPOINT = "/proxy/byteplus/api/v3/contents/generations/tasks"
BYTEPLUS_TASK_STATUS_ENDPOINT = "/proxy/byteplus/api/v3/contents/generations/tasks"  # + /{task_id}


class Text2ImageModelName(str, Enum):
    seedream_3 = "seedream-3-0-t2i-250415"


class Image2ImageModelName(str, Enum):
    seededit_3 = "seededit-3-0-i2i-250628"


class Text2VideoModelName(str, Enum):
    seedance_1_pro = "seedance-1-0-pro-250528"
    seedance_1_lite = "seedance-1-0-lite-t2v-250428"


class Image2VideoModelName(str, Enum):
    """note(August 31): Pro model only supports FirstFrame: https://docs.byteplus.com/en/docs/ModelArk/1520757"""

    seedance_1_pro = "seedance-1-0-pro-250528"
    seedance_1_lite = "seedance-1-0-lite-i2v-250428"


class Text2ImageTaskCreationRequest(BaseModel):
    model: Text2ImageModelName = Text2ImageModelName.seedream_3
    prompt: str = Field(...)
    response_format: Optional[str] = Field("url")
    size: Optional[str] = Field(None)
    seed: Optional[int] = Field(0, ge=0, le=2147483647)
    guidance_scale: Optional[float] = Field(..., ge=1.0, le=10.0)
    watermark: Optional[bool] = Field(True)


class Image2ImageTaskCreationRequest(BaseModel):
    model: Image2ImageModelName = Image2ImageModelName.seededit_3
    prompt: str = Field(...)
    response_format: Optional[str] = Field("url")
    image: str = Field(..., description="Base64 encoded string or image URL")
    size: Optional[str] = Field("adaptive")
    seed: Optional[int] = Field(..., ge=0, le=2147483647)
    guidance_scale: Optional[float] = Field(..., ge=1.0, le=10.0)
    watermark: Optional[bool] = Field(True)


class Seedream4Options(BaseModel):
    max_images: int = Field(15)


class Seedream4TaskCreationRequest(BaseModel):
    model: str = Field("seedream-4-0-250828")
    prompt: str = Field(...)
    response_format: str = Field("url")
    image: Optional[list[str]] = Field(None, description="Image URLs")
    size: str = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    sequential_image_generation: str = Field("disabled")
    sequential_image_generation_options: Seedream4Options = Field(Seedream4Options(max_images=15))
    watermark: bool = Field(True)


class ImageTaskCreationResponse(BaseModel):
    model: str = Field(...)
    created: int = Field(..., description="Unix timestamp (in seconds) indicating time when the request was created.")
    data: list = Field([], description="Contains information about the generated image(s).")
    error: dict = Field({}, description="Contains `code` and `message` fields in case of error.")


class TaskTextContent(BaseModel):
    type: str = Field("text")
    text: str = Field(...)


class TaskImageContentUrl(BaseModel):
    url: str = Field(...)


class TaskImageContent(BaseModel):
    type: str = Field("image_url")
    image_url: TaskImageContentUrl = Field(...)
    role: Optional[Literal["first_frame", "last_frame", "reference_image"]] = Field(None)


class Text2VideoTaskCreationRequest(BaseModel):
    model: Text2VideoModelName = Text2VideoModelName.seedance_1_pro
    content: list[TaskTextContent] = Field(..., min_length=1)


class Image2VideoTaskCreationRequest(BaseModel):
    model: Image2VideoModelName = Image2VideoModelName.seedance_1_pro
    content: list[Union[TaskTextContent, TaskImageContent]] = Field(..., min_length=2)


class TaskCreationResponse(BaseModel):
    id: str = Field(...)


class TaskStatusError(BaseModel):
    code: str = Field(...)
    message: str = Field(...)


class TaskStatusResult(BaseModel):
    video_url: str = Field(...)


class TaskStatusResponse(BaseModel):
    id: str = Field(...)
    model: str = Field(...)
    status: Literal["queued", "running", "cancelled", "succeeded", "failed"] = Field(...)
    error: Optional[TaskStatusError] = Field(None)
    content: Optional[TaskStatusResult] = Field(None)


RECOMMENDED_PRESETS = [
    ("1024x1024 (1:1)", 1024, 1024),
    ("864x1152 (3:4)", 864, 1152),
    ("1152x864 (4:3)", 1152, 864),
    ("1280x720 (16:9)", 1280, 720),
    ("720x1280 (9:16)", 720, 1280),
    ("832x1248 (2:3)", 832, 1248),
    ("1248x832 (3:2)", 1248, 832),
    ("1512x648 (21:9)", 1512, 648),
    ("2048x2048 (1:1)", 2048, 2048),
    ("Custom", None, None),
]

RECOMMENDED_PRESETS_SEEDREAM_4 = [
    ("2048x2048 (1:1)", 2048, 2048),
    ("2304x1728 (4:3)", 2304, 1728),
    ("1728x2304 (3:4)", 1728, 2304),
    ("2560x1440 (16:9)", 2560, 1440),
    ("1440x2560 (9:16)", 1440, 2560),
    ("2496x1664 (3:2)", 2496, 1664),
    ("1664x2496 (2:3)", 1664, 2496),
    ("3024x1296 (21:9)", 3024, 1296),
    ("4096x4096 (1:1)", 4096, 4096),
    ("Custom", None, None),
]

# The time in this dictionary are given for 10 seconds duration.
VIDEO_TASKS_EXECUTION_TIME = {
    "seedance-1-0-lite-t2v-250428": {
        "480p": 40,
        "720p": 60,
        "1080p": 90,
    },
    "seedance-1-0-lite-i2v-250428": {
        "480p": 40,
        "720p": 60,
        "1080p": 90,
    },
    "seedance-1-0-pro-250528": {
        "480p": 70,
        "720p": 85,
        "1080p": 115,
    },
}


def get_image_url_from_response(response: ImageTaskCreationResponse) -> str:
    if response.error:
        error_msg = f"ByteDance request failed. Code: {response.error['code']}, message: {response.error['message']}"
        logging.info(error_msg)
        raise RuntimeError(error_msg)
    logging.info("ByteDance task succeeded, image URL: %s", response.data[0]["url"])
    return response.data[0]["url"]


def get_video_url_from_task_status(response: TaskStatusResponse) -> Union[str, None]:
    """Returns the video URL from the task status response if it exists."""
    if hasattr(response, "content") and response.content:
        return response.content.video_url
    return None


class ByteDanceImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceImageNode",
            display_name="ByteDance Image",
            category="api node/image/ByteDance",
            description="Generate images using ByteDance models via api based on prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=Text2ImageModelName,
                    default=Text2ImageModelName.seedream_3,
                    tooltip="Model name",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the image",
                ),
                IO.Combo.Input(
                    "size_preset",
                    options=[label for label, _, _ in RECOMMENDED_PRESETS],
                    tooltip="Pick a recommended size. Select Custom to use the width and height below",
                ),
                IO.Int.Input(
                    "width",
                    default=1024,
                    min=512,
                    max=2048,
                    step=64,
                    tooltip="Custom width for image. Value is working only if `size_preset` is set to `Custom`",
                ),
                IO.Int.Input(
                    "height",
                    default=1024,
                    min=512,
                    max=2048,
                    step=64,
                    tooltip="Custom height for image. Value is working only if `size_preset` is set to `Custom`",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation",
                    optional=True,
                ),
                IO.Float.Input(
                    "guidance_scale",
                    default=2.5,
                    min=1.0,
                    max=10.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Higher value makes the image follow the prompt more closely",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip='Whether to add an "AI generated" watermark to the image',
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
        model: str,
        prompt: str,
        size_preset: str,
        width: int,
        height: int,
        seed: int,
        guidance_scale: float,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        w = h = None
        for label, tw, th in RECOMMENDED_PRESETS:
            if label == size_preset:
                w, h = tw, th
                break

        if w is None or h is None:
            w, h = width, height
            if not (512 <= w <= 2048) or not (512 <= h <= 2048):
                raise ValueError(
                    f"Custom size out of range: {w}x{h}. " "Both width and height must be between 512 and 2048 pixels."
                )

        payload = Text2ImageTaskCreationRequest(
            model=model,
            prompt=prompt,
            size=f"{w}x{h}",
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=watermark,
        )
        response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_IMAGE_ENDPOINT, method="POST"),
            data=payload,
            response_model=ImageTaskCreationResponse,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))


class ByteDanceImageEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceImageEditNode",
            display_name="ByteDance Image Edit",
            category="api node/image/ByteDance",
            description="Edit images using ByteDance models via api based on prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=Image2ImageModelName,
                    default=Image2ImageModelName.seededit_3,
                    tooltip="Model name",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="The base image to edit",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Instruction to edit image",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation",
                    optional=True,
                ),
                IO.Float.Input(
                    "guidance_scale",
                    default=5.5,
                    min=1.0,
                    max=10.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Higher value makes the image follow the prompt more closely",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip='Whether to add an "AI generated" watermark to the image',
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
        model: str,
        image: torch.Tensor,
        prompt: str,
        seed: int,
        guidance_scale: float,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        validate_image_aspect_ratio(image, (1, 3), (3, 1))
        source_url = (await upload_images_to_comfyapi(cls, image, max_images=1, mime_type="image/png"))[0]
        payload = Image2ImageTaskCreationRequest(
            model=model,
            prompt=prompt,
            image=source_url,
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=watermark,
        )
        response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_IMAGE_ENDPOINT, method="POST"),
            data=payload,
            response_model=ImageTaskCreationResponse,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))


class ByteDanceSeedreamNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceSeedreamNode",
            display_name="ByteDance Seedream 4",
            category="api node/image/ByteDance",
            description="Unified text-to-image generation and precise single-sentence editing at up to 4K resolution.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["seedream-4-0-250828"],
                    tooltip="Model name",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for creating or editing an image.",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Input image(s) for image-to-image generation. "
                    "List of 1-10 images for single or multi-reference generation.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "size_preset",
                    options=[label for label, _, _ in RECOMMENDED_PRESETS_SEEDREAM_4],
                    tooltip="Pick a recommended size. Select Custom to use the width and height below.",
                ),
                IO.Int.Input(
                    "width",
                    default=2048,
                    min=1024,
                    max=4096,
                    step=64,
                    tooltip="Custom width for image. Value is working only if `size_preset` is set to `Custom`",
                    optional=True,
                ),
                IO.Int.Input(
                    "height",
                    default=2048,
                    min=1024,
                    max=4096,
                    step=64,
                    tooltip="Custom height for image. Value is working only if `size_preset` is set to `Custom`",
                    optional=True,
                ),
                IO.Combo.Input(
                    "sequential_image_generation",
                    options=["disabled", "auto"],
                    tooltip="Group image generation mode. "
                    "'disabled' generates a single image. "
                    "'auto' lets the model decide whether to generate multiple related images "
                    "(e.g., story scenes, character variations).",
                    optional=True,
                ),
                IO.Int.Input(
                    "max_images",
                    default=1,
                    min=1,
                    max=15,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Maximum number of images to generate when sequential_image_generation='auto'. "
                    "Total images (input + generated) cannot exceed 15.",
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
                    "watermark",
                    default=True,
                    tooltip='Whether to add an "AI generated" watermark to the image.',
                    optional=True,
                ),
                IO.Boolean.Input(
                    "fail_on_partial",
                    default=True,
                    tooltip="If enabled, abort execution if any requested images are missing or return an error.",
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
        model: str,
        prompt: str,
        image: torch.Tensor = None,
        size_preset: str = RECOMMENDED_PRESETS_SEEDREAM_4[0][0],
        width: int = 2048,
        height: int = 2048,
        sequential_image_generation: str = "disabled",
        max_images: int = 1,
        seed: int = 0,
        watermark: bool = True,
        fail_on_partial: bool = True,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        w = h = None
        for label, tw, th in RECOMMENDED_PRESETS_SEEDREAM_4:
            if label == size_preset:
                w, h = tw, th
                break

        if w is None or h is None:
            w, h = width, height
            if not (1024 <= w <= 4096) or not (1024 <= h <= 4096):
                raise ValueError(
                    f"Custom size out of range: {w}x{h}. " "Both width and height must be between 1024 and 4096 pixels."
                )
        n_input_images = get_number_of_images(image) if image is not None else 0
        if n_input_images > 10:
            raise ValueError(f"Maximum of 10 reference images are supported, but {n_input_images} received.")
        if sequential_image_generation == "auto" and n_input_images + max_images > 15:
            raise ValueError(
                "The maximum number of generated images plus the number of reference images cannot exceed 15."
            )
        reference_images_urls = []
        if n_input_images:
            for i in image:
                validate_image_aspect_ratio(i, (1, 3), (3, 1))
            reference_images_urls = await upload_images_to_comfyapi(
                cls,
                image,
                max_images=n_input_images,
                mime_type="image/png",
            )
        response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_IMAGE_ENDPOINT, method="POST"),
            response_model=ImageTaskCreationResponse,
            data=Seedream4TaskCreationRequest(
                model=model,
                prompt=prompt,
                image=reference_images_urls,
                size=f"{w}x{h}",
                seed=seed,
                sequential_image_generation=sequential_image_generation,
                sequential_image_generation_options=Seedream4Options(max_images=max_images),
                watermark=watermark,
            ),
        )
        if len(response.data) == 1:
            return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))
        urls = [str(d["url"]) for d in response.data if isinstance(d, dict) and "url" in d]
        if fail_on_partial and len(urls) < len(response.data):
            raise RuntimeError(f"Only {len(urls)} of {len(response.data)} images were generated before error.")
        return IO.NodeOutput(torch.cat([await download_url_to_image_tensor(i) for i in urls]))


class ByteDanceTextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceTextToVideoNode",
            display_name="ByteDance Text to Video",
            category="api node/video/ByteDance",
            description="Generate video using ByteDance models via api based on prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=Text2VideoModelName,
                    default=Text2VideoModelName.seedance_1_pro,
                    tooltip="Model name",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
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
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "camera_fixed",
                    default=False,
                    tooltip="Specifies whether to fix the camera. The platform appends an instruction "
                    "to fix the camera to your prompt, but does not guarantee the actual effect.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip='Whether to add an "AI generated" watermark to the video.',
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
        model: str,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])

        prompt = (
            f"{prompt} "
            f"--resolution {resolution} "
            f"--ratio {aspect_ratio} "
            f"--duration {duration} "
            f"--seed {seed} "
            f"--camerafixed {str(camera_fixed).lower()} "
            f"--watermark {str(watermark).lower()}"
        )
        return await process_video_task(
            cls,
            payload=Text2VideoTaskCreationRequest(model=model, content=[TaskTextContent(text=prompt)]),
            estimated_duration=max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model][resolution] * (duration / 10.0))),
        )


class ByteDanceImageToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceImageToVideoNode",
            display_name="ByteDance Image to Video",
            category="api node/video/ByteDance",
            description="Generate video using ByteDance models via api based on image and prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=Image2VideoModelName,
                    default=Image2VideoModelName.seedance_1_pro,
                    tooltip="Model name",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="First frame to be used for the video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
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
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "camera_fixed",
                    default=False,
                    tooltip="Specifies whether to fix the camera. The platform appends an instruction "
                    "to fix the camera to your prompt, but does not guarantee the actual effect.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip='Whether to add an "AI generated" watermark to the video.',
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
        model: str,
        prompt: str,
        image: torch.Tensor,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])
        validate_image_dimensions(image, min_width=300, min_height=300, max_width=6000, max_height=6000)
        validate_image_aspect_ratio(image, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5

        image_url = (await upload_images_to_comfyapi(cls, image, max_images=1))[0]
        prompt = (
            f"{prompt} "
            f"--resolution {resolution} "
            f"--ratio {aspect_ratio} "
            f"--duration {duration} "
            f"--seed {seed} "
            f"--camerafixed {str(camera_fixed).lower()} "
            f"--watermark {str(watermark).lower()}"
        )

        return await process_video_task(
            cls,
            payload=Image2VideoTaskCreationRequest(
                model=model,
                content=[TaskTextContent(text=prompt), TaskImageContent(image_url=TaskImageContentUrl(url=image_url))],
            ),
            estimated_duration=max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model][resolution] * (duration / 10.0))),
        )


class ByteDanceFirstLastFrameNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceFirstLastFrameNode",
            display_name="ByteDance First-Last-Frame to Video",
            category="api node/video/ByteDance",
            description="Generate video using prompt and first and last frames.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=[model.value for model in Image2VideoModelName],
                    default=Image2VideoModelName.seedance_1_lite.value,
                    tooltip="Model name",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Image.Input(
                    "first_frame",
                    tooltip="First frame to be used for the video.",
                ),
                IO.Image.Input(
                    "last_frame",
                    tooltip="Last frame to be used for the video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
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
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "camera_fixed",
                    default=False,
                    tooltip="Specifies whether to fix the camera. The platform appends an instruction "
                    "to fix the camera to your prompt, but does not guarantee the actual effect.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip='Whether to add an "AI generated" watermark to the video.',
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
        model: str,
        prompt: str,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])
        for i in (first_frame, last_frame):
            validate_image_dimensions(i, min_width=300, min_height=300, max_width=6000, max_height=6000)
            validate_image_aspect_ratio(i, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5

        download_urls = await upload_images_to_comfyapi(
            cls,
            image_tensor_pair_to_batch(first_frame, last_frame),
            max_images=2,
            mime_type="image/png",
        )

        prompt = (
            f"{prompt} "
            f"--resolution {resolution} "
            f"--ratio {aspect_ratio} "
            f"--duration {duration} "
            f"--seed {seed} "
            f"--camerafixed {str(camera_fixed).lower()} "
            f"--watermark {str(watermark).lower()}"
        )

        return await process_video_task(
            cls,
            payload=Image2VideoTaskCreationRequest(
                model=model,
                content=[
                    TaskTextContent(text=prompt),
                    TaskImageContent(image_url=TaskImageContentUrl(url=str(download_urls[0])), role="first_frame"),
                    TaskImageContent(image_url=TaskImageContentUrl(url=str(download_urls[1])), role="last_frame"),
                ],
            ),
            estimated_duration=max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model][resolution] * (duration / 10.0))),
        )


class ByteDanceImageReferenceNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceImageReferenceNode",
            display_name="ByteDance Reference Images to Video",
            category="api node/video/ByteDance",
            description="Generate video using prompt and reference images.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=[Image2VideoModelName.seedance_1_lite.value],
                    default=Image2VideoModelName.seedance_1_lite.value,
                    tooltip="Model name",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Image.Input(
                    "images",
                    tooltip="One to four images.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
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
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip='Whether to add an "AI generated" watermark to the video.',
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
        model: str,
        prompt: str,
        images: torch.Tensor,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "watermark"])
        for image in images:
            validate_image_dimensions(image, min_width=300, min_height=300, max_width=6000, max_height=6000)
            validate_image_aspect_ratio(image, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5

        image_urls = await upload_images_to_comfyapi(cls, images, max_images=4, mime_type="image/png")
        prompt = (
            f"{prompt} "
            f"--resolution {resolution} "
            f"--ratio {aspect_ratio} "
            f"--duration {duration} "
            f"--seed {seed} "
            f"--watermark {str(watermark).lower()}"
        )
        x = [
            TaskTextContent(text=prompt),
            *[TaskImageContent(image_url=TaskImageContentUrl(url=str(i)), role="reference_image") for i in image_urls],
        ]
        return await process_video_task(
            cls,
            payload=Image2VideoTaskCreationRequest(model=model, content=x),
            estimated_duration=max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model][resolution] * (duration / 10.0))),
        )


async def process_video_task(
    cls: type[IO.ComfyNode],
    payload: Union[Text2VideoTaskCreationRequest, Image2VideoTaskCreationRequest],
    estimated_duration: Optional[int],
) -> IO.NodeOutput:
    initial_response = await sync_op(
        cls,
        ApiEndpoint(path=BYTEPLUS_TASK_ENDPOINT, method="POST"),
        data=payload,
        response_model=TaskCreationResponse,
    )
    response = await poll_op(
        cls,
        ApiEndpoint(path=f"{BYTEPLUS_TASK_STATUS_ENDPOINT}/{initial_response.id}"),
        status_extractor=lambda r: r.status,
        estimated_duration=estimated_duration,
        response_model=TaskStatusResponse,
    )
    return IO.NodeOutput(await download_url_to_video_output(get_video_url_from_task_status(response)))


def raise_if_text_params(prompt: str, text_params: list[str]) -> None:
    for i in text_params:
        if f"--{i} " in prompt:
            raise ValueError(
                f"--{i} is not allowed in the prompt, use the appropriated widget input to change this value."
            )


class ByteDanceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ByteDanceImageNode,
            ByteDanceImageEditNode,
            ByteDanceSeedreamNode,
            ByteDanceTextToVideoNode,
            ByteDanceImageToVideoNode,
            ByteDanceFirstLastFrameNode,
            ByteDanceImageReferenceNode,
        ]


async def comfy_entrypoint() -> ByteDanceExtension:
    return ByteDanceExtension()

import logging
from enum import Enum
from typing import Optional
from typing_extensions import override

import torch
from pydantic import BaseModel, Field

from comfy_api.latest import ComfyExtension, io as comfy_io
from comfy_api_nodes.util.validation_utils import (
    validate_image_aspect_ratio_range,
    get_number_of_images,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
)
from comfy_api_nodes.apinode_utils import download_url_to_image_tensor, upload_images_to_comfyapi, validate_string


BYTEPLUS_ENDPOINT = "/proxy/byteplus/api/v3/images/generations"


class Text2ImageModelName(str, Enum):
    seedream3 = "seedream-3-0-t2i-250415"


class Image2ImageModelName(str, Enum):
    seededit3 = "seededit-3-0-i2i-250628"


class Text2ImageTaskCreationRequest(BaseModel):
    model: Text2ImageModelName = Text2ImageModelName.seedream3
    prompt: str = Field(...)
    response_format: Optional[str] = Field("url")
    size: Optional[str] = Field(None)
    seed: Optional[int] = Field(0, ge=0, le=2147483647)
    guidance_scale: Optional[float] = Field(..., ge=1.0, le=10.0)
    watermark: Optional[bool] = Field(True)


class Image2ImageTaskCreationRequest(BaseModel):
    model: Image2ImageModelName = Image2ImageModelName.seededit3
    prompt: str = Field(...)
    response_format: Optional[str] = Field("url")
    image: str = Field(..., description="Base64 encoded string or image URL")
    size: Optional[str] = Field("adaptive")
    seed: Optional[int] = Field(..., ge=0, le=2147483647)
    guidance_scale: Optional[float] = Field(..., ge=1.0, le=10.0)
    watermark: Optional[bool] = Field(True)


class ImageTaskCreationResponse(BaseModel):
    model: str = Field(...)
    created: int = Field(..., description="Unix timestamp (in seconds) indicating time when the request was created.")
    data: list = Field([], description="Contains information about the generated image(s).")
    error: dict = Field({}, description="Contains `code` and `message` fields in case of error.")


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


def get_image_url_from_response(response: ImageTaskCreationResponse) -> str:
    if response.error:
        error_msg = f"ByteDance request failed. Code: {response.error['code']}, message: {response.error['message']}"
        logging.info(error_msg)
        raise RuntimeError(error_msg)
    logging.info("ByteDance task succeeded, image URL: %s", response.data[0]["url"])
    return response.data[0]["url"]


class ByteDanceImageNode(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="ByteDanceImageNode",
            display_name="ByteDance Image",
            category="api node/image/ByteDance",
            description="Generate images using ByteDance models via api based on prompt",
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=[model.value for model in Text2ImageModelName],
                    default=Text2ImageModelName.seedream3.value,
                    tooltip="Model name",
                ),
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the image",
                ),
                comfy_io.Combo.Input(
                    "size_preset",
                    options=[label for label, _, _ in RECOMMENDED_PRESETS],
                    tooltip="Pick a recommended size. Select Custom to use the width and height below",
                ),
                comfy_io.Int.Input(
                    "width",
                    default=1024,
                    min=512,
                    max=2048,
                    step=64,
                    tooltip="Custom width for image. Value is working only if `size_preset` is set to `Custom`",
                ),
                comfy_io.Int.Input(
                    "height",
                    default=1024,
                    min=512,
                    max=2048,
                    step=64,
                    tooltip="Custom height for image. Value is working only if `size_preset` is set to `Custom`",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation",
                    optional=True,
                ),
                comfy_io.Float.Input(
                    "guidance_scale",
                    default=2.5,
                    min=1.0,
                    max=10.0,
                    step=0.01,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Higher value makes the image follow the prompt more closely",
                    optional=True,
                ),
                comfy_io.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip="Whether to add an \"AI generated\" watermark to the image",
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
        size_preset: str,
        width: int,
        height: int,
        seed: int,
        guidance_scale: float,
        watermark: bool,
    ) -> comfy_io.NodeOutput:
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
                    f"Custom size out of range: {w}x{h}. "
                    "Both width and height must be between 512 and 2048 pixels."
                )

        payload = Text2ImageTaskCreationRequest(
            model=model,
            prompt=prompt,
            size=f"{w}x{h}",
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=watermark,
        )
        auth_kwargs = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        response = await SynchronousOperation(
            endpoint=ApiEndpoint(
                path=BYTEPLUS_ENDPOINT,
                method=HttpMethod.POST,
                request_model=Text2ImageTaskCreationRequest,
                response_model=ImageTaskCreationResponse,
            ),
            request=payload,
            auth_kwargs=auth_kwargs,
        ).execute()
        return comfy_io.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))


class ByteDanceImageEditNode(comfy_io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="ByteDanceImageEditNode",
            display_name="ByteDance Image Edit",
            category="api node/video/ByteDance",
            description="Edit images using ByteDance models via api based on prompt",
            inputs=[
                comfy_io.Combo.Input(
                    "model",
                    options=[model.value for model in Image2ImageModelName],
                    default=Image2ImageModelName.seededit3.value,
                    tooltip="Model name",
                ),
                comfy_io.Image.Input(
                    "image",
                    tooltip="The base image to edit",
                ),
                comfy_io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Instruction to edit image",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation",
                    optional=True,
                ),
                comfy_io.Float.Input(
                    "guidance_scale",
                    default=5.5,
                    min=1.0,
                    max=10.0,
                    step=0.01,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Higher value makes the image follow the prompt more closely",
                    optional=True,
                ),
                comfy_io.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip="Whether to add an \"AI generated\" watermark to the image",
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
        image: torch.Tensor,
        prompt: str,
        seed: int,
        guidance_scale: float,
        watermark: bool,
    ) -> comfy_io.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        validate_image_aspect_ratio_range(image, (1, 3), (3, 1))
        auth_kwargs = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        source_url = (await upload_images_to_comfyapi(
            image,
            max_images=1,
            mime_type="image/png",
            auth_kwargs=auth_kwargs,
        ))[0]
        payload = Image2ImageTaskCreationRequest(
            model=model,
            prompt=prompt,
            image=source_url,
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=watermark,
        )
        response = await SynchronousOperation(
            endpoint=ApiEndpoint(
                path=BYTEPLUS_ENDPOINT,
                method=HttpMethod.POST,
                request_model=Image2ImageTaskCreationRequest,
                response_model=ImageTaskCreationResponse,
            ),
            request=payload,
            auth_kwargs=auth_kwargs,
        ).execute()
        return comfy_io.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))


class ByteDanceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[comfy_io.ComfyNode]]:
        return [
            ByteDanceImageNode,
            ByteDanceImageEditNode,
        ]

async def comfy_entrypoint() -> ByteDanceExtension:
    return ByteDanceExtension()

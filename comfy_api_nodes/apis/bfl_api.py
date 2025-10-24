from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, confloat, conint


class BFLOutputFormat(str, Enum):
    png = 'png'
    jpeg = 'jpeg'


class BFLFluxExpandImageRequest(BaseModel):
    prompt: str = Field(..., description='The description of the changes you want to make. This text guides the expansion process, allowing you to specify features, styles, or modifications for the expanded areas.')
    prompt_upsampling: Optional[bool] = Field(
        None, description='Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation.'
    )
    seed: Optional[int] = Field(None, description='The seed value for reproducibility.')
    top: conint(ge=0, le=2048) = Field(..., description='Number of pixels to expand at the top of the image')
    bottom: conint(ge=0, le=2048) = Field(..., description='Number of pixels to expand at the bottom of the image')
    left: conint(ge=0, le=2048) = Field(..., description='Number of pixels to expand at the left side of the image')
    right: conint(ge=0, le=2048) = Field(..., description='Number of pixels to expand at the right side of the image')
    steps: conint(ge=15, le=50) = Field(..., description='Number of steps for the image generation process')
    guidance: confloat(ge=1.5, le=100) = Field(..., description='Guidance strength for the image generation process')
    safety_tolerance: Optional[conint(ge=0, le=6)] = Field(
        6, description='Tolerance level for input and output moderation. Between 0 and 6, 0 being most strict, 6 being least strict. Defaults to 2.'
    )
    output_format: Optional[BFLOutputFormat] = Field(
        BFLOutputFormat.png, description="Output format for the generated image. Can be 'jpeg' or 'png'.", examples=['png']
    )
    image: str = Field(None, description='A Base64-encoded string representing the image you wish to expand')


class BFLFluxFillImageRequest(BaseModel):
    prompt: str = Field(..., description='The description of the changes you want to make. This text guides the expansion process, allowing you to specify features, styles, or modifications for the expanded areas.')
    prompt_upsampling: Optional[bool] = Field(
        None, description='Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation.'
    )
    seed: Optional[int] = Field(None, description='The seed value for reproducibility.')
    steps: conint(ge=15, le=50) = Field(..., description='Number of steps for the image generation process')
    guidance: confloat(ge=1.5, le=100) = Field(..., description='Guidance strength for the image generation process')
    safety_tolerance: Optional[conint(ge=0, le=6)] = Field(
        6, description='Tolerance level for input and output moderation. Between 0 and 6, 0 being most strict, 6 being least strict. Defaults to 2.'
    )
    output_format: Optional[BFLOutputFormat] = Field(
        BFLOutputFormat.png, description="Output format for the generated image. Can be 'jpeg' or 'png'.", examples=['png']
    )
    image: str = Field(None, description='A Base64-encoded string representing the image you wish to modify. Can contain alpha mask if desired.')
    mask: str = Field(None, description='A Base64-encoded string representing the mask of the areas you with to modify.')


class BFLFluxProGenerateRequest(BaseModel):
    prompt: str = Field(..., description='The text prompt for image generation.')
    prompt_upsampling: Optional[bool] = Field(
        None, description='Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation.'
    )
    seed: Optional[int] = Field(None, description='The seed value for reproducibility.')
    width: conint(ge=256, le=1440) = Field(1024, description='Width of the generated image in pixels. Must be a multiple of 32.')
    height: conint(ge=256, le=1440) = Field(768, description='Height of the generated image in pixels. Must be a multiple of 32.')
    safety_tolerance: Optional[conint(ge=0, le=6)] = Field(
        6, description='Tolerance level for input and output moderation. Between 0 and 6, 0 being most strict, 6 being least strict. Defaults to 2.'
    )
    output_format: Optional[BFLOutputFormat] = Field(
        BFLOutputFormat.png, description="Output format for the generated image. Can be 'jpeg' or 'png'.", examples=['png']
    )
    image_prompt: Optional[str] = Field(None, description='Optional image to remix in base64 format')
    # image_prompt_strength: Optional[confloat(ge=0.0, le=1.0)] = Field(
    #     None, description='Blend between the prompt and the image prompt.'
    # )


class BFLFluxKontextProGenerateRequest(BaseModel):
    prompt: str = Field(..., description='The text prompt for what you wannt to edit.')
    input_image: Optional[str] = Field(None, description='Image to edit in base64 format')
    seed: Optional[int] = Field(None, description='The seed value for reproducibility.')
    guidance: confloat(ge=0.1, le=99.0) = Field(..., description='Guidance strength for the image generation process')
    steps: conint(ge=1, le=150) = Field(..., description='Number of steps for the image generation process')
    safety_tolerance: Optional[conint(ge=0, le=2)] = Field(
        2, description='Tolerance level for input and output moderation. Between 0 and 2, 0 being most strict, 6 being least strict. Defaults to 2.'
    )
    output_format: Optional[BFLOutputFormat] = Field(
        BFLOutputFormat.png, description="Output format for the generated image. Can be 'jpeg' or 'png'.", examples=['png']
    )
    aspect_ratio: Optional[str] = Field(None, description='Aspect ratio of the image between 21:9 and 9:21.')
    prompt_upsampling: Optional[bool] = Field(
        None, description='Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation.'
    )


class BFLFluxProUltraGenerateRequest(BaseModel):
    prompt: str = Field(..., description='The text prompt for image generation.')
    prompt_upsampling: Optional[bool] = Field(
        None, description='Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation.'
    )
    seed: Optional[int] = Field(None, description='The seed value for reproducibility.')
    aspect_ratio: Optional[str] = Field(None, description='Aspect ratio of the image between 21:9 and 9:21.')
    safety_tolerance: Optional[conint(ge=0, le=6)] = Field(
        6, description='Tolerance level for input and output moderation. Between 0 and 6, 0 being most strict, 6 being least strict. Defaults to 2.'
    )
    output_format: Optional[BFLOutputFormat] = Field(
        BFLOutputFormat.png, description="Output format for the generated image. Can be 'jpeg' or 'png'.", examples=['png']
    )
    raw: Optional[bool] = Field(None, description='Generate less processed, more natural-looking images.')
    image_prompt: Optional[str] = Field(None, description='Optional image to remix in base64 format')
    image_prompt_strength: Optional[confloat(ge=0.0, le=1.0)] = Field(
        None, description='Blend between the prompt and the image prompt.'
    )


class BFLFluxProGenerateResponse(BaseModel):
    id: str = Field(..., description='The unique identifier for the generation task.')
    polling_url: str = Field(..., description='URL to poll for the generation result.')


class BFLStatus(str, Enum):
    task_not_found = "Task not found"
    pending = "Pending"
    request_moderated = "Request Moderated"
    content_moderated = "Content Moderated"
    ready = "Ready"
    error = "Error"


class BFLFluxStatusResponse(BaseModel):
    id: str = Field(..., description="The unique identifier for the generation task.")
    status: BFLStatus = Field(..., description="The status of the task.")
    result: Optional[Dict[str, Any]] = Field(None, description="The result of the task (null if not completed).")
    progress: Optional[float] = Field(None, description="The progress of the task (0.0 to 1.0).", ge=0.0, le=1.0)

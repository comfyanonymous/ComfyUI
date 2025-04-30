from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, confloat, conint


class BFLOutputFormat(str, Enum):
    png = 'png'
    jpeg = 'jpeg'


class BFLFluxProGenerateRequest(BaseModel):
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


class BFLFluxProStatusResponse(BaseModel):
    id: str = Field(..., description="The unique identifier for the generation task.")
    status: BFLStatus = Field(..., description="The status of the task.")
    result: Optional[Dict[str, Any]] = Field(
        None, description="The result of the task (null if not completed)."
    )
    progress: confloat(ge=0.0, le=1.0) = Field(
        ..., description="The progress of the task (0.0 to 1.0)."
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional details about the task (null if not available)."
    )

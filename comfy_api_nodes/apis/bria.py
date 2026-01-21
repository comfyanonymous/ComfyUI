from typing import TypedDict

from pydantic import BaseModel, Field


class InputModerationSettings(TypedDict):
    prompt_content_moderation: bool
    visual_input_moderation: bool
    visual_output_moderation: bool


class BriaEditImageRequest(BaseModel):
    instruction: str | None = Field(...)
    structured_instruction: str | None = Field(
        ...,
        description="Use this instead of instruction for precise, programmatic control.",
    )
    images: list[str] = Field(
        ...,
        description="Required. Publicly available URL or Base64-encoded. Must contain exactly one item.",
    )
    mask: str | None = Field(
        None,
        description="Mask image (black and white). Black areas will be preserved, white areas will be edited. "
        "If omitted, the edit applies to the entire image. "
        "The input image and the the input mask must be of the same size.",
    )
    negative_prompt: str | None = Field(None)
    guidance_scale: float = Field(...)
    model_version: str = Field(...)
    steps_num: int = Field(...)
    seed: int = Field(...)
    ip_signal: bool = Field(
        False,
        description="If true, returns a warning for potential IP content in the instruction.",
    )
    prompt_content_moderation: bool = Field(
        False, description="If true, returns 422 on instruction moderation failure."
    )
    visual_input_content_moderation: bool = Field(
        False, description="If true, returns 422 on images or mask moderation failure."
    )
    visual_output_content_moderation: bool = Field(
        False, description="If true, returns 422 on visual output moderation failure."
    )


class BriaStatusResponse(BaseModel):
    request_id: str = Field(...)
    status_url: str = Field(...)
    warning: str | None = Field(None)


class BriaResult(BaseModel):
    structured_prompt: str = Field(...)
    image_url: str = Field(...)


class BriaResponse(BaseModel):
    status: str = Field(...)
    result: BriaResult | None = Field(None)

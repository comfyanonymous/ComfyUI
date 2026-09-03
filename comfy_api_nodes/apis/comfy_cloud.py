from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Only the workflows this build ships. The backend serves more.
ComfyCloudWorkflow = Literal[
    "default/text-to-image",
    "default/text-to-video",
    "default/image-to-video",
    "default/image-edit",
    "krea-2/text-to-image",
    "qwen-image-edit-2511/image-edit",
    "seedvr2/upscale-image",
    "z-image-turbo/text-to-image",
    "flux-2/text-to-image",
    "ideogram-4/text-to-image",
    "longcat/text-to-image",
    "capybara-0.1/text-to-image",
    "minimax-h3/text-to-video",
    "minimax-h3/image-to-video",
    "ltx-2.3/image-to-video",
    "wan-2.2/first-last-frame-to-video",
]


# Only the inputs the shipped nodes send. The manifests accept more.
class ComfyCloudWorkflowInputs(BaseModel):
    prompt: str | None = Field(None)
    image_url: str | None = Field(None)
    assets: dict[str, "ComfyCloudAssetInput"] | None = Field(None)
    audio_url: str | None = Field(None)
    first_frame_url: str | None = Field(None)
    last_frame_url: str | None = Field(None)
    instruction: str | None = Field(None)
    prompt_enhance: bool | None = Field(None)
    enhance_prompt: bool | None = Field(None)
    negative_prompt: str | None = Field(None)
    aspect_ratio: str | None = Field(None)
    duration_seconds: float | None = Field(None)
    quality_mode: str | None = Field(None)
    seed: int | None = Field(None, ge=0, le=0xFFFFFFFFFFFFFFFF)
    scale: str | None = Field(None)
    width: int | None = Field(None, ge=256, le=2048)
    height: int | None = Field(None, ge=256, le=2048)
    resolution: str | None = Field(None)
    rendering_speed: str | None = Field(None)
    color_correction: str | None = Field(None)
    turbo: bool | None = Field(None)
    style_lora: bool | None = Field(None)
    steps: int | None = Field(None, ge=1)
    turbo_steps: int | None = Field(None, ge=1)
    fast_steps: int | None = Field(None, ge=1)
    cfg: float | None = Field(None, ge=0)
    guidance: float | None = Field(None, ge=0)
    shift: float | None = Field(None, ge=0)
    turbo_strength: float | None = Field(None, ge=0)
    style_strength: float | None = Field(None, ge=0)


class ComfyCloudAssetInput(BaseModel):
    type: Literal["IMAGE", "VIDEO", "AUDIO"] = Field(...)
    url: str = Field(...)


class ComfyCloudGenerateRequest(BaseModel):
    workflow: ComfyCloudWorkflow = Field(...)
    inputs: ComfyCloudWorkflowInputs = Field(...)


class ComfyCloudGenerateResponse(BaseModel):
    task_id: str = Field(..., min_length=1)
    status: str = Field(...)
    polling_url: str | None = Field(None)
    cancel_url: str | None = Field(None)

    @field_validator("task_id")
    @classmethod
    def task_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task_id must not be blank")
        return value


class ComfyCloudStatusResponse(BaseModel):
    task_id: str = Field(..., min_length=1)
    status: str = Field(...)
    progress: float | None = Field(None)
    output_url: str | None = Field(None)
    error: str | None = Field(None)

    @field_validator("task_id")
    @classmethod
    def task_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task_id must not be blank")
        return value

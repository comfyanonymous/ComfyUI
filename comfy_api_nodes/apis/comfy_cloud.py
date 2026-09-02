from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Only the workflows this build actually ships. The backend serves more; a node must not
# reference one that has not been vetted for the client surface.
ComfyCloudWorkflow = Literal[
    "text-to-image",
    "text-to-video",
    "image-to-video",
    "image-edit",
    "image.krea-2-creative-image.v1",
    "image.qwen-image-edit-2511.v1",
    "image.seedvr2-image-upscale.v1",
    "image.z-image-turbo.v1",
    "image.flux-2-text-to-image.v1",
    "image.ideogram-4-text-to-image.v1",
    "image.longcat-text-to-image.v1",
    "image.capybara-0-1-text-to-image.v1",
    "video.minimax-h3-text-sound.v1",
    "video.minimax-h3-image-sound.v1",
    "video.ltx-2-3-image-audio-performance.v1",
    "video.wan-2-2-14b-first-last-frame.v1",
]


# Only the inputs the shipped nodes actually send, for the same reason
# ComfyCloudWorkflow is narrowed above: the frozen manifests accept more, but a field
# no node sets is surface this build cannot exercise or test.
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


class ComfyCloudAssetInput(BaseModel):
    type: Literal["IMAGE", "VIDEO", "AUDIO"] = Field(...)
    url: str = Field(...)


class ComfyCloudGenerateRequest(BaseModel):
    workflow: ComfyCloudWorkflow = Field(...)
    inputs: ComfyCloudWorkflowInputs = Field(...)


class ComfyCloudGenerateResponse(BaseModel):
    task_id: str = Field(..., min_length=1)
    status: str = Field(...)
    polling_url: str = Field(...)
    cancel_url: str = Field(...)

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
    output_urls: dict[str, str] | None = Field(None)
    error: str | None = Field(None)

    @field_validator("task_id")
    @classmethod
    def task_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task_id must not be blank")
        return value

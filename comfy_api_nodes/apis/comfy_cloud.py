from typing import Literal

from pydantic import BaseModel, Field, field_validator


ComfyCloudWorkflow = Literal[
    "text-to-image",
    "text-to-video",
    "image-to-video",
    "image-edit",
    "image.ideogram-4-design.v1",
    "image.krea-2-creative-image.v1",
    "image.mage-flow-image.v1",
    "image.flux-2-reference-edit.v1",
    "image.qwen-image-edit-2511.v1",
    "image.seedvr2-image-upscale.v1",
    "video.minimax-h3-text-sound.v1",
    "video.minimax-h3-image-sound.v1",
    "video.ltx-2-3-image-audio-performance.v1",
    "video.ltx-2-3-first-last-frame.v1",
    "video.wan-2-2-14b-first-last-frame.v1",
    "video.scail-2-character-replacement.v1",
    "audio.ace-step-1-5-xl-turbo.v1",
    "audio.stable-audio-3-medium.v1",
    "audio.chatterbox-multilingual-voice-clone.v1",
    "audio.chatterbox-dialogue.v1",
    "audio.chatterbox-voice-conversion.v1",
    "audio.melbandroformer-stem-separation.v1",
    "3d.triposplat-image-to-gaussian-splat.v1",
    "3d.hunyuan3d-2-1-image-to-3d.v1",
    "3d.hunyuan3d-multiview-to-3d.v1",
    "3d.moge-2-photo-to-textured-mesh.v1",
    "3d.moge-2-panorama-to-3d-scene.v1",
]


class ComfyCloudWorkflowInputs(BaseModel):
    prompt: str | None = Field(None)
    image_url: str | None = Field(None)
    assets: dict[str, "ComfyCloudAssetInput"] | None = Field(None)
    audio_url: str | None = Field(None)
    first_frame_url: str | None = Field(None)
    last_frame_url: str | None = Field(None)
    reference_character_url: str | None = Field(None)
    driving_video_url: str | None = Field(None)
    instruction: str | None = Field(None)
    prompt_enhance: bool | None = Field(None)
    enhance_prompt: bool | None = Field(None)
    negative_prompt: str | None = Field(None)
    aspect_ratio: str | None = Field(None)
    duration_seconds: float | None = Field(None)
    guidance: float | None = Field(None)
    quality_mode: str | None = Field(None)
    seed: int | None = Field(None, ge=0, le=0xFFFFFFFFFFFFFFFF)
    scale: str | None = Field(None)
    scene_prompt: str | None = Field(None)
    driving_subject: str | None = Field(None)
    reference_subject: str | None = Field(None)
    style_prompt: str | None = Field(None)
    lyrics: str | None = Field(None)
    bpm: int | None = Field(None)
    time_signature: str | None = Field(None)
    language: str | None = Field(None)
    key: str | None = Field(None)
    expand_prompt: bool | None = Field(None)
    category: str | None = Field(None)
    text: str | None = Field(None)
    exaggeration: float | None = Field(None)
    cfg_weight: float | None = Field(None)
    temperature: float | None = Field(None)
    script: str | None = Field(None)
    remove_background: bool | None = Field(None)
    gaussian_count: int | None = Field(None)
    fov_degrees: float | None = Field(None)
    detail: int | None = Field(None)
    mesh_decimation: int | None = Field(None)
    gap_threshold: float | None = Field(None)
    texture: bool | None = Field(None)
    split_resolution: int | None = Field(None)
    merge_resolution: int | None = Field(None)


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

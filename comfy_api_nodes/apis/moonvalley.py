from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, StrictBytes


class MoonvalleyPromptResponse(BaseModel):
    error: Optional[Dict[str, Any]] = None
    frame_conditioning: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    inference_params: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    model_params: Optional[Dict[str, Any]] = None
    output_url: Optional[str] = None
    prompt_text: Optional[str] = None
    status: Optional[str] = None


class MoonvalleyTextToVideoInferenceParams(BaseModel):
    add_quality_guidance: Optional[bool] = Field(
        True, description='Whether to add quality guidance'
    )
    caching_coefficient: Optional[float] = Field(
        0.3, description='Caching coefficient for optimization'
    )
    caching_cooldown: Optional[int] = Field(
        3, description='Number of caching cooldown steps'
    )
    caching_warmup: Optional[int] = Field(
        3, description='Number of caching warmup steps'
    )
    clip_value: Optional[float] = Field(
        3, description='CLIP value for generation control'
    )
    conditioning_frame_index: Optional[int] = Field(
        0, description='Index of the conditioning frame'
    )
    cooldown_steps: Optional[int] = Field(
        75, description='Number of cooldown steps (calculated based on num_frames)'
    )
    fps: Optional[int] = Field(
        24, description='Frames per second of the generated video'
    )
    guidance_scale: Optional[float] = Field(
        10, description='Guidance scale for generation control'
    )
    height: Optional[int] = Field(
        1080, description='Height of the generated video in pixels'
    )
    negative_prompt: Optional[str] = Field(None, description='Negative prompt text')
    num_frames: Optional[int] = Field(64, description='Number of frames to generate')
    seed: Optional[int] = Field(
        None, description='Random seed for generation (default: random)'
    )
    shift_value: Optional[float] = Field(
        3, description='Shift value for generation control'
    )
    steps: Optional[int] = Field(80, description='Number of denoising steps')
    use_guidance_schedule: Optional[bool] = Field(
        True, description='Whether to use guidance scheduling'
    )
    use_negative_prompts: Optional[bool] = Field(
        False, description='Whether to use negative prompts'
    )
    use_timestep_transform: Optional[bool] = Field(
        True, description='Whether to use timestep transformation'
    )
    warmup_steps: Optional[int] = Field(
        0, description='Number of warmup steps (calculated based on num_frames)'
    )
    width: Optional[int] = Field(
        1920, description='Width of the generated video in pixels'
    )


class MoonvalleyTextToVideoRequest(BaseModel):
    image_url: Optional[str] = None
    inference_params: Optional[MoonvalleyTextToVideoInferenceParams] = None
    prompt_text: Optional[str] = None
    webhook_url: Optional[str] = None


class MoonvalleyUploadFileRequest(BaseModel):
    file: Optional[StrictBytes] = None


class MoonvalleyUploadFileResponse(BaseModel):
    access_url: Optional[str] = None


class MoonvalleyVideoToVideoInferenceParams(BaseModel):
    add_quality_guidance: Optional[bool] = Field(
        True, description='Whether to add quality guidance'
    )
    caching_coefficient: Optional[float] = Field(
        0.3, description='Caching coefficient for optimization'
    )
    caching_cooldown: Optional[int] = Field(
        3, description='Number of caching cooldown steps'
    )
    caching_warmup: Optional[int] = Field(
        3, description='Number of caching warmup steps'
    )
    clip_value: Optional[float] = Field(
        3, description='CLIP value for generation control'
    )
    conditioning_frame_index: Optional[int] = Field(
        0, description='Index of the conditioning frame'
    )
    cooldown_steps: Optional[int] = Field(
        36, description='Number of cooldown steps (calculated based on num_frames)'
    )
    guidance_scale: Optional[float] = Field(
        15, description='Guidance scale for generation control'
    )
    negative_prompt: Optional[str] = Field(None, description='Negative prompt text')
    seed: Optional[int] = Field(
        None, description='Random seed for generation (default: random)'
    )
    shift_value: Optional[float] = Field(
        3, description='Shift value for generation control'
    )
    steps: Optional[int] = Field(80, description='Number of denoising steps')
    use_guidance_schedule: Optional[bool] = Field(
        True, description='Whether to use guidance scheduling'
    )
    use_negative_prompts: Optional[bool] = Field(
        False, description='Whether to use negative prompts'
    )
    use_timestep_transform: Optional[bool] = Field(
        True, description='Whether to use timestep transformation'
    )
    warmup_steps: Optional[int] = Field(
        24, description='Number of warmup steps (calculated based on num_frames)'
    )


class ControlType(str, Enum):
    motion_control = 'motion_control'
    pose_control = 'pose_control'


class MoonvalleyVideoToVideoRequest(BaseModel):
    control_type: ControlType = Field(
        ..., description='Supported types for video control'
    )
    inference_params: Optional[MoonvalleyVideoToVideoInferenceParams] = None
    prompt_text: str = Field(..., description='Describes the video to generate')
    video_url: str = Field(..., description='Url to control video')
    webhook_url: Optional[str] = Field(
        None, description='Optional webhook URL for notifications'
    )

from pydantic import BaseModel, Field


class MultiPromptEntry(BaseModel):
    index: int = Field(...)
    prompt: str = Field(...)
    duration: str = Field(...)


class OmniProText2VideoRequest(BaseModel):
    model_name: str = Field(..., description="kling-video-o1")
    aspect_ratio: str = Field(..., description="'16:9', '9:16' or '1:1'")
    duration: str = Field(..., description="'5' or '10'")
    prompt: str = Field(...)
    mode: str = Field("pro")
    multi_shot: bool | None = Field(None)
    multi_prompt: list[MultiPromptEntry] | None = Field(None)
    shot_type: str | None = Field(None)
    sound: str = Field(..., description="'on' or 'off'")


class OmniParamImage(BaseModel):
    image_url: str = Field(...)
    type: str | None = Field(None, description="Can be 'first_frame' or 'end_frame'")


class OmniParamVideo(BaseModel):
    video_url: str = Field(...)
    refer_type: str | None = Field(..., description="Can be 'base' or 'feature'")
    keep_original_sound: str = Field(..., description="'yes' or 'no'")


class OmniProFirstLastFrameRequest(BaseModel):
    model_name: str = Field(..., description="kling-video-o1")
    image_list: list[OmniParamImage] = Field(..., min_length=1, max_length=7)
    duration: str = Field(..., description="'5' or '10'")
    prompt: str = Field(...)
    mode: str = Field("pro")
    sound: str | None = Field(None, description="'on' or 'off'")
    multi_shot: bool | None = Field(None)
    multi_prompt: list[MultiPromptEntry] | None = Field(None)
    shot_type: str | None = Field(None)


class OmniProReferences2VideoRequest(BaseModel):
    model_name: str = Field(..., description="kling-video-o1")
    aspect_ratio: str | None = Field(..., description="'16:9', '9:16' or '1:1'")
    image_list: list[OmniParamImage] | None = Field(
        None, max_length=7, description="Max length 4 when video is present."
    )
    video_list: list[OmniParamVideo] | None = Field(None, max_length=1)
    duration: str | None = Field(..., description="From 3 to 10.")
    prompt: str = Field(...)
    mode: str = Field("pro")
    sound: str | None = Field(None, description="'on' or 'off'")
    multi_shot: bool | None = Field(None)
    multi_prompt: list[MultiPromptEntry] | None = Field(None)
    shot_type: str | None = Field(None)


class TaskStatusVideoResult(BaseModel):
    duration: str | None = Field(None, description="Total video duration")
    id: str | None = Field(None, description="Generated video ID")
    url: str | None = Field(None, description="URL for generated video")


class TaskStatusImageResult(BaseModel):
    index: int = Field(..., description="Image Number，0-9")
    url: str = Field(..., description="URL for generated image")


class TaskStatusResults(BaseModel):
    videos: list[TaskStatusVideoResult] | None = Field(None)
    images: list[TaskStatusImageResult] | None = Field(None)
    series_images: list[TaskStatusImageResult] | None = Field(None)


class TaskStatusResponseData(BaseModel):
    created_at: int | None = Field(None, description="Task creation time")
    updated_at: int | None = Field(None, description="Task update time")
    task_status: str | None = None
    task_status_msg: str | None = Field(None, description="Additional failure reason. Only for polling endpoint.")
    task_id: str | None = Field(None, description="Task ID")
    task_result: TaskStatusResults | None = Field(None)


class TaskStatusResponse(BaseModel):
    code: int | None = Field(None, description="Error code")
    message: str | None = Field(None, description="Error message")
    request_id: str | None = Field(None, description="Request ID")
    data: TaskStatusResponseData | None = Field(None)


class OmniImageParamImage(BaseModel):
    image: str = Field(...)


class OmniProImageRequest(BaseModel):
    model_name: str = Field(...)
    resolution: str = Field(...)
    aspect_ratio: str | None = Field(...)
    prompt: str = Field(...)
    mode: str = Field("pro")
    n: int | None = Field(1, le=9)
    image_list: list[OmniImageParamImage] | None = Field(..., max_length=10)
    result_type: str | None = Field(None, description="Set to 'series' for series generation")
    series_amount: int | None = Field(None, ge=2, le=9, description="Number of images in a series")


class TextToVideoWithAudioRequest(BaseModel):
    model_name: str = Field(...)
    aspect_ratio: str = Field(..., description="'16:9', '9:16' or '1:1'")
    duration: str = Field(...)
    prompt: str | None = Field(...)
    negative_prompt: str | None = Field(None)
    mode: str = Field("pro")
    sound: str = Field(..., description="'on' or 'off'")
    multi_shot: bool | None = Field(None)
    multi_prompt: list[MultiPromptEntry] | None = Field(None)
    shot_type: str | None = Field(None)


class ImageToVideoWithAudioRequest(BaseModel):
    model_name: str = Field(...)
    image: str = Field(...)
    image_tail: str | None = Field(None)
    duration: str = Field(...)
    prompt: str | None = Field(...)
    negative_prompt: str | None = Field(None)
    mode: str = Field("pro")
    sound: str = Field(..., description="'on' or 'off'")
    multi_shot: bool | None = Field(None)
    multi_prompt: list[MultiPromptEntry] | None = Field(None)
    shot_type: str | None = Field(None)


class KlingAvatarRequest(BaseModel):
    image: str = Field(...)
    sound_file: str = Field(...)
    prompt: str | None = Field(None)
    mode: str = Field(...)


class MotionControlRequest(BaseModel):
    prompt: str = Field(...)
    image_url: str = Field(...)
    video_url: str = Field(...)
    keep_original_sound: str = Field(...)
    character_orientation: str = Field(...)
    mode: str = Field(..., description="'pro' or 'std'")

from pydantic import BaseModel, Field


class OmniProText2VideoRequest(BaseModel):
    model_name: str = Field(..., description="kling-video-o1")
    aspect_ratio: str = Field(..., description="'16:9', '9:16' or '1:1'")
    duration: str = Field(..., description="'5' or '10'")
    prompt: str = Field(...)
    mode: str = Field("pro")


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


class TaskStatusVideoResult(BaseModel):
    duration: str | None = Field(None, description="Total video duration")
    id: str | None = Field(None, description="Generated video ID")
    url: str | None = Field(None, description="URL for generated video")


class TaskStatusVideoResults(BaseModel):
    videos: list[TaskStatusVideoResult] | None = Field(None)


class TaskStatusVideoResponseData(BaseModel):
    created_at: int | None = Field(None, description="Task creation time")
    updated_at: int | None = Field(None, description="Task update time")
    task_status: str | None = None
    task_status_msg: str | None = Field(None, description="Additional failure reason. Only for polling endpoint.")
    task_id: str | None = Field(None, description="Task ID")
    task_result: TaskStatusVideoResults | None = Field(None)


class TaskStatusVideoResponse(BaseModel):
    code: int | None = Field(None, description="Error code")
    message: str | None = Field(None, description="Error message")
    request_id: str | None = Field(None, description="Request ID")
    data: TaskStatusVideoResponseData | None = Field(None)

from typing import TypedDict

from pydantic import BaseModel, Field


class InputVideoModel(TypedDict):
    model: str
    resolution: str


class ImageEnhanceTaskCreateRequest(BaseModel):
    model_name: str = Field(...)
    img_url: str = Field(...)
    extension: str = Field(".png")
    exif: bool = Field(False)
    DPI: int | None = Field(None)


class VideoEnhanceTaskCreateRequest(BaseModel):
    video_url: str = Field(...)
    extension: str = Field(".mp4")
    model_name: str | None = Field(...)
    resolution: list[int] = Field(..., description="Target resolution [width, height]")
    original_resolution: list[int] = Field(..., description="Original video resolution [width, height]")


class TaskCreateDataResponse(BaseModel):
    job_id: str = Field(...)
    consume_coins: int | None = Field(None)


class TaskStatusPollRequest(BaseModel):
    job_id: str = Field(...)


class TaskCreateResponse(BaseModel):
    code: int = Field(...)
    message: str = Field(...)
    data: TaskCreateDataResponse | None = Field(None)


class TaskStatusDataResponse(BaseModel):
    job_id: str = Field(...)
    status: str = Field(...)
    res_url: str = Field("")


class TaskStatusResponse(BaseModel):
    code: int = Field(...)
    message: str = Field(...)
    data: TaskStatusDataResponse = Field(...)

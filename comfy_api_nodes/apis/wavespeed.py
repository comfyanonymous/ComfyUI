from typing import Optional

from pydantic import BaseModel, Field


class WavespeedTextToImageRequest(BaseModel):
    prompt: str = Field(...)
    width: int = Field(1024)
    height: int = Field(1024)
    num_inference_steps: int = Field(28)
    guidance_scale: float = Field(3.5)
    seed: int = Field(-1)
    enable_safety_checker: bool = Field(True)
    enable_sync_mode: bool = Field(False)
    num_images: int = Field(1)
    output_format: str = Field("jpeg")
    negative_prompt: Optional[str] = Field(None)


class SeedVR2ImageRequest(BaseModel):
    image: str = Field(...)
    target_resolution: str = Field(...)
    output_format: str = Field("png")
    enable_sync_mode: bool = Field(False)


class FlashVSRRequest(BaseModel):
    target_resolution: str = Field(...)
    video: str = Field(...)
    duration: float = Field(...)


class TaskCreatedDataResponse(BaseModel):
    id: str = Field(...)


class TaskCreatedResponse(BaseModel):
    code: int = Field(...)
    message: str = Field(...)
    data: TaskCreatedDataResponse | None = Field(None)


class TaskResultDataResponse(BaseModel):
    status: str = Field(...)
    outputs: list[str] = Field([])


class TaskResultResponse(BaseModel):
    code: int = Field(...)
    message: str = Field(...)
    data: TaskResultDataResponse | None = Field(None)

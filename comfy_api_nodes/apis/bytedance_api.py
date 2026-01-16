from typing import Literal

from pydantic import BaseModel, Field


class Text2ImageTaskCreationRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(...)
    response_format: str | None = Field("url")
    size: str | None = Field(None)
    seed: int | None = Field(0, ge=0, le=2147483647)
    guidance_scale: float | None = Field(..., ge=1.0, le=10.0)
    watermark: bool | None = Field(False)


class Image2ImageTaskCreationRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(...)
    response_format: str | None = Field("url")
    image: str = Field(..., description="Base64 encoded string or image URL")
    size: str | None = Field("adaptive")
    seed: int | None = Field(..., ge=0, le=2147483647)
    guidance_scale: float | None = Field(..., ge=1.0, le=10.0)
    watermark: bool | None = Field(False)


class Seedream4Options(BaseModel):
    max_images: int = Field(15)


class Seedream4TaskCreationRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(...)
    response_format: str = Field("url")
    image: list[str] | None = Field(None, description="Image URLs")
    size: str = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    sequential_image_generation: str = Field("disabled")
    sequential_image_generation_options: Seedream4Options = Field(Seedream4Options(max_images=15))
    watermark: bool = Field(False)


class ImageTaskCreationResponse(BaseModel):
    model: str = Field(...)
    created: int = Field(..., description="Unix timestamp (in seconds) indicating time when the request was created.")
    data: list = Field([], description="Contains information about the generated image(s).")
    error: dict = Field({}, description="Contains `code` and `message` fields in case of error.")


class TaskTextContent(BaseModel):
    type: str = Field("text")
    text: str = Field(...)


class TaskImageContentUrl(BaseModel):
    url: str = Field(...)


class TaskImageContent(BaseModel):
    type: str = Field("image_url")
    image_url: TaskImageContentUrl = Field(...)
    role: Literal["first_frame", "last_frame", "reference_image"] | None = Field(None)


class Text2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    content: list[TaskTextContent] = Field(..., min_length=1)
    generate_audio: bool | None = Field(...)


class Image2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    content: list[TaskTextContent | TaskImageContent] = Field(..., min_length=2)
    generate_audio: bool | None = Field(...)


class TaskCreationResponse(BaseModel):
    id: str = Field(...)


class TaskStatusError(BaseModel):
    code: str = Field(...)
    message: str = Field(...)


class TaskStatusResult(BaseModel):
    video_url: str = Field(...)


class TaskStatusResponse(BaseModel):
    id: str = Field(...)
    model: str = Field(...)
    status: Literal["queued", "running", "cancelled", "succeeded", "failed"] = Field(...)
    error: TaskStatusError | None = Field(None)
    content: TaskStatusResult | None = Field(None)


RECOMMENDED_PRESETS = [
    ("1024x1024 (1:1)", 1024, 1024),
    ("864x1152 (3:4)", 864, 1152),
    ("1152x864 (4:3)", 1152, 864),
    ("1280x720 (16:9)", 1280, 720),
    ("720x1280 (9:16)", 720, 1280),
    ("832x1248 (2:3)", 832, 1248),
    ("1248x832 (3:2)", 1248, 832),
    ("1512x648 (21:9)", 1512, 648),
    ("2048x2048 (1:1)", 2048, 2048),
    ("Custom", None, None),
]

RECOMMENDED_PRESETS_SEEDREAM_4 = [
    ("2048x2048 (1:1)", 2048, 2048),
    ("2304x1728 (4:3)", 2304, 1728),
    ("1728x2304 (3:4)", 1728, 2304),
    ("2560x1440 (16:9)", 2560, 1440),
    ("1440x2560 (9:16)", 1440, 2560),
    ("2496x1664 (3:2)", 2496, 1664),
    ("1664x2496 (2:3)", 1664, 2496),
    ("3024x1296 (21:9)", 3024, 1296),
    ("4096x4096 (1:1)", 4096, 4096),
    ("Custom", None, None),
]

# The time in this dictionary are given for 10 seconds duration.
VIDEO_TASKS_EXECUTION_TIME = {
    "seedance-1-0-lite-t2v-250428": {
        "480p": 40,
        "720p": 60,
        "1080p": 90,
    },
    "seedance-1-0-lite-i2v-250428": {
        "480p": 40,
        "720p": 60,
        "1080p": 90,
    },
    "seedance-1-0-pro-250528": {
        "480p": 70,
        "720p": 85,
        "1080p": 115,
    },
    "seedance-1-0-pro-fast-251015": {
        "480p": 50,
        "720p": 65,
        "1080p": 100,
    },
    "seedance-1-5-pro-251215": {
        "480p": 80,
        "720p": 100,
        "1080p": 150,
    },
}

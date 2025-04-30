from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


pixverse_templates = {
    "Microwave": 324641385496960,
    "Suit Swagger": 328545151283968,
    "Anything, Robot": 313358700761536,
    "Subject 3 Fever": 327828816843648,
    "kiss kiss": 315446315336768,
}


class PixverseIO:
    TEMPLATE = "PIXVERSE_TEMPLATE"


class PixverseStatus(int, Enum):
    successful = 1
    generating = 5
    contents_moderation = 7
    failed = 8


class PixverseAspectRatio(str, Enum):
    ratio_16_9 = "16:9"
    ratio_4_3 = "4:3"
    ratio_1_1 = "1:1"
    ratio_3_4 = "3:4"
    ratio_9_16 = "9:16"


class PixverseQuality(str, Enum):
    res_360p = "360p"
    res_540p = "540p"
    res_720p = "720p"
    res_1080p = "1080p"


class PixverseDuration(int, Enum):
    dur_5 = 5
    dur_8 = 8


class PixverseMotionMode(str, Enum):
    normal = "normal"
    fast = "fast"


class PixverseStyle(str, Enum):
    anime = "anime"
    animation_3d = "3d_animation"
    clay = "clay"
    comic = "comic"
    cyberpunk = "cyberpunk"


# NOTE: forgoing descriptions for now in return for dev speed
class PixverseDto_V2OpenAPIT2VReq(BaseModel):
    aspect_ratio: PixverseAspectRatio = Field(...)
    quality: PixverseQuality = Field(...)
    duration: PixverseDuration = Field(...)
    model: Optional[str] = Field("v3.5")
    motion_mode: Optional[PixverseMotionMode] = Field(PixverseMotionMode.normal)
    prompt: str = Field(...)
    negative_prompt: Optional[str] = Field(None)
    seed: Optional[int] = Field(None)
    style: Optional[str] = Field(None)
    template_id: Optional[int] = Field(None)
    water_mark: Optional[bool] = Field(None)


class PixverseController_ResponseData(BaseModel):
    ErrCode: Optional[int] = Field(None)
    ErrMsg: Optional[str] = Field(None)
    Resp: Optional[PixverseDto_V2OpenAPII2VResp] = Field(None)


class PixverseDto_V2OpenAPII2VResp(BaseModel):
    video_id: int = Field(..., description='Video_id')


class PixverseGenerationStatusResponse(BaseModel):
    ErrCode: Optional[int] = Field(None)
    ErrMsg: Optional[str] = Field(None)
    Resp: Optional[PixverseDto_GetOpenapiMediaDetailResp] = Field(None)


class PixverseDto_GetOpenapiMediaDetailResp(BaseModel):
    create_time: Optional[str] = Field(None)
    id: Optional[int] = Field(None)
    modify_time: Optional[str] = Field(None)
    negative_prompt: Optional[str] = Field(None)
    outputHeight: Optional[int] = Field(None)
    outputWidth: Optional[int] = Field(None)
    prompt: Optional[str] = Field(None)
    resolution_ratio: Optional[int] = Field(None)
    seed: Optional[int] = Field(None)
    size: Optional[int] = Field(None)
    status: Optional[int] = Field(None)
    style: Optional[str] = Field(None)
    url: Optional[str] = Field(None)

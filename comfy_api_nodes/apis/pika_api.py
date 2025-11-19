from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class Pikaffect(str, Enum):
    Cake_ify = "Cake-ify"
    Crumble = "Crumble"
    Crush = "Crush"
    Decapitate = "Decapitate"
    Deflate = "Deflate"
    Dissolve = "Dissolve"
    Explode = "Explode"
    Eye_pop = "Eye-pop"
    Inflate = "Inflate"
    Levitate = "Levitate"
    Melt = "Melt"
    Peel = "Peel"
    Poke = "Poke"
    Squish = "Squish"
    Ta_da = "Ta-da"
    Tear = "Tear"


class PikaBodyGenerate22C2vGenerate22PikascenesPost(BaseModel):
    aspectRatio: Optional[float] = Field(None, description='Aspect ratio (width / height)')
    duration: Optional[int] = Field(5)
    ingredientsMode: str = Field(...)
    negativePrompt: Optional[str] = Field(None)
    promptText: Optional[str] = Field(None)
    resolution: Optional[str] = Field('1080p')
    seed: Optional[int] = Field(None)


class PikaGenerateResponse(BaseModel):
    video_id: str = Field(...)


class PikaBodyGenerate22I2vGenerate22I2vPost(BaseModel):
    duration: Optional[int] = 5
    negativePrompt: Optional[str] = Field(None)
    promptText: Optional[str] = Field(None)
    resolution: Optional[str] = '1080p'
    seed: Optional[int] = Field(None)


class PikaBodyGenerate22KeyframeGenerate22PikaframesPost(BaseModel):
    duration: Optional[int] = Field(None, ge=5, le=10)
    negativePrompt: Optional[str] = Field(None)
    promptText: str = Field(...)
    resolution: Optional[str] = '1080p'
    seed: Optional[int] = Field(None)


class PikaBodyGenerate22T2vGenerate22T2vPost(BaseModel):
    aspectRatio: Optional[float] = Field(
        1.7777777777777777,
        description='Aspect ratio (width / height)',
        ge=0.4,
        le=2.5,
    )
    duration: Optional[int] = 5
    negativePrompt: Optional[str] = Field(None)
    promptText: str = Field(...)
    resolution: Optional[str] = '1080p'
    seed: Optional[int] = Field(None)


class PikaBodyGeneratePikadditionsGeneratePikadditionsPost(BaseModel):
    negativePrompt: Optional[str] = Field(None)
    promptText: Optional[str] = Field(None)
    seed: Optional[int] = Field(None)


class PikaBodyGeneratePikaffectsGeneratePikaffectsPost(BaseModel):
    negativePrompt: Optional[str] = Field(None)
    pikaffect: Optional[str] = None
    promptText: Optional[str] = Field(None)
    seed: Optional[int] = Field(None)


class PikaBodyGeneratePikaswapsGeneratePikaswapsPost(BaseModel):
    negativePrompt: Optional[str] = Field(None)
    promptText: Optional[str] = Field(None)
    seed: Optional[int] = Field(None)
    modifyRegionRoi: Optional[str] = Field(None)


class PikaStatusEnum(str, Enum):
    queued = "queued"
    started = "started"
    finished = "finished"
    failed = "failed"


class PikaVideoResponse(BaseModel):
    id: str = Field(...)
    progress: Optional[int] = Field(None)
    status: PikaStatusEnum
    url: Optional[str] = Field(None)

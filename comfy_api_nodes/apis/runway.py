from enum import Enum
from typing import Optional, List, Union
from datetime import datetime

from pydantic import BaseModel, Field, RootModel


class RunwayAspectRatioEnum(str, Enum):
    field_1280_720 = '1280:720'
    field_720_1280 = '720:1280'
    field_1104_832 = '1104:832'
    field_832_1104 = '832:1104'
    field_960_960 = '960:960'
    field_1584_672 = '1584:672'
    field_1280_768 = '1280:768'
    field_768_1280 = '768:1280'


class Position(str, Enum):
    first = 'first'
    last = 'last'


class RunwayPromptImageDetailedObject(BaseModel):
    position: Position = Field(
        ...,
        description="The position of the image in the output video. 'last' is currently supported for gen3a_turbo only.",
    )
    uri: str = Field(
        ..., description='A HTTPS URL or data URI containing an encoded image.'
    )


class RunwayPromptImageObject(
    RootModel[Union[str, List[RunwayPromptImageDetailedObject]]]
):
    root: Union[str, List[RunwayPromptImageDetailedObject]] = Field(
        ...,
        description='Image(s) to use for the video generation. Can be a single URI or an array of image objects with positions.',
    )


class RunwayModelEnum(str, Enum):
    gen4_turbo = 'gen4_turbo'
    gen3a_turbo = 'gen3a_turbo'


class RunwayDurationEnum(int, Enum):
    integer_5 = 5
    integer_10 = 10


class RunwayImageToVideoRequest(BaseModel):
    duration: RunwayDurationEnum
    model: RunwayModelEnum
    promptImage: RunwayPromptImageObject
    promptText: Optional[str] = Field(
        None, description='Text prompt for the generation', max_length=1000
    )
    ratio: RunwayAspectRatioEnum
    seed: int = Field(
        ..., description='Random seed for generation', ge=0, le=4294967295
    )


class RunwayImageToVideoResponse(BaseModel):
    id: Optional[str] = Field(None, description='Task ID')


class RunwayTaskStatusEnum(str, Enum):
    SUCCEEDED = 'SUCCEEDED'
    RUNNING = 'RUNNING'
    FAILED = 'FAILED'
    PENDING = 'PENDING'
    CANCELLED = 'CANCELLED'
    THROTTLED = 'THROTTLED'


class RunwayTaskStatusResponse(BaseModel):
    createdAt: datetime = Field(..., description='Task creation timestamp')
    id: str = Field(..., description='Task ID')
    output: Optional[List[str]] = Field(None, description='Array of output video URLs')
    progress: Optional[float] = Field(
        None,
        description='Float value between 0 and 1 representing the progress of the task. Only available if status is RUNNING.',
        ge=0.0,
        le=1.0,
    )
    status: RunwayTaskStatusEnum


class Model4(str, Enum):
    gen4_image = 'gen4_image'


class ReferenceImage(BaseModel):
    uri: Optional[str] = Field(
        None, description='A HTTPS URL or data URI containing an encoded image'
    )


class RunwayTextToImageAspectRatioEnum(str, Enum):
    field_1920_1080 = '1920:1080'
    field_1080_1920 = '1080:1920'
    field_1024_1024 = '1024:1024'
    field_1360_768 = '1360:768'
    field_1080_1080 = '1080:1080'
    field_1168_880 = '1168:880'
    field_1440_1080 = '1440:1080'
    field_1080_1440 = '1080:1440'
    field_1808_768 = '1808:768'
    field_2112_912 = '2112:912'


class RunwayTextToImageRequest(BaseModel):
    model: Model4 = Field(..., description='Model to use for generation')
    promptText: str = Field(
        ..., description='Text prompt for the image generation', max_length=1000
    )
    ratio: RunwayTextToImageAspectRatioEnum
    referenceImages: Optional[List[ReferenceImage]] = Field(
        None, description='Array of reference images to guide the generation'
    )


class RunwayTextToImageResponse(BaseModel):
    id: Optional[str] = Field(None, description='Task ID')

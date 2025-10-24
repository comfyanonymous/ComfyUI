from typing import Optional, Union
from enum import Enum

from pydantic import BaseModel, Field


class Image2(BaseModel):
    bytesBase64Encoded: str
    gcsUri: Optional[str] = None
    mimeType: Optional[str] = None


class Image3(BaseModel):
    bytesBase64Encoded: Optional[str] = None
    gcsUri: str
    mimeType: Optional[str] = None


class Instance1(BaseModel):
    image: Optional[Union[Image2, Image3]] = Field(
        None, description='Optional image to guide video generation'
    )
    prompt: str = Field(..., description='Text description of the video')


class PersonGeneration1(str, Enum):
    ALLOW = 'ALLOW'
    BLOCK = 'BLOCK'


class Parameters1(BaseModel):
    aspectRatio: Optional[str] = Field(None, examples=['16:9'])
    durationSeconds: Optional[int] = None
    enhancePrompt: Optional[bool] = None
    generateAudio: Optional[bool] = Field(
        None,
        description='Generate audio for the video. Only supported by veo 3 models.',
    )
    negativePrompt: Optional[str] = None
    personGeneration: Optional[PersonGeneration1] = None
    sampleCount: Optional[int] = None
    seed: Optional[int] = None
    storageUri: Optional[str] = Field(
        None, description='Optional Cloud Storage URI to upload the video'
    )


class VeoGenVidRequest(BaseModel):
    instances: Optional[list[Instance1]] = None
    parameters: Optional[Parameters1] = None


class VeoGenVidResponse(BaseModel):
    name: str = Field(
        ...,
        description='Operation resource name',
        examples=[
            'projects/PROJECT_ID/locations/us-central1/publishers/google/models/MODEL_ID/operations/a1b07c8e-7b5a-4aba-bb34-3e1ccb8afcc8'
        ],
    )


class VeoGenVidPollRequest(BaseModel):
    operationName: str = Field(
        ...,
        description='Full operation name (from predict response)',
        examples=[
            'projects/PROJECT_ID/locations/us-central1/publishers/google/models/MODEL_ID/operations/OPERATION_ID'
        ],
    )


class Video(BaseModel):
    bytesBase64Encoded: Optional[str] = Field(
        None, description='Base64-encoded video content'
    )
    gcsUri: Optional[str] = Field(None, description='Cloud Storage URI of the video')
    mimeType: Optional[str] = Field(None, description='Video MIME type')


class Error1(BaseModel):
    code: Optional[int] = Field(None, description='Error code')
    message: Optional[str] = Field(None, description='Error message')


class Response1(BaseModel):
    field_type: Optional[str] = Field(
        None,
        alias='@type',
        examples=[
            'type.googleapis.com/cloud.ai.large_models.vision.GenerateVideoResponse'
        ],
    )
    raiMediaFilteredCount: Optional[int] = Field(
        None, description='Count of media filtered by responsible AI policies'
    )
    raiMediaFilteredReasons: Optional[list[str]] = Field(
        None, description='Reasons why media was filtered by responsible AI policies'
    )
    videos: Optional[list[Video]] = None


class VeoGenVidPollResponse(BaseModel):
    done: Optional[bool] = None
    error: Optional[Error1] = Field(
        None, description='Error details if operation failed'
    )
    name: Optional[str] = None
    response: Optional[Response1] = Field(
        None, description='The actual prediction response if done is true'
    )

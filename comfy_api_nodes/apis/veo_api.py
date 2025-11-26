from typing import Optional

from pydantic import BaseModel, Field


class VeoRequestInstanceImage(BaseModel):
    bytesBase64Encoded: str | None = Field(None)
    gcsUri: str | None = Field(None)
    mimeType: str | None = Field(None)


class VeoRequestInstance(BaseModel):
    image: VeoRequestInstanceImage | None = Field(None)
    lastFrame: VeoRequestInstanceImage | None = Field(None)
    prompt: str = Field(..., description='Text description of the video')


class VeoRequestParameters(BaseModel):
    aspectRatio: Optional[str] = Field(None, examples=['16:9'])
    durationSeconds: Optional[int] = None
    enhancePrompt: Optional[bool] = None
    generateAudio: Optional[bool] = Field(
        None,
        description='Generate audio for the video. Only supported by veo 3 models.',
    )
    negativePrompt: Optional[str] = None
    personGeneration: str | None = Field(None, description="ALLOW or BLOCK")
    sampleCount: Optional[int] = None
    seed: Optional[int] = None
    storageUri: Optional[str] = Field(
        None, description='Optional Cloud Storage URI to upload the video'
    )
    resolution: str | None = Field(None)


class VeoGenVidRequest(BaseModel):
    instances: list[VeoRequestInstance] | None = Field(None)
    parameters: VeoRequestParameters | None = Field(None)


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

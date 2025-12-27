from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MinimaxBaseResponse(BaseModel):
    status_code: int = Field(
        ...,
        description='Status code. 0 indicates success, other values indicate errors.',
    )
    status_msg: str = Field(
        ..., description='Specific error details or success message.'
    )


class File(BaseModel):
    bytes: Optional[int] = Field(None, description='File size in bytes')
    created_at: Optional[int] = Field(
        None, description='Unix timestamp when the file was created, in seconds'
    )
    download_url: Optional[str] = Field(
        None, description='The URL to download the video'
    )
    backup_download_url: Optional[str] = Field(
        None, description='The backup URL to download the video'
    )

    file_id: Optional[int] = Field(None, description='Unique identifier for the file')
    filename: Optional[str] = Field(None, description='The name of the file')
    purpose: Optional[str] = Field(None, description='The purpose of using the file')


class MinimaxFileRetrieveResponse(BaseModel):
    base_resp: MinimaxBaseResponse
    file: File


class MiniMaxModel(str, Enum):
    T2V_01_Director = 'T2V-01-Director'
    I2V_01_Director = 'I2V-01-Director'
    S2V_01 = 'S2V-01'
    I2V_01 = 'I2V-01'
    I2V_01_live = 'I2V-01-live'
    T2V_01 = 'T2V-01'
    Hailuo_02 = 'MiniMax-Hailuo-02'


class Status6(str, Enum):
    Queueing = 'Queueing'
    Preparing = 'Preparing'
    Processing = 'Processing'
    Success = 'Success'
    Fail = 'Fail'


class MinimaxTaskResultResponse(BaseModel):
    base_resp: MinimaxBaseResponse
    file_id: Optional[str] = Field(
        None,
        description='After the task status changes to Success, this field returns the file ID corresponding to the generated video.',
    )
    status: Status6 = Field(
        ...,
        description="Task status: 'Queueing' (in queue), 'Preparing' (task is preparing), 'Processing' (generating), 'Success' (task completed successfully), or 'Fail' (task failed).",
    )
    task_id: str = Field(..., description='The task ID being queried.')


class SubjectReferenceItem(BaseModel):
    image: Optional[str] = Field(
        None, description='URL or base64 encoding of the subject reference image.'
    )
    mask: Optional[str] = Field(
        None,
        description='URL or base64 encoding of the mask for the subject reference image.',
    )


class MinimaxVideoGenerationRequest(BaseModel):
    callback_url: Optional[str] = Field(
        None,
        description='Optional. URL to receive real-time status updates about the video generation task.',
    )
    first_frame_image: Optional[str] = Field(
        None,
        description='URL or base64 encoding of the first frame image. Required when model is I2V-01, I2V-01-Director, or I2V-01-live.',
    )
    model: MiniMaxModel = Field(
        ...,
        description='Required. ID of model. Options: T2V-01-Director, I2V-01-Director, S2V-01, I2V-01, I2V-01-live, T2V-01',
    )
    prompt: Optional[str] = Field(
        None,
        description='Description of the video. Should be less than 2000 characters. Supports camera movement instructions in [brackets].',
        max_length=2000,
    )
    prompt_optimizer: Optional[bool] = Field(
        True,
        description='If true (default), the model will automatically optimize the prompt. Set to false for more precise control.',
    )
    subject_reference: Optional[list[SubjectReferenceItem]] = Field(
        None,
        description='Only available when model is S2V-01. The model will generate a video based on the subject uploaded through this parameter.',
    )
    duration: Optional[int] = Field(
        None,
        description="The length of the output video in seconds."
    )
    resolution: Optional[str] = Field(
        None,
        description="The dimensions of the video display. 1080p corresponds to 1920 x 1080 pixels, 768p corresponds to 1366 x 768 pixels."
    )


class MinimaxVideoGenerationResponse(BaseModel):
    base_resp: MinimaxBaseResponse
    task_id: str = Field(
        ..., description='The task ID for the asynchronous video generation task.'
    )

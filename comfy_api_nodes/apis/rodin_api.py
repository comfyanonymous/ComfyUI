from __future__ import annotations

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class Rodin3DGenerateRequest(BaseModel):
    seed: int = Field(..., description="seed_")
    tier: str = Field(..., description="Tier of generation.")
    material: str = Field(..., description="The material type.")
    quality_override: int = Field(..., description="The poly count of the mesh.")
    mesh_mode: str = Field(..., description="It controls the type of faces of generated models.")
    TAPose: Optional[bool] = Field(None, description="")

class GenerateJobsData(BaseModel):
    uuids: List[str] = Field(..., description="str LIST")
    subscription_key: str = Field(..., description="subscription key")

class Rodin3DGenerateResponse(BaseModel):
    message: Optional[str] = Field(None, description="Return message.")
    prompt: Optional[str] = Field(None, description="Generated Prompt from image.")
    submit_time: Optional[str] = Field(None, description="Submit Time")
    uuid: Optional[str] = Field(None, description="Task str")
    jobs: Optional[GenerateJobsData] = Field(None, description="Details of jobs")

class JobStatus(str, Enum):
    """
    Status for jobs
    """
    Done = "Done"
    Failed = "Failed"
    Generating = "Generating"
    Waiting = "Waiting"

class Rodin3DCheckStatusRequest(BaseModel):
    subscription_key: str = Field(..., description="subscription from generate endpoint")

class JobItem(BaseModel):
    uuid: str = Field(..., description="uuid")
    status: JobStatus = Field(...,description="Status Currently")

class Rodin3DCheckStatusResponse(BaseModel):
    jobs: List[JobItem] = Field(..., description="Job status List")

class Rodin3DDownloadRequest(BaseModel):
    task_uuid: str = Field(..., description="Task str")

class RodinResourceItem(BaseModel):
    url: str = Field(..., description="Download Url")
    name: str = Field(..., description="File name with ext")

class Rodin3DDownloadResponse(BaseModel):
    list: List[RodinResourceItem] = Field(..., description="Source List")

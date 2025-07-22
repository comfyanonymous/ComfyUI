from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class MeshyAIModel(str, Enum):
    meshy_4 = "meshy-4"
    meshy_5 = "meshy-5"


class MeshyTopology(str, Enum):
    quad = "quad"
    triangle = "triangle"


class MeshySymmetryMode(str, Enum):
    off = "off"
    auto = "auto"
    on = "on"


class MeshyArtStyle(str, Enum):
    realistic = "realistic"
    sculpture = "sculpture"


class MeshyTaskStatus(str, Enum):
    pending = "PENDING"
    in_progress = "IN_PROGRESS"
    completed = "SUCCEEDED"
    failed = "FAILED"
    canceled = "CANCELED"


class MeshyTask(BaseModel):
    id: str = Field(..., description="The task ID")
    status: MeshyTaskStatus = Field(..., description="The status of the task")
    progress: int = Field(..., description="The progress of the task")
    message: str = Field(..., description="The message of the task")


class MeshyTaskResponse(BaseModel):
    code: int = Field(..., description="The response code")
    data: MeshyTask = Field(..., description="The task data")


class MeshyTextToModelPreviewRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="The text prompt describing the model to generate",
        max_length=1024,
    )
    mode: str = "preview"
    art_style: Optional[MeshyArtStyle] = None
    seed: Optional[int] = None
    ai_model: Optional[MeshyAIModel] = None
    topology: Optional[MeshyTopology] = None
    target_polycount: Optional[int] = None
    should_remesh: Optional[bool] = None
    symmetry_mode: Optional[MeshySymmetryMode] = None
    should_simplify: Optional[bool] = None


class MeshyTaskResponse(BaseModel):
    model_file: str = Field(..., description="The model file")
    model_task_id: str = Field(..., description="The model task ID")


class MeshyTextToModelRefineRequest(BaseModel):
    mode: str = "refine"
    preview_task_id: str = Field(..., description="The preview task ID")
    enable_pbr: bool = Field(..., description="Whether to enable PBR")
    texture_prompt: str = Field(..., description="The texture prompt")
    ai_model: Optional[MeshyAIModel] = None


class MeshyImageToModelRequest(BaseModel):
    image_url: str = Field(..., description="The image URL")
    ai_model: Optional[MeshyAIModel] = None
    topology: Optional[MeshyTopology] = None
    target_polycount: Optional[int] = None
    symmetry_mode: Optional[MeshySymmetryMode] = None
    should_remesh: Optional[bool] = None
    should_texture: Optional[bool] = None
    enable_pbr: Optional[bool] = None
    texture_prompt: Optional[str] = None
    moderation: Optional[bool] = None

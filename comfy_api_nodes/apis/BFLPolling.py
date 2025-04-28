from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, confloat


class BFLStatus(str, Enum):
    task_not_found = "Task not found"
    pending = "Pending"
    request_moderated = "Request Moderated"
    content_moderated = "Content Moderated"
    ready = "Ready"
    error = "Error"


class BFLFluxProStatusResponse(BaseModel):
    id: str = Field(..., description="The unique identifier for the generation task.")
    status: BFLStatus = Field(..., description="The status of the task.")
    result: Optional[Dict[str, Any]] = Field(
        None, description="The result of the task (null if not completed)."
    )
    progress: confloat(ge=0.0, le=1.0) = Field(
        ..., description="The progress of the task (0.0 to 1.0)."
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional details about the task (null if not available)."
    )

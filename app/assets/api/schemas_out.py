from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class AssetSummary(BaseModel):
    id: str
    name: str
    asset_hash: Optional[str]
    size: Optional[int] = None
    mime_type: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    preview_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_access_time: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("created_at", "updated_at", "last_access_time")
    def _ser_dt(self, v: Optional[datetime], _info):
        return v.isoformat() if v else None


class AssetsList(BaseModel):
    assets: list[AssetSummary]
    total: int
    has_more: bool


class AssetUpdated(BaseModel):
    id: str
    name: str
    asset_hash: Optional[str]
    tags: list[str] = Field(default_factory=list)
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("updated_at")
    def _ser_updated(self, v: Optional[datetime], _info):
        return v.isoformat() if v else None


class AssetDetail(BaseModel):
    id: str
    name: str
    asset_hash: Optional[str]
    size: Optional[int] = None
    mime_type: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    preview_id: Optional[str] = None
    created_at: Optional[datetime] = None
    last_access_time: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("created_at", "last_access_time")
    def _ser_dt(self, v: Optional[datetime], _info):
        return v.isoformat() if v else None


class AssetCreated(AssetDetail):
    created_new: bool


class TagUsage(BaseModel):
    name: str
    count: int
    type: str


class TagsList(BaseModel):
    tags: list[TagUsage] = Field(default_factory=list)
    total: int
    has_more: bool


class TagsAdd(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    added: list[str] = Field(default_factory=list)
    already_present: list[str] = Field(default_factory=list)
    total_tags: list[str] = Field(default_factory=list)


class TagsRemove(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    removed: list[str] = Field(default_factory=list)
    not_present: list[str] = Field(default_factory=list)
    total_tags: list[str] = Field(default_factory=list)


class AssetScanError(BaseModel):
    path: str
    message: str
    at: Optional[str] = Field(None, description="ISO timestamp")


class AssetScanStatus(BaseModel):
    scan_id: str
    root: Literal["models", "input", "output"]
    status: Literal["scheduled", "running", "completed", "failed", "cancelled"]
    scheduled_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    discovered: int = 0
    processed: int = 0
    file_errors: list[AssetScanError] = Field(default_factory=list)


class AssetScanStatusResponse(BaseModel):
    scans: list[AssetScanStatus] = Field(default_factory=list)

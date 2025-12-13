from datetime import datetime
from typing import Any, Optional

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


class TagUsage(BaseModel):
    name: str
    count: int
    type: str


class TagsList(BaseModel):
    tags: list[TagUsage] = Field(default_factory=list)
    total: int
    has_more: bool

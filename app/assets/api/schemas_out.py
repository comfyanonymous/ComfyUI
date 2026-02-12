from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class AssetSummary(BaseModel):
    id: str
    name: str
    asset_hash: str | None = None
    size: int | None = None
    mime_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    preview_url: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_access_time: datetime | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("created_at", "updated_at", "last_access_time")
    def _ser_dt(self, v: datetime | None, _info):
        return v.isoformat() if v else None


class AssetsList(BaseModel):
    assets: list[AssetSummary]
    total: int
    has_more: bool


class AssetUpdated(BaseModel):
    id: str
    name: str
    asset_hash: str | None = None
    tags: list[str] = Field(default_factory=list)
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("updated_at")
    def _ser_updated(self, v: datetime | None, _info):
        return v.isoformat() if v else None


class AssetDetail(BaseModel):
    id: str
    name: str
    asset_hash: str | None = None
    size: int | None = None
    mime_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    preview_id: str | None = None
    created_at: datetime | None = None
    last_access_time: datetime | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("created_at", "last_access_time")
    def _ser_dt(self, v: datetime | None, _info):
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

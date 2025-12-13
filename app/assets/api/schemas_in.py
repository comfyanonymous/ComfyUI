import json
import uuid
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    conint,
    field_validator,
)


class ListAssetsQuery(BaseModel):
    include_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)
    name_contains: str | None = None

    # Accept either a JSON string (query param) or a dict
    metadata_filter: dict[str, Any] | None = None

    limit: conint(ge=1, le=500) = 20
    offset: conint(ge=0) = 0

    sort: Literal["name", "created_at", "updated_at", "size", "last_access_time"] = "created_at"
    order: Literal["asc", "desc"] = "desc"

    @field_validator("include_tags", "exclude_tags", mode="before")
    @classmethod
    def _split_csv_tags(cls, v):
        # Accept "a,b,c" or ["a","b"] (we are liberal in what we accept)
        if v is None:
            return []
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                if isinstance(item, str):
                    out.extend([t.strip() for t in item.split(",") if t.strip()])
            return out
        return v

    @field_validator("metadata_filter", mode="before")
    @classmethod
    def _parse_metadata_json(cls, v):
        if v is None or isinstance(v, dict):
            return v
        if isinstance(v, str) and v.strip():
            try:
                parsed = json.loads(v)
            except Exception as e:
                raise ValueError(f"metadata_filter must be JSON: {e}") from e
            if not isinstance(parsed, dict):
                raise ValueError("metadata_filter must be a JSON object")
            return parsed
        return None


class TagsListQuery(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    prefix: str | None = Field(None, min_length=1, max_length=256)
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0, le=10_000_000)
    order: Literal["count_desc", "name_asc"] = "count_desc"
    include_zero: bool = True

    @field_validator("prefix")
    @classmethod
    def normalize_prefix(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        return v.lower() or None


class SetPreviewBody(BaseModel):
    """Set or clear the preview for an AssetInfo. Provide an Asset.id or null."""
    preview_id: str | None = None

    @field_validator("preview_id", mode="before")
    @classmethod
    def _norm_uuid(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        try:
            uuid.UUID(s)
        except Exception:
            raise ValueError("preview_id must be a UUID")
        return s

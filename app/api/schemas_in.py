from __future__ import annotations

from typing import Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, conint


class ListAssetsQuery(BaseModel):
    include_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)
    name_contains: Optional[str] = None

    # Accept either a JSON string (query param) or a dict
    metadata_filter: Optional[dict[str, Any]] = None

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
            import json
            try:
                parsed = json.loads(v)
            except Exception as e:
                raise ValueError(f"metadata_filter must be JSON: {e}") from e
            if not isinstance(parsed, dict):
                raise ValueError("metadata_filter must be a JSON object")
            return parsed
        return None


class UpdateAssetBody(BaseModel):
    name: Optional[str] = None
    tags: Optional[list[str]] = None
    user_metadata: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def _at_least_one(self):
        if self.name is None and self.tags is None and self.user_metadata is None:
            raise ValueError("Provide at least one of: name, tags, user_metadata.")
        if self.tags is not None:
            if not isinstance(self.tags, list) or not all(isinstance(t, str) for t in self.tags):
                raise ValueError("Field 'tags' must be an array of strings.")
        return self

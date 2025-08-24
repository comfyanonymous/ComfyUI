import json

from typing import Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, conint


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


class CreateFromHashBody(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    hash: str
    name: str
    tags: list[str] = Field(default_factory=list)
    user_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("hash")
    @classmethod
    def _require_blake3(cls, v):
        s = (v or "").strip().lower()
        if ":" not in s:
            raise ValueError("hash must be 'blake3:<hex>'")
        algo, digest = s.split(":", 1)
        if algo != "blake3":
            raise ValueError("only canonical 'blake3:<hex>' is accepted here")
        if not digest or any(c for c in digest if c not in "0123456789abcdef"):
            raise ValueError("hash digest must be lowercase hex")
        return s

    @field_validator("tags", mode="before")
    @classmethod
    def _tags_norm(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            out = [str(t).strip().lower() for t in v if str(t).strip()]
            seen = set(); dedup = []
            for t in out:
                if t not in seen:
                    seen.add(t); dedup.append(t)
            return dedup
        if isinstance(v, str):
            return [t.strip().lower() for t in v.split(",") if t.strip()]
        return []


class TagsListQuery(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    prefix: Optional[str] = Field(None, min_length=1, max_length=256)
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0, le=10_000_000)
    order: Literal["count_desc", "name_asc"] = "count_desc"
    include_zero: bool = True

    @field_validator("prefix")
    @classmethod
    def normalize_prefix(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        return v.lower() or None


class TagsAdd(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tags: list[str] = Field(..., min_length=1)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: list[str]) -> list[str]:
        out = []
        for t in v:
            if not isinstance(t, str):
                raise TypeError("tags must be strings")
            tnorm = t.strip().lower()
            if tnorm:
                out.append(tnorm)
        seen = set()
        deduplicated = []
        for x in out:
            if x not in seen:
                seen.add(x)
                deduplicated.append(x)
        return deduplicated


class TagsRemove(TagsAdd):
    pass

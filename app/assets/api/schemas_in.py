import json
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    conint,
    field_validator,
    model_validator,
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


class UpdateAssetBody(BaseModel):
    name: str | None = None
    user_metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _at_least_one(self):
        if self.name is None and self.user_metadata is None:
            raise ValueError("Provide at least one of: name, user_metadata.")
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
            seen = set()
            dedup = []
            for t in out:
                if t not in seen:
                    seen.add(t)
                    dedup.append(t)
            return dedup
        if isinstance(v, str):
            return [t.strip().lower() for t in v.split(",") if t.strip()]
        return []


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


class UploadAssetSpec(BaseModel):
    """Upload Asset operation.
    - tags: ordered; first is root ('models'|'input'|'output');
            if root == 'models', second must be a valid category from folder_paths.folder_names_and_paths
    - name: display name
    - user_metadata: arbitrary JSON object (optional)
    - hash: optional canonical 'blake3:<hex>' provided by the client for validation / fast-path

    Files created via this endpoint are stored on disk using the **content hash** as the filename stem
    and the original extension is preserved when available.
    """
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    tags: list[str] = Field(..., min_length=1)
    name: str | None = Field(default=None, max_length=512, description="Display Name")
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    hash: str | None = Field(default=None)

    @field_validator("hash", mode="before")
    @classmethod
    def _parse_hash(cls, v):
        if v is None:
            return None
        s = str(v).strip().lower()
        if not s:
            return None
        if ":" not in s:
            raise ValueError("hash must be 'blake3:<hex>'")
        algo, digest = s.split(":", 1)
        if algo != "blake3":
            raise ValueError("only canonical 'blake3:<hex>' is accepted here")
        if not digest or any(c for c in digest if c not in "0123456789abcdef"):
            raise ValueError("hash digest must be lowercase hex")
        return f"{algo}:{digest}"

    @field_validator("tags", mode="before")
    @classmethod
    def _parse_tags(cls, v):
        """
        Accepts a list of strings (possibly multiple form fields),
        where each string can be:
          - JSON array (e.g., '["models","loras","foo"]')
          - comma-separated ('models, loras, foo')
          - single token ('models')
        Returns a normalized, deduplicated, ordered list.
        """
        items: list[str] = []
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]

        if isinstance(v, list):
            for item in v:
                if item is None:
                    continue
                s = str(item).strip()
                if not s:
                    continue
                if s.startswith("["):
                    try:
                        arr = json.loads(s)
                        if isinstance(arr, list):
                            items.extend(str(x) for x in arr)
                            continue
                    except Exception:
                        pass  # fallback to CSV parse below
                items.extend([p for p in s.split(",") if p.strip()])
        else:
            return []

        # normalize + dedupe
        norm = []
        seen = set()
        for t in items:
            tnorm = str(t).strip().lower()
            if tnorm and tnorm not in seen:
                seen.add(tnorm)
                norm.append(tnorm)
        return norm

    @field_validator("user_metadata", mode="before")
    @classmethod
    def _parse_metadata_json(cls, v):
        if v is None or isinstance(v, dict):
            return v or {}
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return {}
            try:
                parsed = json.loads(s)
            except Exception as e:
                raise ValueError(f"user_metadata must be JSON: {e}") from e
            if not isinstance(parsed, dict):
                raise ValueError("user_metadata must be a JSON object")
            return parsed
        return {}

    @model_validator(mode="after")
    def _validate_order(self):
        if not self.tags:
            raise ValueError("tags must be provided and non-empty")
        root = self.tags[0]
        if root not in {"models", "input", "output"}:
            raise ValueError("first tag must be one of: models, input, output")
        if root == "models":
            if len(self.tags) < 2:
                raise ValueError("models uploads require a category tag as the second tag")
        return self

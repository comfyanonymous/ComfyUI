from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
import uuid

from sqlalchemy import (
    Integer,
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
    JSON,
    String,
    Text,
    CheckConstraint,
    Numeric,
    Boolean,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, foreign

from .timeutil import utcnow


class Base(DeclarativeBase):
    pass


def to_dict(obj: Any, include_none: bool = False) -> dict[str, Any]:
    fields = obj.__table__.columns.keys()
    out: dict[str, Any] = {}
    for field in fields:
        val = getattr(obj, field)
        if val is None and not include_none:
            continue
        if isinstance(val, datetime):
            out[field] = val.isoformat()
        else:
            out[field] = val
    return out


class Asset(Base):
    __tablename__ = "assets"

    hash: Mapped[str] = mapped_column(String(256), primary_key=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )

    infos: Mapped[list["AssetInfo"]] = relationship(
        "AssetInfo",
        back_populates="asset",
        primaryjoin=lambda: Asset.hash == foreign(AssetInfo.asset_hash),
        foreign_keys=lambda: [AssetInfo.asset_hash],
        cascade="all,delete-orphan",
        passive_deletes=True,
    )

    preview_of: Mapped[list["AssetInfo"]] = relationship(
        "AssetInfo",
        back_populates="preview_asset",
        primaryjoin=lambda: Asset.hash == foreign(AssetInfo.preview_hash),
        foreign_keys=lambda: [AssetInfo.preview_hash],
        viewonly=True,
    )

    cache_states: Mapped[list["AssetCacheState"]] = relationship(
        back_populates="asset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    locations: Mapped[list["AssetLocation"]] = relationship(
        back_populates="asset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_assets_mime_type", "mime_type"),
    )

    def to_dict(self, include_none: bool = False) -> dict[str, Any]:
        return to_dict(self, include_none=include_none)

    def __repr__(self) -> str:
        return f"<Asset hash={self.hash[:12]}>"


class AssetCacheState(Base):
    __tablename__ = "asset_cache_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_hash: Mapped[str] = mapped_column(String(256), ForeignKey("assets.hash", ondelete="CASCADE"), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    mtime_ns: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    asset: Mapped["Asset"] = relationship(back_populates="cache_states")

    __table_args__ = (
        Index("ix_asset_cache_state_file_path", "file_path"),
        Index("ix_asset_cache_state_asset_hash", "asset_hash"),
        CheckConstraint("(mtime_ns IS NULL) OR (mtime_ns >= 0)", name="ck_acs_mtime_nonneg"),
        UniqueConstraint("file_path", name="uq_asset_cache_state_file_path"),
    )

    def to_dict(self, include_none: bool = False) -> dict[str, Any]:
        return to_dict(self, include_none=include_none)

    def __repr__(self) -> str:
        return f"<AssetCacheState id={self.id} hash={self.asset_hash[:12]} path={self.file_path!r}>"


class AssetLocation(Base):
    __tablename__ = "asset_locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_hash: Mapped[str] = mapped_column(String(256), ForeignKey("assets.hash", ondelete="CASCADE"), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)  # "gcs"
    locator: Mapped[str] = mapped_column(Text, nullable=False)         # "gs://bucket/object"
    expected_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    etag: Mapped[str | None] = mapped_column(String(256), nullable=True)
    last_modified: Mapped[str | None] = mapped_column(String(128), nullable=True)

    asset: Mapped["Asset"] = relationship(back_populates="locations")

    __table_args__ = (
        UniqueConstraint("asset_hash", "provider", "locator", name="uq_asset_locations_triplet"),
        Index("ix_asset_locations_hash", "asset_hash"),
        Index("ix_asset_locations_provider", "provider"),
    )


class AssetInfo(Base):
    __tablename__ = "assets_info"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    asset_hash: Mapped[str] = mapped_column(
        String(256), ForeignKey("assets.hash", ondelete="RESTRICT"), nullable=False
    )
    preview_hash: Mapped[str | None] = mapped_column(String(256), ForeignKey("assets.hash", ondelete="SET NULL"))
    user_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    last_access_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )

    # Relationships
    asset: Mapped[Asset] = relationship(
        "Asset",
        back_populates="infos",
        foreign_keys=[asset_hash],
    )
    preview_asset: Mapped[Asset | None] = relationship(
        "Asset",
        back_populates="preview_of",
        foreign_keys=[preview_hash],
    )

    metadata_entries: Mapped[list["AssetInfoMeta"]] = relationship(
        back_populates="asset_info",
        cascade="all,delete-orphan",
        passive_deletes=True,
    )

    tag_links: Mapped[list["AssetInfoTag"]] = relationship(
        back_populates="asset_info",
        cascade="all,delete-orphan",
        passive_deletes=True,
        overlaps="tags,asset_infos",
    )

    tags: Mapped[list["Tag"]] = relationship(
        secondary="asset_info_tags",
        back_populates="asset_infos",
        lazy="joined",
        viewonly=True,
        overlaps="tag_links,asset_info_links,asset_infos,tag",
    )

    __table_args__ = (
        UniqueConstraint("asset_hash", "owner_id", "name", name="uq_assets_info_hash_owner_name"),
        Index("ix_assets_info_owner_name", "owner_id", "name"),
        Index("ix_assets_info_owner_id", "owner_id"),
        Index("ix_assets_info_asset_hash", "asset_hash"),
        Index("ix_assets_info_name", "name"),
        Index("ix_assets_info_created_at", "created_at"),
        Index("ix_assets_info_last_access_time", "last_access_time"),
    )

    def to_dict(self, include_none: bool = False) -> dict[str, Any]:
        data = to_dict(self, include_none=include_none)
        data["tags"] = [t.name for t in self.tags]
        return data

    def __repr__(self) -> str:
        return f"<AssetInfo id={self.id} name={self.name!r} hash={self.asset_hash[:12]}>"


class AssetInfoMeta(Base):
    __tablename__ = "asset_info_meta"

    asset_info_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assets_info.id", ondelete="CASCADE"), primary_key=True
    )
    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True, default=0)

    val_str: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    val_num: Mapped[Optional[float]] = mapped_column(Numeric(38, 10), nullable=True)
    val_bool: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    val_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    asset_info: Mapped["AssetInfo"] = relationship(back_populates="metadata_entries")

    __table_args__ = (
        Index("ix_asset_info_meta_key", "key"),
        Index("ix_asset_info_meta_key_val_str", "key", "val_str"),
        Index("ix_asset_info_meta_key_val_num", "key", "val_num"),
        Index("ix_asset_info_meta_key_val_bool", "key", "val_bool"),
    )


class AssetInfoTag(Base):
    __tablename__ = "asset_info_tags"

    asset_info_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assets_info.id", ondelete="CASCADE"), primary_key=True
    )
    tag_name: Mapped[str] = mapped_column(
        String(512), ForeignKey("tags.name", ondelete="RESTRICT"), primary_key=True
    )
    origin: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )

    asset_info: Mapped["AssetInfo"] = relationship(back_populates="tag_links")
    tag: Mapped["Tag"] = relationship(back_populates="asset_info_links")

    __table_args__ = (
        Index("ix_asset_info_tags_tag_name", "tag_name"),
        Index("ix_asset_info_tags_asset_info_id", "asset_info_id"),
    )


class Tag(Base):
    __tablename__ = "tags"

    name: Mapped[str] = mapped_column(String(512), primary_key=True)
    tag_type: Mapped[str] = mapped_column(String(32), nullable=False, default="user")

    asset_info_links: Mapped[list["AssetInfoTag"]] = relationship(
        back_populates="tag",
        overlaps="asset_infos,tags",
    )
    asset_infos: Mapped[list["AssetInfo"]] = relationship(
        secondary="asset_info_tags",
        back_populates="tags",
        viewonly=True,
        overlaps="asset_info_links,tag_links,tags,asset_info",
    )

    __table_args__ = (
        Index("ix_tags_tag_type", "tag_type"),
    )

    def __repr__(self) -> str:
        return f"<Tag {self.name}>"

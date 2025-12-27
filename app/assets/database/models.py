from __future__ import annotations

import uuid
from datetime import datetime

from typing import Any
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, foreign, mapped_column, relationship

from app.assets.helpers import utcnow
from app.database.models import to_dict, Base


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    hash: Mapped[str | None] = mapped_column(String(256), nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )

    infos: Mapped[list[AssetInfo]] = relationship(
        "AssetInfo",
        back_populates="asset",
        primaryjoin=lambda: Asset.id == foreign(AssetInfo.asset_id),
        foreign_keys=lambda: [AssetInfo.asset_id],
        cascade="all,delete-orphan",
        passive_deletes=True,
    )

    preview_of: Mapped[list[AssetInfo]] = relationship(
        "AssetInfo",
        back_populates="preview_asset",
        primaryjoin=lambda: Asset.id == foreign(AssetInfo.preview_id),
        foreign_keys=lambda: [AssetInfo.preview_id],
        viewonly=True,
    )

    cache_states: Mapped[list[AssetCacheState]] = relationship(
        back_populates="asset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("uq_assets_hash", "hash", unique=True),
        Index("ix_assets_mime_type", "mime_type"),
        CheckConstraint("size_bytes >= 0", name="ck_assets_size_nonneg"),
    )

    def to_dict(self, include_none: bool = False) -> dict[str, Any]:
        return to_dict(self, include_none=include_none)

    def __repr__(self) -> str:
        return f"<Asset id={self.id} hash={(self.hash or '')[:12]}>"


class AssetCacheState(Base):
    __tablename__ = "asset_cache_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(String(36), ForeignKey("assets.id", ondelete="CASCADE"), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    mtime_ns: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    needs_verify: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    asset: Mapped[Asset] = relationship(back_populates="cache_states")

    __table_args__ = (
        Index("ix_asset_cache_state_file_path", "file_path"),
        Index("ix_asset_cache_state_asset_id", "asset_id"),
        CheckConstraint("(mtime_ns IS NULL) OR (mtime_ns >= 0)", name="ck_acs_mtime_nonneg"),
        UniqueConstraint("file_path", name="uq_asset_cache_state_file_path"),
    )

    def to_dict(self, include_none: bool = False) -> dict[str, Any]:
        return to_dict(self, include_none=include_none)

    def __repr__(self) -> str:
        return f"<AssetCacheState id={self.id} asset_id={self.asset_id} path={self.file_path!r}>"


class AssetInfo(Base):
    __tablename__ = "assets_info"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    asset_id: Mapped[str] = mapped_column(String(36), ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False)
    preview_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("assets.id", ondelete="SET NULL"))
    user_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=utcnow)
    last_access_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=utcnow)

    asset: Mapped[Asset] = relationship(
        "Asset",
        back_populates="infos",
        foreign_keys=[asset_id],
        lazy="selectin",
    )
    preview_asset: Mapped[Asset | None] = relationship(
        "Asset",
        back_populates="preview_of",
        foreign_keys=[preview_id],
    )

    metadata_entries: Mapped[list[AssetInfoMeta]] = relationship(
        back_populates="asset_info",
        cascade="all,delete-orphan",
        passive_deletes=True,
    )

    tag_links: Mapped[list[AssetInfoTag]] = relationship(
        back_populates="asset_info",
        cascade="all,delete-orphan",
        passive_deletes=True,
        overlaps="tags,asset_infos",
    )

    tags: Mapped[list[Tag]] = relationship(
        secondary="asset_info_tags",
        back_populates="asset_infos",
        lazy="selectin",
        viewonly=True,
        overlaps="tag_links,asset_info_links,asset_infos,tag",
    )

    __table_args__ = (
        UniqueConstraint("asset_id", "owner_id", "name", name="uq_assets_info_asset_owner_name"),
        Index("ix_assets_info_owner_name", "owner_id", "name"),
        Index("ix_assets_info_owner_id", "owner_id"),
        Index("ix_assets_info_asset_id", "asset_id"),
        Index("ix_assets_info_name", "name"),
        Index("ix_assets_info_created_at", "created_at"),
        Index("ix_assets_info_last_access_time", "last_access_time"),
    )

    def to_dict(self, include_none: bool = False) -> dict[str, Any]:
        data = to_dict(self, include_none=include_none)
        data["tags"] = [t.name for t in self.tags]
        return data

    def __repr__(self) -> str:
        return f"<AssetInfo id={self.id} name={self.name!r} asset_id={self.asset_id}>"


class AssetInfoMeta(Base):
    __tablename__ = "asset_info_meta"

    asset_info_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assets_info.id", ondelete="CASCADE"), primary_key=True
    )
    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True, default=0)

    val_str: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    val_num: Mapped[float | None] = mapped_column(Numeric(38, 10), nullable=True)
    val_bool: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    val_json: Mapped[Any | None] = mapped_column(JSON(none_as_null=True), nullable=True)

    asset_info: Mapped[AssetInfo] = relationship(back_populates="metadata_entries")

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

    asset_info: Mapped[AssetInfo] = relationship(back_populates="tag_links")
    tag: Mapped[Tag] = relationship(back_populates="asset_info_links")

    __table_args__ = (
        Index("ix_asset_info_tags_tag_name", "tag_name"),
        Index("ix_asset_info_tags_asset_info_id", "asset_info_id"),
    )


class Tag(Base):
    __tablename__ = "tags"

    name: Mapped[str] = mapped_column(String(512), primary_key=True)
    tag_type: Mapped[str] = mapped_column(String(32), nullable=False, default="user")

    asset_info_links: Mapped[list[AssetInfoTag]] = relationship(
        back_populates="tag",
        overlaps="asset_infos,tags",
    )
    asset_infos: Mapped[list[AssetInfo]] = relationship(
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

"""SQLAlchemy models for the download manager.

Two tables:

- ``downloads``         one row per requested file (job + queue state).
- ``download_segments`` per-segment byte progress, for segmented resume.

On completion a finished file is registered into the assets catalog;
``downloads`` is kept only as job history.
"""

from __future__ import annotations

import time
import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.models import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> int:
    return int(time.time())


class Download(Base):
    __tablename__ = "downloads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # Original requested URL and the final URL after validated redirects.
    url: Mapped[str] = mapped_column(Text, nullable=False)
    final_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Canonical "<directory>/<filename>" identifier (resolved via folder_paths).
    model_id: Mapped[str] = mapped_column(String(1024), nullable=False)
    # Final on-disk location and the .part write target.
    dest_path: Mapped[str] = mapped_column(Text, nullable=False)
    temp_path: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(String(16), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    total_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    bytes_done: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    etag: Mapped[str | None] = mapped_column(String(512), nullable=True)
    last_modified: Mapped[str | None] = mapped_column(String(128), nullable=True)
    accept_ranges: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Optional hub-provided checksum to verify against (NOT the dedup key).
    expected_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    allow_any_extension: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    # How many retryable failures we have seen (for backoff capping).
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False, default=_now)
    updated_at: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=_now, onupdate=_now
    )

    segments: Mapped[list[DownloadSegment]] = relationship(
        "DownloadSegment",
        back_populates="download",
        cascade="all,delete-orphan",
        passive_deletes=True,
        order_by="DownloadSegment.idx",
    )

    __table_args__ = (
        Index("ix_downloads_status", "status"),
        Index("ix_downloads_priority", "priority"),
        Index("ix_downloads_model_id", "model_id"),
        CheckConstraint("bytes_done >= 0", name="ck_downloads_bytes_done_nonneg"),
        CheckConstraint(
            "total_bytes IS NULL OR total_bytes >= 0",
            name="ck_downloads_total_bytes_nonneg",
        ),
    )

    def __repr__(self) -> str:
        return f"<Download id={self.id} model_id={self.model_id!r} status={self.status}>"


class DownloadSegment(Base):
    __tablename__ = "download_segments"

    download_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("downloads.id", ondelete="CASCADE"),
        primary_key=True,
    )
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    start_offset: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_offset: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bytes_done: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    download: Mapped[Download] = relationship("Download", back_populates="segments")

    __table_args__ = (
        CheckConstraint("bytes_done >= 0", name="ck_segments_bytes_done_nonneg"),
        CheckConstraint("end_offset >= start_offset", name="ck_segments_range"),
    )

    def __repr__(self) -> str:
        return (
            f"<DownloadSegment {self.download_id}#{self.idx} "
            f"{self.start_offset}-{self.end_offset} done={self.bytes_done}>"
        )

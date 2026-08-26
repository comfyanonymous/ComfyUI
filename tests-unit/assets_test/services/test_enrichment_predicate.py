"""Enrichment-candidate predicate: system_metadata None must be SQL NULL.

The OFF-mode (metadata) enrich predicate is ``Asset.system_metadata.is_(None)``
(a SQL ``IS NULL``). SQLAlchemy's default ``JSON`` type serialises Python
``None`` to the JSON text ``'null'``, which is NOT SQL NULL, so the predicate
would match zero rows and enrichment would be a silent no-op. These tests lock
the contract: a metadata-less record is stored as SQL NULL and is therefore
returned as an enrich candidate, while a record that already carries
system_metadata is not.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.assets.database.queries import create_content, create_record
from app.assets.scanner import get_unenriched_assets_for_roots


@contextmanager
def _reuse_session(session: Session) -> Iterator[Session]:
    """Hand the seeded session to scanner.create_session without closing it."""
    yield session


def _raw_system_metadata(session: Session, record_id: str) -> object:
    """Read the stored column value with no ORM/JSON type processing."""
    return session.execute(
        sa.text("SELECT system_metadata FROM assets WHERE id = :id"),
        {"id": record_id},
    ).scalar()


def test_create_record_stores_none_system_metadata_as_sql_null(
    session: Session,
) -> None:
    # Given a content row
    content = create_content(session, "/models/no-meta.safetensors", hash=None)

    # When a record is created without system_metadata (defaults to None)
    record = create_record(session, content.id, "no-meta.safetensors")
    session.commit()

    # Then the column holds SQL NULL, not the JSON text 'null'
    assert _raw_system_metadata(session, record.id) is None


def test_off_mode_returns_metadata_less_seeded_asset(
    session: Session, temp_dir: Path
) -> None:
    # Given a freshly-seeded metadata-less asset under a scan prefix
    path = temp_dir / "fresh.safetensors"
    content = create_content(session, str(path), hash=None)
    record = create_record(session, content.id, path.name)
    session.commit()
    record_id = record.id

    # When enrichment candidates are queried in OFF (metadata) mode
    with (
        patch("app.assets.scanner.create_session", lambda: _reuse_session(session)),
        patch(
            "app.assets.scanner.get_scan_prefixes_for_root",
            return_value=[str(temp_dir)],
        ),
    ):
        rows = get_unenriched_assets_for_roots(("models",), compute_hashes=False)

    # Then the seeded asset is an enrich candidate
    assert record_id in {row.record_id for row in rows}


def test_off_mode_excludes_asset_with_system_metadata(
    session: Session, temp_dir: Path
) -> None:
    # Given an asset that already carries system_metadata
    path = temp_dir / "enriched.safetensors"
    content = create_content(session, str(path), hash=None)
    record = create_record(
        session,
        content.id,
        path.name,
        system_metadata={"architecture": "flux"},
    )
    session.commit()
    record_id = record.id

    # When enrichment candidates are queried in OFF (metadata) mode
    with (
        patch("app.assets.scanner.create_session", lambda: _reuse_session(session)),
        patch(
            "app.assets.scanner.get_scan_prefixes_for_root",
            return_value=[str(temp_dir)],
        ),
    ):
        rows = get_unenriched_assets_for_roots(("models",), compute_hashes=False)

    # Then it is not returned as an enrich candidate
    assert record_id not in {row.record_id for row in rows}

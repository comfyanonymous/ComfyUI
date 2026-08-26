"""Unified split policy: identity by (path, size), NULL-metadata replacements.

Three properties are locked here:

1. Enrich candidacy (hash mode). The hash-mode enrich predicate must return a
   split-created record — hash set, ``system_metadata`` NULL — so its metadata
   is filled in a later pass. A fully enriched record (hash + metadata) must
   not be returned.

2. Split gate. ``detect_content_change`` splits only when *both* ``mtime_ns``
   and ``size_bytes`` change. A bare mtime bump at the same size (rsync, cloud
   sync, backup restore) is the same file and must not split — user tags and
   metadata survive on the live record. ``mtime`` unchanged stays the existing
   Ruling #10 early return. Accepting that bump also refreshes the stored stat
   snapshot: ``lookup._stat_consistent`` demands exact ``mtime_ns`` equality, so
   a stale stored mtime would keep the row unservable by hash forever and make
   every later scan re-detect the same change.

3. Replacement shape. Both split sites (``scanner_changes.split_content`` and
   ``hash_mode_state.drain_transition_queue``) create the replacement record
   with real SQL NULL ``system_metadata`` — byte-derived metadata from the old
   bytes is never carried onto the new bytes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.orm import Session

import pytest

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries import create_content, create_record
from app.assets.database.queries.records import fetch_record_tags
from app.assets.helpers import to_stored_hash
from app.assets.scanner import get_unenriched_assets_for_roots
from app.assets.scanner_changes import (
    clear_pending_verifications,
    detect_content_change,
    drain_pending_verifications,
)
from app.assets.services.hash_mode_state import (
    clear_transition_queue,
    drain_transition_queue,
    enqueue_transition_work,
)
from app.assets.services.lookup import lookup_for_view
from app.assets.services.snapshot_hash import snapshot_hash


@dataclass(frozen=True, slots=True)
class _FakeStat:
    """Minimal stand-in for ``os.stat_result`` (only size + mtime are read)."""

    st_size: int
    st_mtime_ns: int


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


def _candidates_under(session: Session, temp_dir: Path, *, compute_hashes: bool) -> set[str]:
    with (
        patch("app.assets.scanner.create_session", lambda: _reuse_session(session)),
        patch(
            "app.assets.scanner.get_scan_prefixes_for_root",
            return_value=[str(temp_dir)],
        ),
    ):
        rows = get_unenriched_assets_for_roots(("models",), compute_hashes=compute_hashes)
    return {row.record_id for row in rows}


@pytest.fixture(autouse=True)
def _transition_queue_isolation() -> Iterator[None]:
    clear_transition_queue()
    yield
    clear_transition_queue()


@pytest.fixture(autouse=True)
def _input_base_is_temp_dir(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make temp_dir a recognized input base so path-derived tags resolve.

    ``split_content`` and ``drain_transition_queue`` derive replacement tags from
    the path; a bare ``/tmp`` path is under no known root and would raise.
    """
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))


@pytest.fixture(autouse=True)
def _pending_verification_isolation() -> Iterator[None]:
    """``scanner_changes`` keeps its verification queue in module globals."""
    clear_pending_verifications()
    yield
    clear_pending_verifications()


def _seed_hashed_content(session: Session, path: Path, data: bytes) -> AssetContent:
    """Write real bytes and store a row whose stat snapshot matches the file."""
    path.write_bytes(data)
    snapshot = snapshot_hash(str(path))
    assert snapshot is not None
    digest, verified_stat = snapshot
    return create_content(
        session,
        str(path),
        to_stored_hash(digest),
        verified_stat.st_size,
        verified_stat.st_mtime_ns,
    )


def _bump_mtime(path: Path) -> os.stat_result:
    """Touch the file's mtime without changing a single byte."""
    before = path.stat()
    bumped = before.st_mtime_ns + 5_000_000_000
    os.utime(path, ns=(bumped, bumped))
    after = path.stat()
    assert after.st_mtime_ns != before.st_mtime_ns
    assert after.st_size == before.st_size
    return after


# --- (i) + (ii): hash-mode enrich candidacy ---------------------------------


def test_split_record_is_enrich_candidate_in_hash_mode(
    session: Session, temp_dir: Path
) -> None:
    # Given a split-created record: hash set, system_metadata NULL
    path = temp_dir / "split.safetensors"
    content = create_content(session, str(path), hash="blake3:deadbeef")
    record = create_record(session, content.id, path.name)
    session.commit()
    record_id = record.id

    # When enrichment candidates are queried in hash mode
    candidates = _candidates_under(session, temp_dir, compute_hashes=True)

    # Then the split record is a candidate: it still needs metadata
    assert record_id in candidates


def test_enriched_record_is_not_enrich_candidate_in_hash_mode(
    session: Session, temp_dir: Path
) -> None:
    # Given a fully enriched record: hash set AND system_metadata set
    path = temp_dir / "enriched.safetensors"
    content = create_content(session, str(path), hash="blake3:deadbeef")
    record = create_record(
        session, content.id, path.name, system_metadata={"architecture": "flux"}
    )
    session.commit()
    record_id = record.id

    # When enrichment candidates are queried in hash mode
    candidates = _candidates_under(session, temp_dir, compute_hashes=True)

    # Then a fully enriched record is not returned
    assert record_id not in candidates


# --- (iii) + (iv) + (v): the split gate -------------------------------------


def test_same_size_mtime_bump_does_not_split(session: Session, temp_dir: Path) -> None:
    # Given a live record carrying user tags and metadata
    path = temp_dir / "touched.safetensors"
    content = create_content(session, str(path), hash=None, size_bytes=100, mtime_ns=1000)
    record = create_record(
        session, content.id, path.name, tags=["keepme"], system_metadata={"k": "v"}
    )
    session.commit()
    content_id, record_id = content.id, record.id

    # When a same-size mtime bump is observed (OFF mode)
    detect_content_change(
        session, content, _FakeStat(st_size=100, st_mtime_ns=2000), hashing_is_enabled=False
    )
    session.commit()
    session.expire_all()

    # Then nothing splits: one live row, tags and metadata survive
    live = session.get(AssetContent, content_id)
    assert live is not None and live.is_missing is False
    rows_at_path = list(
        session.scalars(select(AssetContent).where(AssetContent.path == str(path)))
    )
    assert len(rows_at_path) == 1
    surviving = session.get(Asset, record_id)
    assert surviving.system_metadata == {"k": "v"}
    tags = fetch_record_tags(session, record_id)
    assert "keepme" in tags
    assert "missing" not in tags

    # ... and accepting the bump adopts the observed stat as the row's snapshot
    assert live.mtime_ns == 2000
    assert live.size_bytes == 100


def test_accepted_mtime_bump_keeps_content_servable_by_hash(
    session: Session, temp_dir: Path
) -> None:
    # Given a live hashed record whose stored stat matches the file on disk
    path = temp_dir / "synced.safetensors"
    content = _seed_hashed_content(session, path, b"rsynced bytes")
    create_record(session, content.id, path.name, tags=["keepme"])
    session.commit()
    content_id, stored_hash = content.id, content.hash
    assert lookup_for_view(session, stored_hash) is not None

    # When a same-size mtime bump on the real file is accepted (OFF mode)
    observed = _bump_mtime(path)
    detect_content_change(session, content, observed, hashing_is_enabled=False)
    session.commit()
    session.expire_all()

    # Then the row is still resolvable by hash: /view, from-hash and upload dedup
    # all gate on lookup._stat_consistent, which demands exact mtime equality
    served = lookup_for_view(session, stored_hash)
    assert served is not None and served.id == content_id

    # ... because the stored snapshot tracks the file, hash and identity untouched
    live = session.get(AssetContent, content_id)
    assert live.mtime_ns == observed.st_mtime_ns
    assert live.size_bytes == observed.st_size
    assert live.hash == stored_hash
    assert "keepme" in fetch_record_tags(
        session, session.scalar(select(Asset).where(Asset.content_id == content_id)).id
    )


def test_accepted_mtime_bump_is_not_re_detected_by_the_next_scan(
    session: Session, temp_dir: Path
) -> None:
    # Given a live hashed record whose same-size mtime bump was already accepted
    path = temp_dir / "resynced.safetensors"
    content = _seed_hashed_content(session, path, b"cloud synced bytes")
    create_record(session, content.id, path.name, tags=["keepme"])
    session.commit()
    content_id = content.id
    detect_content_change(session, content, _bump_mtime(path), hashing_is_enabled=False)
    session.commit()

    # When the next scan pass re-observes the very same, untouched file
    detect_content_change(session, content, path.stat(), hashing_is_enabled=True)

    # Then it does not re-enter the change branch: no verification work is queued
    # (a stale stored mtime would re-queue a rehash of this file on every pass)
    assert drain_pending_verifications(session) == 0

    # ... and an OFF-mode pass is equally inert: still one live row, no split
    detect_content_change(session, content, path.stat(), hashing_is_enabled=False)
    session.commit()
    session.expire_all()
    rows_at_path = list(
        session.scalars(select(AssetContent).where(AssetContent.path == str(path)))
    )
    assert len(rows_at_path) == 1
    assert rows_at_path[0].id == content_id
    assert rows_at_path[0].is_missing is False


def test_mtime_and_size_change_splits_with_null_metadata(
    session: Session, temp_dir: Path
) -> None:
    # Given a live record carrying user tags and metadata
    path = temp_dir / "grown.safetensors"
    content = create_content(session, str(path), hash=None, size_bytes=100, mtime_ns=1000)
    create_record(
        session, content.id, path.name, tags=["oldtag"], system_metadata={"k": "v"}
    )
    session.commit()
    old_content_id = content.id

    # When both mtime AND size change (genuine content change, OFF mode)
    detect_content_change(
        session, content, _FakeStat(st_size=200, st_mtime_ns=2000), hashing_is_enabled=False
    )
    session.commit()
    session.expire_all()

    # Then the old content is retired and a fresh replacement takes its place
    assert session.get(AssetContent, old_content_id).is_missing is True
    live = session.scalar(
        select(AssetContent).where(
            AssetContent.path == str(path), AssetContent.is_missing.is_(False)
        )
    )
    assert live is not None and live.id != old_content_id
    assert live.size_bytes == 200 and live.mtime_ns == 2000
    new_record = session.scalar(select(Asset).where(Asset.content_id == live.id))
    assert new_record is not None

    # ... the replacement carries real NULL metadata (never the old bytes' meta)
    assert new_record.system_metadata is None
    assert _raw_system_metadata(session, new_record.id) is None
    # ... a genuine change drops the old user tags
    assert "oldtag" not in fetch_record_tags(session, new_record.id)
    # ... and the fresh replacement is immediately enrichable
    assert new_record.id in _candidates_under(session, temp_dir, compute_hashes=True)


def test_mtime_unchanged_size_changed_does_not_split(
    session: Session, temp_dir: Path
) -> None:
    # Given a live record
    path = temp_dir / "weird.safetensors"
    content = create_content(session, str(path), hash=None, size_bytes=100, mtime_ns=1000)
    create_record(session, content.id, path.name, tags=["keepme"])
    session.commit()
    content_id = content.id

    # When size drifts but mtime is unchanged (Ruling #10 undefined territory)
    detect_content_change(
        session, content, _FakeStat(st_size=999, st_mtime_ns=1000), hashing_is_enabled=False
    )
    session.commit()
    session.expire_all()

    # Then the existing early return holds: no split
    assert session.get(AssetContent, content_id).is_missing is False
    rows_at_path = list(
        session.scalars(select(AssetContent).where(AssetContent.path == str(path)))
    )
    assert len(rows_at_path) == 1


# --- (vi): hash_mode_state split site produces the same replacement shape ----


def test_transition_drain_split_replacement_has_null_metadata(
    session: Session, temp_dir: Path
) -> None:
    # Given a live hashed record carrying user tags and metadata
    path = temp_dir / "changed.bin"
    path.write_bytes(b"old bytes")
    old_snapshot = snapshot_hash(str(path))
    assert old_snapshot is not None
    old_digest, _ = old_snapshot
    stat = path.stat()
    old_content = create_content(
        session, str(path), to_stored_hash(old_digest), stat.st_size, stat.st_mtime_ns
    )
    old_content_id = old_content.id
    create_record(
        session, old_content_id, "changed.bin", tags=["oldtag"], system_metadata={"k": "v"}
    )
    # ... whose bytes then change to a different hash
    path.write_bytes(b"different new bytes")

    # When the OFF->ON transition drains and splits the changed row
    enqueue_transition_work(session, "off_to_on")
    drain_transition_queue(session)
    session.commit()
    session.expire_all()

    # Then the replacement has the same shape as the scanner split: NULL metadata
    assert session.get(AssetContent, old_content_id).is_missing is True
    live = session.scalar(
        select(AssetContent).where(
            AssetContent.path == str(path), AssetContent.is_missing.is_(False)
        )
    )
    assert live is not None and live.id != old_content_id
    new_record = session.scalar(select(Asset).where(Asset.content_id == live.id))
    assert new_record is not None
    assert new_record.system_metadata is None
    assert _raw_system_metadata(session, new_record.id) is None
    assert "oldtag" not in fetch_record_tags(session, new_record.id)

from pathlib import Path

import pytest
from blake3 import blake3
from sqlalchemy import select

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, create_record
from app.assets.helpers import to_stored_hash
from app.assets.services import hash_mode_state
from app.assets.services.hash_mode_state import (
    clear_transition_queue,
    drain_transition_queue,
    enqueue_transition_work,
    read_stored_mode,
    record_transition_intent,
    write_stored_mode,
)
from app.assets.services.path_utils import get_name_and_tags_from_asset_path
from app.assets.services.snapshot_hash import snapshot_hash


@pytest.fixture(autouse=True)
def transition_queue():
    clear_transition_queue()
    yield
    clear_transition_queue()


def _stored_hash(path: Path) -> str:
    snapshot = snapshot_hash(str(path))
    assert snapshot is not None
    digest, _ = snapshot
    return to_stored_hash(digest)


def test_off_to_on_transition_hashes_null_rows_and_persists_mode(session, temp_dir, monkeypatch):
    paths = [temp_dir / "first.bin", temp_dir / "second.bin"]
    for index, path in enumerate(paths):
        path.write_bytes(f"bytes-{index}".encode())
        stat = path.stat()
        create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)

    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    drain_transition_queue(session)
    session.commit()

    contents = list(session.scalars(select(AssetContent)))
    assert {content.hash for content in contents} == {_stored_hash(path) for path in paths}
    assert read_stored_mode(session) == "on"


def test_transition_drain_splits_changed_content(session, temp_dir, monkeypatch):
    path = temp_dir / "changed.bin"
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    path.write_bytes(b"old bytes")
    old_snapshot = snapshot_hash(str(path))
    assert old_snapshot is not None
    old_digest, _ = old_snapshot
    stat = path.stat()
    old_content = create_content(
        session, str(path), to_stored_hash(old_digest), stat.st_size, stat.st_mtime_ns
    )
    old_content_id = old_content.id
    create_record(session, old_content_id, "changed.bin")
    path.write_bytes(b"new bytes")

    enqueue_transition_work(session, "off_to_on")
    drain_transition_queue(session)
    session.commit()

    contents = list(session.scalars(select(AssetContent)))
    live_content = next(content for content in contents if not content.is_missing)
    records = list(session.scalars(select(Asset)))
    assert session.get(AssetContent, old_content_id).is_missing is True
    assert live_content.hash == _stored_hash(path)
    assert len(records) == 2
    assert any(record.content_id == live_content.id for record in records)


def test_transition_drain_requeues_permission_errors_and_processes_other_paths(
    session, temp_dir, monkeypatch
):
    # Given
    protected_path = temp_dir / "protected.bin"
    healthy_path = temp_dir / "healthy.bin"
    protected_path.write_bytes(b"protected")
    healthy_payload = b"healthy"
    healthy_path.write_bytes(healthy_payload)
    for path in (protected_path, healthy_path):
        stat = path.stat()
        create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)

    def hash_or_raise(candidate_path: str):
        if candidate_path == str(protected_path):
            raise PermissionError("denied")
        return snapshot_hash(candidate_path)

    monkeypatch.setattr(hash_mode_state, "snapshot_hash", hash_or_raise)
    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)

    # When
    drain_transition_queue(session)

    # Then
    healthy_content = session.scalar(
        select(AssetContent).where(AssetContent.path == str(healthy_path))
    )
    assert healthy_content is not None
    assert healthy_content.hash == to_stored_hash(blake3(healthy_payload).hexdigest())
    assert hash_mode_state.pending_transition_count() == 1
    assert read_stored_mode(session) == "off"


def test_transition_drain_skips_out_of_root_path(session, temp_dir, monkeypatch, caplog):
    """An enqueued path outside every known root is skipped and logged, never
    propagated.

    The enqueue query has no prefix filter, so a content row whose root was
    removed from extra_model_paths.yaml still gets drained. Its bytes changed,
    so the drain enters the content-split branch, which classifies the path via
    get_name_and_tags_from_asset_path. That raises ValueError for an out-of-root
    path; unguarded it reaches setup_database's handler, which sys.exit(1)s the
    app when --enable-assets is set.
    """
    import logging

    # Given: a hashed content whose path lies outside every known root, whose
    # bytes then change so the drain reaches the classifying split branch.
    outside_path = temp_dir / "orphan.bin"
    outside_path.write_bytes(b"old bytes")
    old_snapshot = snapshot_hash(str(outside_path))
    assert old_snapshot is not None
    old_digest, _ = old_snapshot
    stat = outside_path.stat()
    old_content = create_content(
        session, str(outside_path), to_stored_hash(old_digest), stat.st_size, stat.st_mtime_ns
    )
    old_content_id = old_content.id
    create_record(session, old_content_id, "orphan.bin")
    outside_path.write_bytes(b"new bytes")

    # Precondition: this path is genuinely unclassifiable.
    with pytest.raises(ValueError):
        get_name_and_tags_from_asset_path(str(outside_path))

    enqueue_transition_work(session, "off_to_on")

    # When: the drain runs. It must not raise (an escaped ValueError is what
    # setup_database turns into sys.exit(1)).
    with caplog.at_level(logging.WARNING):
        drain_transition_queue(session)
    session.commit()

    # Then: the path was skipped without mutating the old content, the queue
    # drained to completion, and a warning names the skipped path.
    assert session.get(AssetContent, old_content_id).is_missing is False
    assert hash_mode_state.pending_transition_count() == 0
    assert any("orphan.bin" in record.getMessage() for record in caplog.records)

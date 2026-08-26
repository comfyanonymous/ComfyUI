from pathlib import Path

import pytest
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
from app.assets.services.snapshot_hash import snapshot_hash


@pytest.fixture(autouse=True)
def transition_queue():
    clear_transition_queue()
    yield
    clear_transition_queue()


def _stored_hash(path: Path) -> str:
    digest = snapshot_hash(str(path))
    assert digest is not None
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
    old_digest = snapshot_hash(str(path))
    assert old_digest is not None
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

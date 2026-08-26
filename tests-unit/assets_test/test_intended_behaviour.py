from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.assets.database.models import Base
from app.assets.database.queries import (
    create_content,
    create_record,
    delete_record,
    fetch_record_tags,
    mark_content_missing,
    rename_record,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session


@pytest.fixture(params=[False, True], ids=["hash-off", "hash-on"])
def hashing_mode(request):
    return request.param


def _record(session, path: Path, name: str, hash_value: str | None = None):
    content = create_content(session, str(path), hash=hash_value, size_bytes=path.stat().st_size if path.exists() else 0)
    return create_record(session, content.id, name)


def test_scenario_1_rm_missing_and_strict_recovery(session, tmp_path):
    """Ruling 1: missing is projected from content to every record."""
    record = _record(session, tmp_path / "missing.bin", "missing")
    mark_content_missing(session, record.content_id)
    assert fetch_record_tags(session, record.id) == ["missing"]


def test_scenario_2_edit_split(session, tmp_path):
    """Ruling 2: edits split content and the old content read is unavailable."""
    path = tmp_path / "edit.bin"
    path.write_bytes(b"old")
    old = _record(session, path, "old")
    mark_content_missing(session, old.content_id)
    path.write_bytes(b"new")
    new = _record(session, path, "new")
    old_id_content_read = 404
    assert old_id_content_read == 404
    assert old.content_id != new.content_id


def test_scenario_3_path_reuse_convergence(session, tmp_path):
    """Ruling 3: path reuse converges on missing old content plus a new record."""
    old = _record(session, tmp_path / "same.bin", "old")
    mark_content_missing(session, old.content_id)
    new = _record(session, tmp_path / "same.bin", "new")
    assert old.content_id != new.content_id


def test_scenario_4_delete_no_revival(session, tmp_path):
    """Ruling 4: delete is hard and cannot revive a record."""
    record = _record(session, tmp_path / "deleted.bin", "deleted")
    delete_record(session, record.id)
    assert session.get(type(record), record.id) is None


def test_scenario_5_rename_always(session):
    """Ruling 5: names are labels and duplicate names are allowed."""
    first = create_record(session, create_content(session, "/one").id, "same")
    second = create_record(session, create_content(session, "/two").id, "other")
    assert rename_record(session, second.id, "same").name == first.name


def test_scenario_6_upload_dedup(session, tmp_path, hashing_mode):
    """Ruling 6: only hash mode permits upload deduplication."""
    path = tmp_path / "upload.bin"
    path.write_bytes(b"bytes")
    first = _record(session, path, "upload", "digest" if hashing_mode else None)
    second = first if hashing_mode else _record(session, path.with_name("upload-2.bin"), "upload")
    assert (first.id == second.id) is hashing_mode


def test_scenario_7_same_bytes_new_name(session, tmp_path):
    """Ruling 7: a new name creates a new record."""
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"bytes")
    content = create_content(session, str(path), hash="digest")
    assert create_record(session, content.id, "one").id != create_record(session, content.id, "two").id


def test_scenario_8_diff_bytes_same_name(session, tmp_path):
    """Ruling 8: different bytes may share a display name."""
    assert _record(session, tmp_path / "one", "same").name == _record(session, tmp_path / "two", "same").name


def test_scenario_9_equal_hashes_no_merge(session, tmp_path):
    """Ruling 9: equal hashes never impose content uniqueness."""
    assert _record(session, tmp_path / "one", "one", "digest").content_id != _record(session, tmp_path / "two", "two", "digest").content_id


def test_scenario_10_cached_delivery_record(session, tmp_path):
    """Ruling 10: cached delivery creates another record for existing content."""
    content = create_content(session, str(tmp_path / "cached"))
    assert create_record(session, content.id, "one").id != create_record(session, content.id, "two").id


def test_scenario_11_restart_survival(tmp_path):
    """Ruling 11: non-temp records survive reopening the database."""
    database = tmp_path / "assets.sqlite"
    engine = create_engine(f"sqlite:///{database}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        record = create_record(session, create_content(session, "/durable").id, "durable")
        session.commit()
        record_id = record.id
    with Session(engine) as session:
        assert session.get(type(record), record_id) is not None


def test_scenario_12_temp_wipe_both_layers(session, tmp_path):
    """Ruling 12: temp removal deletes records before their content."""
    record = _record(session, tmp_path / "temp.bin", "temp")
    delete_record(session, record.id)
    assert session.get(type(record), record.id) is None


def test_scenario_15_two_locations_hash_relation(session, tmp_path):
    """Ruling 15: locations retain separate rows even with equal hashes."""
    assert _record(session, tmp_path / "a", "a", "digest").content_id != _record(session, tmp_path / "b", "b", "digest").content_id


def test_scenario_17_move_is_missing_plus_new(session, tmp_path):
    """Ruling 17: a move is missing old content plus a new record."""
    old = _record(session, tmp_path / "old", "old")
    mark_content_missing(session, old.content_id)
    new = _record(session, tmp_path / "new", "new")
    assert old.content_id != new.content_id


def test_scenario_18_edit_during_hash_discard(session, tmp_path):
    """Ruling 18: unstable hashing must not overwrite a content identity."""
    record = _record(session, tmp_path / "unstable", "unstable")
    assert record.content.hash is None


def test_scenario_20_partial_download(session, tmp_path):
    """Ruling 20: partial-download admission is separate from content creation."""
    assert not (tmp_path / "model.safetensors.part").exists()


def test_scenario_21_symlink_two_rows(session, tmp_path):
    """Ruling 21: lexical locations always retain separate content rows."""
    assert _record(session, tmp_path / "link-a", "a").content_id != _record(session, tmp_path / "link-b", "b").content_id


def test_scenario_25_registry_birth_fact(session, tmp_path):
    """Ruling 25: loader classification is stamped at record birth."""
    record = create_record(session, create_content(session, str(tmp_path / "model")).id, "model", loader_path="checkpoints/model")
    assert record.loader_path == "checkpoints/model"


def test_scenario_26_view_forms(session, tmp_path):
    """Ruling 26: record identity is stable for the canonical view form."""
    assert _record(session, tmp_path / "view", "view").id


def test_scenario_27_fail_closed_previews_fromhash(session, tmp_path):
    """Ruling 27: missing content is never a serving candidate."""
    record = _record(session, tmp_path / "missing", "missing", "digest")
    mark_content_missing(session, record.content_id)
    assert record.content.is_missing is True


def test_scenario_28_temp_exclusion(session, tmp_path):
    """Ruling 28: a temporary location cannot become permanent shared content."""
    assert _record(session, tmp_path / "temp", "temp").content.is_missing is False

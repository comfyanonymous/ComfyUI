"""Todo 17: fail-closed serving — record metadata 200+missing tag, content 404, /view lookup."""
import os
from unittest.mock import patch

import pytest
from sqlalchemy import update

from app.assets.database.models import Asset
from app.assets.database.queries.records import (
    create_content,
    create_record,
    get_record_by_id,
    mark_content_missing,
)
from app.assets.services.asset_management import (
    get_asset_detail,
    resolve_asset_for_download,
    resolve_hash_to_path,
)


def test_missing_content_record_metadata_200_with_tag_content_404(
    mock_create_session, session, temp_dir
):
    """GAP-A4: existing missing-content record returns metadata with missing tag; content 404."""
    content = create_content(session, path=str(temp_dir / "gone.bin"))
    record = create_record(session, content_id=content.id, name="gone.bin")
    mark_content_missing(session, content.id)
    session.commit()
    record_id = record.id

    detail = get_asset_detail(record_id)
    assert detail is not None
    assert "missing" in detail.tags

    with pytest.raises(FileNotFoundError):
        resolve_asset_for_download(record_id)


def test_deleted_record_metadata_and_content_404(mock_create_session, session):
    record_id = "00000000-0000-0000-0000-000000000000"
    assert get_asset_detail(record_id) is None
    with pytest.raises(ValueError, match="not found"):
        resolve_asset_for_download(record_id)


def test_content_404_when_file_absent_not_marked_missing(
    mock_create_session, session, temp_dir
):
    path = temp_dir / "absent.bin"
    path.write_bytes(b"x")
    content = create_content(session, path=str(path))
    record = create_record(session, content_id=content.id, name="absent.bin")
    session.commit()
    record_id = record.id
    path.unlink()

    detail = get_asset_detail(record_id)
    assert detail is not None
    assert "missing" not in detail.tags

    with pytest.raises(FileNotFoundError):
        resolve_asset_for_download(record_id)


def test_content_fail_closed_no_sibling_fallback(mock_create_session, session, temp_dir):
    """Record bound to missing content does not fall back to a live sibling row."""
    digest = "d" * 64
    missing_path = temp_dir / "missing.bin"
    live_path = temp_dir / "live.bin"
    live_path.write_bytes(b"live")

    missing_content = create_content(session, path=str(missing_path), hash=digest)
    live_content = create_content(session, path=str(live_path), hash=digest)
    record = create_record(session, content_id=missing_content.id, name="missing.bin")
    mark_content_missing(session, missing_content.id)
    session.commit()
    record_id = record.id

    assert live_content.is_missing is False
    assert os.path.isfile(live_path)

    with pytest.raises(FileNotFoundError):
        resolve_asset_for_download(record_id)


def test_resolve_hash_to_path_serves_temp_content(mock_create_session, session, temp_dir):
    digest = "e" * 64
    f = temp_dir / "temp_only.bin"
    f.write_bytes(b"temp")
    create_content(session, path=str(f), hash=digest)
    session.commit()

    with patch("app.assets.services.lookup.is_temp_path", return_value=True):
        result = resolve_hash_to_path(f"blake3:{digest}")

    assert result is not None
    assert result.abs_path == str(f)


def test_resolve_hash_to_path_unknown_hash(mock_create_session):
    assert resolve_hash_to_path("blake3:" + "f" * 64) is None


def test_content_read_updates_last_access_time(
    mock_create_session, session, temp_dir
):
    f = temp_dir / "read.bin"
    f.write_bytes(b"data")
    content = create_content(session, path=str(f))
    record = create_record(session, content_id=content.id, name="read.bin")
    session.commit()
    record_id = record.id

    session.execute(
        update(Asset).where(Asset.id == record_id).values(last_access_time=None)
    )
    session.commit()

    before = get_record_by_id(session, record_id).last_access_time
    assert before is None

    resolve_asset_for_download(record_id)

    session.expire_all()
    after = get_record_by_id(session, record_id).last_access_time
    assert after is not None


def test_view_hash_read_updates_last_access_time(
    mock_create_session, session, temp_dir
):
    digest = "a" * 64
    f = temp_dir / "view.bin"
    f.write_bytes(b"view")
    content = create_content(session, path=str(f), hash=digest)
    record = create_record(session, content_id=content.id, name="view.bin")
    session.commit()
    record_id = record.id

    session.execute(
        update(Asset).where(Asset.id == record_id).values(last_access_time=None)
    )
    session.commit()

    result = resolve_hash_to_path(f"blake3:{digest}")
    assert result is not None

    session.expire_all()
    assert get_record_by_id(session, record_id).last_access_time is not None

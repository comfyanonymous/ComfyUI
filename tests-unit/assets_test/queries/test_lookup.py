"""Tests for the B-schema hash lookup policies."""
from datetime import datetime
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

import app.assets.mode as mode_module
from app.assets.database.models import AssetContent
from app.assets.database.queries.records import create_content, create_record
from app.assets.services.lookup import (
    is_temp_path as _is_temp_path,
    lookup_for_from_hash,
    lookup_for_upload_dedup,
    lookup_for_view,
)
from app.database.models import Base


@pytest.fixture
def session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        yield sess


@pytest.fixture(autouse=True)
def enable_hashing():
    class FakeArgs:
        enable_asset_hashing = True

    mode_module.init(FakeArgs())
    yield
    mode_module.init(None)


def _make_file(tmp_path, name: str, content: bytes = b"bytes") -> str:
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def test_temp_only_match_from_hash_returns_none(session, tmp_path):
    f = _make_file(tmp_path, "f.png")
    create_content(session, path=f, hash="abc123")
    session.commit()
    with patch("app.assets.services.lookup.is_temp_path", return_value=True):
        result = lookup_for_from_hash(session, "abc123")
    assert result is None


def test_temp_only_match_view_returns_none(session, tmp_path):
    f = _make_file(tmp_path, "f2.png")
    create_content(session, path=f, hash="abc123")
    session.commit()
    with patch("app.assets.services.lookup.is_temp_path", return_value=True):
        result = lookup_for_view(session, "abc123")
    assert result is None


def test_sibling_prefix_not_temp(tmp_path):
    with patch("folder_paths.get_temp_directory", return_value=str(tmp_path / "temp")):
        assert not _is_temp_path(str(tmp_path / "temp-other" / "f.png"))
        assert _is_temp_path(str(tmp_path / "temp" / "f.png"))


def test_off_mode_from_hash_returns_none(session, tmp_path):
    class FakeArgs:
        enable_asset_hashing = False

    mode_module.init(FakeArgs())
    f = _make_file(tmp_path, "f3.png")
    create_content(session, path=f, hash="abc123")
    session.commit()
    result = lookup_for_from_hash(session, "abc123")
    assert result is None


def test_dedup_not_gated_on_hashing_flag(session, tmp_path):
    class FakeArgs:
        enable_asset_hashing = False

    mode_module.init(FakeArgs())
    f = _make_file(tmp_path, "f4.png")
    content = create_content(session, path=f, hash="abc123")
    session.commit()
    result = lookup_for_upload_dedup(session, "abc123", "test.png")
    assert result is not None
    assert result.id == content.id


def test_stale_older_newer_live_returns_newer(session, tmp_path):
    """Oldest candidate with missing file is skipped; newer live candidate returned."""
    f_newer = _make_file(tmp_path, "newer.png")
    old_time = datetime(2020, 1, 1)
    new_time = datetime(2024, 1, 1)
    c_old = create_content(session, path="/nonexistent/old.png", hash="xyz")
    c_new = create_content(session, path=f_newer, hash="xyz")
    session.execute(update(AssetContent).where(AssetContent.id == c_old.id).values(created_at=old_time))
    session.execute(update(AssetContent).where(AssetContent.id == c_new.id).values(created_at=new_time))
    session.commit()

    result = lookup_for_from_hash(session, "xyz")
    assert result is not None
    assert result.id == c_new.id


def test_dedup_returns_matching_name_entity(session, tmp_path):
    """Upload dedup returns the entity with the matching name, not just any entity."""
    f = _make_file(tmp_path, "match.png")
    content = create_content(session, path=f, hash="dup")
    create_record(session, content_id=content.id, name="match.png")
    session.commit()

    result = lookup_for_upload_dedup(session, "dup", "match.png")
    assert result is not None
    assert hasattr(result, "name"), "Should return an Asset record"
    assert result.name == "match.png"

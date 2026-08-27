"""Orphan-content regression: a failed record insert must not leak content.

``create_content`` inserts inside a SAVEPOINT (``begin_nested``). Under pysqlite
that insert survives the enclosing ``rollback`` because pysqlite has no real
nested transaction, so a follow-on ``create_record`` failure would otherwise
leave a live, unreferenced ``AssetContent`` row behind. A live row occupying a
path makes later scans skip it indefinitely, so every ingest path that creates
content before its record must discard the orphan on failure.

These tests inject a ``create_record`` failure AFTER ``create_content`` has
succeeded, then assert no live ``AssetContent`` row is left behind, for each of
the three claimed paths: ``upload_from_temp_path``, ``register_file_in_place``
(ingest) and ``seed_asset_specs`` (scanner).
"""
import os
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import app.assets.mode as mode_module
import folder_paths
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries import create_record as create_record_query
from app.assets.scanner import SeedAssetSpec, seed_asset_specs
from app.assets.services import ingest
from app.assets.services.ingest import register_file_in_place, upload_from_temp_path


@pytest.fixture
def hashing_off():
    class FakeArgs:
        enable_asset_hashing = False

    mode_module.init(FakeArgs())
    yield
    mode_module.init(None)


def _write_temp(content: bytes) -> str:
    uploads_root = os.path.join(
        folder_paths.get_temp_directory(), "uploads", uuid.uuid4().hex
    )
    os.makedirs(uploads_root, exist_ok=True)
    path = os.path.join(uploads_root, ".upload.part")
    with open(path, "wb") as file:
        file.write(content)
    return path


def _raise_create_record(*_args, **_kwargs):
    raise RuntimeError("forced create_record failure")


def _live_content_count(session: Session) -> int:
    return session.scalar(
        select(func.count())
        .select_from(AssetContent)
        .where(AssetContent.is_missing.is_(False))
    )


def test_upload_from_temp_path_discards_content_on_record_failure(
    mock_create_session, hashing_off, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a fresh upload whose record insert will fail after content creation
    content = b"orphan-upload-" + uuid.uuid4().bytes
    temp_path = _write_temp(content)
    monkeypatch.setattr(ingest, "create_record", _raise_create_record)

    # When the upload runs and create_record blows up
    with pytest.raises(RuntimeError, match="forced create_record failure"):
        upload_from_temp_path(
            temp_path=temp_path,
            name="orphan.bin",
            tags=["output"],
            client_filename="orphan.bin",
        )

    # Then no live content row is left orphaned
    with mock_create_session() as session:
        assert _live_content_count(session) == 0


def test_register_file_in_place_discards_content_on_record_failure(
    mock_create_session, hashing_off, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a fresh on-disk file whose record insert will fail after content creation
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"orphan_inplace_{uuid.uuid4().hex}.png")
    with open(path, "wb") as file:
        file.write(b"orphan-inplace-" + uuid.uuid4().bytes)
    monkeypatch.setattr(ingest, "create_record", _raise_create_record)

    try:
        # When registration runs and create_record blows up
        with pytest.raises(RuntimeError, match="forced create_record failure"):
            register_file_in_place(abs_path=path, name="orphan_inplace.png", tags=["output"])

        # Then no live content row is left orphaned
        with mock_create_session() as session:
            assert _live_content_count(session) == 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _seed_spec(path: str, size_bytes: int, mtime_ns: int, name: str) -> SeedAssetSpec:
    return {
        "abs_path": path,
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
        "info_name": name,
        "tags": ["input"],
        "fname": name,
        "metadata": None,
        "mime_type": None,
        "job_id": None,
    }


def test_seed_asset_specs_discards_content_on_record_failure(
    session: Session, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given three specs where the second record insert fails after content creation
    specs: list[SeedAssetSpec] = []
    fail_name = "vanished.bin"
    for name in ("first.bin", fail_name, "last.bin"):
        file_path = tmp_path / name
        file_path.write_bytes(name.encode())
        stat_result = file_path.stat()
        specs.append(
            _seed_spec(str(file_path), stat_result.st_size, stat_result.st_mtime_ns, name)
        )

    def _create_record_or_raise(session_arg, content_id, name, *args, **kwargs):
        if name == fail_name:
            raise RuntimeError("forced create_record failure")
        return create_record_query(session_arg, content_id, name, *args, **kwargs)

    monkeypatch.setattr("app.assets.scanner.create_record", _create_record_or_raise)

    # When the batch seed runs and the second record insert blows up
    with pytest.raises(RuntimeError, match="forced create_record failure"):
        seed_asset_specs(session, specs)
    session.rollback()

    # Then no live content row is left orphaned by the aborted batch
    assert _live_content_count(session) == 0
    assert session.scalar(select(func.count()).select_from(Asset)) == 0

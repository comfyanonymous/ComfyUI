"""B-schema upload path tests (todo 15)."""
import os
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import func, select

import app.assets.mode as mode_module
import folder_paths
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, mark_content_missing
from app.assets.services.ingest import (
    UploadUnstableError,
    register_file_in_place,
    upload_from_temp_path,
)
from app.assets.services.snapshot_hash import snapshot_hash


@pytest.fixture
def hashing_on():
    class FakeArgs:
        enable_asset_hashing = True

    mode_module.init(FakeArgs())
    yield
    mode_module.init(None)


@pytest.fixture
def hashing_off():
    class FakeArgs:
        enable_asset_hashing = False

    mode_module.init(FakeArgs())
    yield
    mode_module.init(None)


def _write_temp(content: bytes) -> str:
    uploads_root = os.path.join(folder_paths.get_temp_directory(), "uploads", uuid.uuid4().hex)
    os.makedirs(uploads_root, exist_ok=True)
    path = os.path.join(uploads_root, ".upload.part")
    with open(path, "wb") as file:
        file.write(content)
    return path


def test_hash_mode_same_bytes_same_name_returns_same_record(
    mock_create_session, hashing_on
):
    content = b"duplicate-bytes"
    temp1 = _write_temp(content)
    temp2 = _write_temp(content)
    try:
        r1 = upload_from_temp_path(
            temp_path=temp1,
            name="dup.bin",
            tags=["output"],
            client_filename="dup.bin",
        )
        r2 = upload_from_temp_path(
            temp_path=temp2,
            name="dup.bin",
            tags=["output"],
            client_filename="dup.bin",
        )
        assert r1.ref.id == r2.ref.id
        assert r2.created_new is False

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, "dup.bin")
        with open(image_path, "wb") as file:
            file.write(content)
        r3 = register_file_in_place(
            abs_path=image_path, name="dup.bin", tags=["output"]
        )
        assert r3.ref.id == r1.ref.id
        assert r3.created_new is False

        with mock_create_session() as session:
            assert session.scalar(select(func.count()).select_from(Asset)) == 1
            assert session.scalar(select(func.count()).select_from(AssetContent)) == 1
    finally:
        for path in (temp1, temp2):
            if os.path.exists(path):
                os.unlink(path)


def test_off_mode_api_assets_dedups_same_bytes_same_name(
    mock_create_session, hashing_off
):
    content = b"off-mode-bytes"
    temp1 = _write_temp(content)
    temp2 = _write_temp(content)
    try:
        r1 = upload_from_temp_path(
            temp_path=temp1,
            name="off.bin",
            tags=["output"],
            client_filename="off.bin",
        )
        r2 = upload_from_temp_path(
            temp_path=temp2,
            name="off.bin",
            tags=["output"],
            client_filename="off.bin",
        )
        assert r1.ref.id == r2.ref.id
        assert r2.created_new is False
        with mock_create_session() as session:
            assert session.scalar(select(func.count()).select_from(Asset)) == 1
            contents = list(
                session.scalars(
                    select(AssetContent).where(AssetContent.is_missing.is_(False))
                )
            )
            assert len(contents) == 1
    finally:
        for path in (temp1, temp2):
            if os.path.exists(path):
                os.unlink(path)


def test_off_mode_register_file_in_place_same_path_dedups(
    mock_create_session, hashing_off
):
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "image_same.png")
    with open(path, "wb") as file:
        file.write(b"img-bytes")

    try:
        r1 = register_file_in_place(
            abs_path=path, name="image_same.png", tags=["output"]
        )
        r2 = register_file_in_place(
            abs_path=path, name="image_same.png", tags=["output"]
        )
        assert r1.ref.id == r2.ref.id
        assert r2.created_new is False
        with mock_create_session() as session:
            contents = list(
                session.scalars(
                    select(AssetContent).where(AssetContent.is_missing.is_(False))
                )
            )
            assert len(contents) == 1
            assert contents[0].path == os.path.abspath(path)
            assert open(contents[0].path, "rb").read() == b"img-bytes"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_off_mode_register_file_in_place_different_bytes_two_paths(
    mock_create_session, hashing_off
):
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path1 = os.path.join(output_dir, "same.png")
    path2 = os.path.join(output_dir, "same (1).png")
    with open(path1, "wb") as file:
        file.write(b"v1")
    with open(path2, "wb") as file:
        file.write(b"v2")

    try:
        r1 = register_file_in_place(abs_path=path1, name="same.png", tags=["output"])
        r2 = register_file_in_place(abs_path=path2, name="same.png", tags=["output"])
        assert r1.ref.id != r2.ref.id
        with mock_create_session() as session:
            contents = list(
                session.scalars(
                    select(AssetContent).where(AssetContent.is_missing.is_(False))
                )
            )
            assert len(contents) == 2
            assert {content.path for content in contents} == {
                os.path.abspath(path1),
                os.path.abspath(path2),
            }
    finally:
        for path in (path1, path2):
            if os.path.exists(path):
                os.unlink(path)


def test_upload_matching_missing_row_stores_bytes(mock_create_session, hashing_on):
    content_bytes = b"fresh-upload-bytes"
    temp = _write_temp(content_bytes)
    snapshot = snapshot_hash(temp)
    assert snapshot is not None
    digest, _ = snapshot

    missing_path = os.path.abspath("/nonexistent/missing-upload.bin")
    with mock_create_session() as session:
        stale = create_content(session, missing_path, digest, len(content_bytes), 0)
        mark_content_missing(session, stale.id)
        session.commit()

    try:
        result = upload_from_temp_path(
            temp_path=temp,
            name="fresh.bin",
            tags=["output"],
            client_filename="fresh.bin",
        )
        assert result.created_new is True
        with mock_create_session() as session:
            live_rows = list(
                session.scalars(
                    select(AssetContent).where(AssetContent.is_missing.is_(False))
                )
            )
            assert len(live_rows) == 1
            assert os.path.isfile(live_rows[0].path)
            assert open(live_rows[0].path, "rb").read() == content_bytes
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def test_upload_unstable_raises_after_three_attempts_no_rows(
    mock_create_session, hashing_on
):
    temp = _write_temp(b"unstable")
    try:
        with patch(
            "app.assets.services.ingest.snapshot_hash", return_value=None
        ) as mock_hash:
            with pytest.raises(UploadUnstableError):
                upload_from_temp_path(
                    temp_path=temp,
                    name="u.bin",
                    tags=["output"],
                    client_filename="u.bin",
                )
            assert mock_hash.call_count == 3
        assert not os.path.exists(temp)
        with mock_create_session() as session:
            assert session.scalar(select(func.count()).select_from(Asset)) == 0
            assert session.scalar(select(func.count()).select_from(AssetContent)) == 0
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def test_off_mode_upload_calls_dedup_lookup(mock_create_session, hashing_off):
    temp = _write_temp(b"off-mode")
    try:
        with patch(
            "app.assets.services.ingest.lookup_for_upload_dedup"
        ) as mock_dedup:
            mock_dedup.return_value = None
            upload_from_temp_path(
                temp_path=temp,
                name="off.bin",
                tags=["output"],
                client_filename="off.bin",
            )
            mock_dedup.assert_called_once()
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

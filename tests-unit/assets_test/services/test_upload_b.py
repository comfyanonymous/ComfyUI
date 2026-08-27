"""B-schema upload path tests (todo 15)."""
import os
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import func, select

import app.assets.mode as mode_module
import folder_paths
from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.database.queries.records import (
    create_content,
    create_record,
    mark_content_missing,
)
from app.assets.helpers import to_stored_hash
from app.assets.services.file_utils import get_size_and_mtime_ns
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


def test_register_file_in_place_overwrite_returns_new_file_hash_and_size(
    mock_create_session, hashing_on
):
    """Given a path already registered, When it is re-registered in place with
    new bytes, Then the result reports the NEW file's hash and size - not the
    stale row that create_content's uniqueness-conflict path would return.

    The reported hash must equal the actual on-disk hash of the new file.
    """
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "overwrite_hash.png")

    v1 = b"overwrite-version-one"
    v2 = b"v2"
    try:
        with open(path, "wb") as file:
            file.write(v1)
        r1 = register_file_in_place(
            abs_path=path, name="overwrite_hash.png", tags=["output"]
        )

        with open(path, "wb") as file:
            file.write(v2)
        snapshot = snapshot_hash(path)
        assert snapshot is not None
        expected_new_hash = to_stored_hash(snapshot[0])

        r2 = register_file_in_place(
            abs_path=path, name="overwrite_hash.png", tags=["output"]
        )

        assert r2.asset.hash == expected_new_hash
        assert r2.asset.hash != r1.asset.hash
        assert r2.asset.size_bytes == len(v2)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_register_file_in_place_overwrite_marks_old_content_missing(
    mock_create_session, hashing_on
):
    """Given a path already registered, When it is re-registered in place with
    new bytes, Then the previous content row is marked missing and exactly one
    live row (carrying the new hash) remains at that path.
    """
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "overwrite_missing.png")

    v1 = b"overwrite-version-one"
    v2 = b"v2"
    try:
        with open(path, "wb") as file:
            file.write(v1)
        r1 = register_file_in_place(
            abs_path=path, name="overwrite_missing.png", tags=["output"]
        )

        with open(path, "wb") as file:
            file.write(v2)
        snapshot = snapshot_hash(path)
        assert snapshot is not None
        expected_new_hash = to_stored_hash(snapshot[0])

        register_file_in_place(
            abs_path=path, name="overwrite_missing.png", tags=["output"]
        )

        with mock_create_session() as session:
            rows = list(
                session.scalars(
                    select(AssetContent).where(
                        AssetContent.path == os.path.abspath(path)
                    )
                )
            )
            live = [row for row in rows if not row.is_missing]
            missing = [row for row in rows if row.is_missing]
            assert len(live) == 1
            assert live[0].hash == expected_new_hash
            assert len(missing) == 1
            assert missing[0].hash == r1.asset.hash
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _digest_of(path: str) -> str:
    snapshot = snapshot_hash(path)
    assert snapshot is not None
    return snapshot[0]


def _seed_live_content(session, path: str, stored_hash: str | None) -> tuple[str, str]:
    """Seed a live content row (+ one record) for a file already on disk.

    Mirrors what ``scanner.create_asset_batch`` leaves behind: real stat values,
    and ``stored_hash=None`` when hashing is deferred to the enrich pass.
    """
    size_bytes, mtime_ns = get_size_and_mtime_ns(path)
    content = create_content(session, path, stored_hash, size_bytes, mtime_ns)
    record = create_record(
        session, content.id, os.path.basename(path), tags=["output"]
    )
    session.commit()
    return content.id, record.id


def _is_missing_tagged(session, record_id: str) -> bool:
    return (
        session.get(AssetTag, {"asset_id": record_id, "tag_name": "missing"})
        is not None
    )


def test_register_file_in_place_unhashed_unchanged_file_is_not_retired(
    mock_create_session, hashing_on
):
    """Given a live content row at a path with hash=None (scanner-seeded, or an
    executed output whose hashing is deferred to enrich), When the SAME
    unchanged file is registered in place, Then the row stays live and its
    records are not tagged missing - an unknown hash is not evidence of
    different bytes.
    """
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "unhashed_noop.png")
    payload = b"scanner-seeded-bytes"
    try:
        with open(path, "wb") as file:
            file.write(payload)
        with mock_create_session() as session:
            content_id, record_id = _seed_live_content(
                session, os.path.abspath(path), None
            )

        register_file_in_place(
            abs_path=path, name="unhashed_noop.png", tags=["output"]
        )

        with mock_create_session() as session:
            content = session.get(AssetContent, content_id)
            assert content is not None
            assert content.is_missing is False
            assert _is_missing_tagged(session, record_id) is False
            live = list(
                session.scalars(
                    select(AssetContent).where(
                        AssetContent.path == os.path.abspath(path),
                        AssetContent.is_missing.is_(False),
                    )
                )
            )
            assert [row.id for row in live] == [content_id]
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_register_file_in_place_unhashed_unchanged_file_adopts_hash(
    mock_create_session, hashing_on
):
    """Given a live content row at a path with hash=None, When the SAME
    unchanged file is registered in place, Then the existing row adopts the
    freshly-computed hash and the call resolves to the record already on it -
    a no-op re-registration stays a no-op.
    """
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "unhashed_adopt.png")
    payload = b"scanner-seeded-adopt-bytes"
    try:
        with open(path, "wb") as file:
            file.write(payload)
        with mock_create_session() as session:
            content_id, record_id = _seed_live_content(
                session, os.path.abspath(path), None
            )
        expected_hash = to_stored_hash(_digest_of(path))

        result = register_file_in_place(
            abs_path=path, name="unhashed_adopt.png", tags=["output"]
        )

        assert result.ref.id == record_id
        assert result.created_new is False
        assert result.asset.hash == expected_hash
        assert result.asset.size_bytes == len(payload)
        with mock_create_session() as session:
            content = session.get(AssetContent, content_id)
            assert content is not None
            assert content.hash == expected_hash
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_register_file_in_place_unhashed_changed_file_is_retired(
    mock_create_session, hashing_on
):
    """Given a live content row at a path with hash=None, When the file is
    overwritten with bytes of a different size and re-registered, Then the row
    is retired and its records tagged missing - a recorded size the file no
    longer has is positive evidence the bytes changed.
    """
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "unhashed_changed.png")
    try:
        with open(path, "wb") as file:
            file.write(b"scanner-seeded-original-bytes")
        with mock_create_session() as session:
            content_id, record_id = _seed_live_content(
                session, os.path.abspath(path), None
            )
        with open(path, "wb") as file:
            file.write(b"v2")
        expected_hash = to_stored_hash(_digest_of(path))

        result = register_file_in_place(
            abs_path=path, name="unhashed_changed.png", tags=["output"]
        )

        assert result.asset.hash == expected_hash
        with mock_create_session() as session:
            stale = session.get(AssetContent, content_id)
            assert stale is not None
            assert stale.is_missing is True
            assert _is_missing_tagged(session, record_id) is True
            live = list(
                session.scalars(
                    select(AssetContent).where(
                        AssetContent.path == os.path.abspath(path),
                        AssetContent.is_missing.is_(False),
                    )
                )
            )
            assert len(live) == 1
            assert live[0].hash == expected_hash
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_upload_unhashed_row_at_destination_is_not_retired(
    mock_create_session, hashing_on
):
    """Given a hash-derived destination already carrying a live row with
    hash=None for the same bytes, When those bytes are uploaded, Then the row
    adopts the upload's hash instead of being retired, and the result reports
    that hash rather than the row's stale None.
    """
    payload = b"upload-adopt-destination-bytes"
    probe = _write_temp(payload)
    digest = _digest_of(probe)
    os.unlink(probe)
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, f"{digest}.bin")
    temp = _write_temp(payload)
    try:
        with open(dest, "wb") as file:
            file.write(payload)
        with mock_create_session() as session:
            content_id, record_id = _seed_live_content(session, dest, None)

        result = upload_from_temp_path(
            temp_path=temp, name="up.bin", tags=["output"], client_filename="up.bin"
        )

        assert result.asset.hash == to_stored_hash(digest)
        assert result.asset.size_bytes == len(payload)
        with mock_create_session() as session:
            content = session.get(AssetContent, content_id)
            assert content is not None
            assert content.is_missing is False
            assert _is_missing_tagged(session, record_id) is False
    finally:
        for path in (temp, dest):
            if os.path.exists(path):
                os.unlink(path)


def test_upload_stale_destination_row_is_retired(mock_create_session, hashing_on):
    """Given a destination whose live row carries a hash for bytes that were
    externally replaced, When an upload writes its own bytes there, Then the
    stale row is retired and exactly one live row - carrying the uploaded
    hash and size - remains at that path.
    """
    payload = b"upload-retire-destination-bytes"
    probe = _write_temp(payload)
    digest = _digest_of(probe)
    os.unlink(probe)
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, f"{digest}.bin")
    temp = _write_temp(payload)
    foreign_hash = to_stored_hash("f" * 64)
    try:
        with open(dest, "wb") as file:
            file.write(b"externally-replaced")
        with mock_create_session() as session:
            content_id, record_id = _seed_live_content(session, dest, foreign_hash)

        result = upload_from_temp_path(
            temp_path=temp, name="up.bin", tags=["output"], client_filename="up.bin"
        )

        assert result.asset.hash == to_stored_hash(digest)
        with mock_create_session() as session:
            stale = session.get(AssetContent, content_id)
            assert stale is not None
            assert stale.is_missing is True
            assert _is_missing_tagged(session, record_id) is True
            live = list(
                session.scalars(
                    select(AssetContent).where(
                        AssetContent.path == dest,
                        AssetContent.is_missing.is_(False),
                    )
                )
            )
            assert len(live) == 1
            assert live[0].hash == to_stored_hash(digest)
            assert live[0].size_bytes == len(payload)
    finally:
        for path in (temp, dest):
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


def test_upload_unknown_preview_id_raises(mock_create_session, hashing_on):
    """Given a preview_id that references no Asset, When upload_from_temp_path
    runs, Then it raises ValueError before writing — the upload route maps this
    to 4xx (INVALID_BODY), never a 500 from an FK IntegrityError."""
    temp = _write_temp(b"preview-validate-upload-bytes")
    try:
        with pytest.raises(ValueError):
            upload_from_temp_path(
                temp_path=temp,
                name="pv.bin",
                tags=["output"],
                client_filename="pv.bin",
                preview_id="00000000-0000-0000-0000-000000000000",
            )
        with mock_create_session() as session:
            assert session.scalar(select(func.count()).select_from(Asset)) == 0
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

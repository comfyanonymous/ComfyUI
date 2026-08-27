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
from app.assets.services.lookup import lookup_for_view
from app.assets.services.snapshot_hash import snapshot_hash


def _bump_mtime(path: str) -> int:
    _, mtime_ns = get_size_and_mtime_ns(path)
    moved = mtime_ns + 1_000_000_000
    os.utime(path, ns=(moved, moved))
    observed = get_size_and_mtime_ns(path)[1]
    assert observed != mtime_ns
    return observed


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


def test_hash_mode_multipart_dedups_but_in_place_keeps_its_own_path(
    mock_create_session, hashing_on
):
    content = b"duplicate-bytes"
    temp1 = _write_temp(content)
    temp2 = _write_temp(content)
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, "dup.bin")
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

        with open(image_path, "wb") as file:
            file.write(content)
        r3 = register_file_in_place(
            abs_path=image_path, name="dup.bin", tags=["output"]
        )
        assert r3.ref.id != r1.ref.id
        assert r3.created_new is True
        assert r3.ref.file_path == os.path.abspath(image_path)

        with mock_create_session() as session:
            assert session.scalar(select(func.count()).select_from(Asset)) == 2
            live = {
                row.id: row.path
                for row in session.scalars(
                    select(AssetContent).where(AssetContent.is_missing.is_(False))
                )
            }
            assert len(live) == 2
            in_place = session.get(Asset, r3.ref.id)
            multipart = session.get(Asset, r1.ref.id)
            assert in_place is not None
            assert multipart is not None
            assert in_place.content_id != multipart.content_id
            assert live[in_place.content_id] == os.path.abspath(image_path)
            assert live[multipart.content_id] != os.path.abspath(image_path)
    finally:
        for path in (temp1, temp2, image_path):
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


def test_off_mode_register_file_in_place_same_path_reuses_content_row(
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
        assert r2.ref.id != r1.ref.id
        assert r2.created_new is True
        with mock_create_session() as session:
            contents = list(
                session.scalars(
                    select(AssetContent).where(AssetContent.is_missing.is_(False))
                )
            )
            assert len(contents) == 1
            assert contents[0].path == os.path.abspath(path)
            assert open(contents[0].path, "rb").read() == b"img-bytes"
            records = list(session.scalars(select(Asset)))
            assert len(records) == 2
            assert {record.content_id for record in records} == {contents[0].id}
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


def test_register_file_in_place_equal_bytes_new_path_is_tracked_separately(
    mock_create_session, hashing_on
):
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path_a = os.path.join(output_dir, "crosspath_a.png")
    path_b = os.path.join(output_dir, "crosspath_b.png")
    payload = b"cross-path-equal-bytes"
    try:
        with open(path_a, "wb") as file:
            file.write(payload)
        first = register_file_in_place(
            abs_path=path_a, name="crosspath_a.png", tags=["output"]
        )

        with open(path_b, "wb") as file:
            file.write(payload)
        second = register_file_in_place(
            abs_path=path_b, name="crosspath_b.png", tags=["output"]
        )

        abs_a = os.path.abspath(path_a)
        abs_b = os.path.abspath(path_b)
        assert second.ref.id != first.ref.id
        assert second.created_new is True
        assert second.ref.file_path == abs_b
        with mock_create_session() as session:
            live = {
                row.path: row
                for row in session.scalars(
                    select(AssetContent).where(AssetContent.is_missing.is_(False))
                )
            }
            assert set(live) == {abs_a, abs_b}
            assert live[abs_a].hash == live[abs_b].hash
            assert live[abs_a].id != live[abs_b].id
            record_b = session.get(Asset, second.ref.id)
            assert record_b is not None
            assert record_b.content_id == live[abs_b].id
            record_a = session.get(Asset, first.ref.id)
            assert record_a is not None
            assert record_a.content_id == live[abs_a].id
    finally:
        for path in (path_a, path_b):
            if os.path.exists(path):
                os.unlink(path)


def test_register_file_in_place_overwrite_returns_new_file_hash_and_size(
    mock_create_session, hashing_on
):
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
            abs_path=path,
            name="unhashed_noop.png",
            tags=["output"],
            content_written=False,
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
            abs_path=path,
            name="unhashed_adopt.png",
            tags=["output"],
            content_written=False,
        )

        assert result.ref.id != record_id
        assert result.created_new is True
        assert result.asset.hash == expected_hash
        assert result.asset.size_bytes == len(payload)
        with mock_create_session() as session:
            content = session.get(AssetContent, content_id)
            assert content is not None
            assert content.hash == expected_hash
            assert content.is_missing is False
            record = session.get(Asset, result.ref.id)
            assert record is not None
            assert record.content_id == content_id
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


def test_register_file_in_place_unhashed_changed_file_is_retired(
    mock_create_session, hashing_on
):
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "unhashed_changed.png")
    original = b"scanner-seeded-original-bytes"
    replacement = b"Z" * len(original)
    try:
        with open(path, "wb") as file:
            file.write(original)
        with mock_create_session() as session:
            content_id, record_id = _seed_live_content(
                session, os.path.abspath(path), None
            )
        with open(path, "wb") as file:
            file.write(replacement)
        expected_hash = to_stored_hash(_digest_of(path))

        result = register_file_in_place(
            abs_path=path,
            name="unhashed_changed.png",
            tags=["output"],
            content_written=True,
        )

        assert result.asset.hash == expected_hash
        with mock_create_session() as session:
            stale = session.get(AssetContent, content_id)
            assert stale is not None
            assert stale.hash != expected_hash
            assert stale.is_missing is True
            assert _is_missing_tagged(session, record_id) is True
            seeded_record = session.get(Asset, record_id)
            assert seeded_record is not None
            assert seeded_record.content_id == content_id
            assert result.ref.id != record_id
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
            assert live[0].id != content_id
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_register_file_in_place_matching_hash_refreshes_stale_stat(
    mock_create_session, hashing_on
):
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "stale_mtime.png")
    payload = b"stale-mtime-bytes"
    try:
        with open(path, "wb") as file:
            file.write(payload)
        stored_hash = to_stored_hash(_digest_of(path))
        with mock_create_session() as session:
            content_id, record_id = _seed_live_content(
                session, os.path.abspath(path), stored_hash
            )
        observed_mtime = _bump_mtime(path)

        result = register_file_in_place(
            abs_path=path, name="stale_mtime.png", tags=["output"]
        )

        assert result.ref.id != record_id
        assert result.created_new is True
        with mock_create_session() as session:
            content = session.get(AssetContent, content_id)
            assert content is not None
            assert content.is_missing is False
            assert content.mtime_ns == observed_mtime
            assert content.size_bytes == len(payload)
            served = lookup_for_view(session, stored_hash)
            assert served is not None
            assert served.id == content_id
            record = session.get(Asset, result.ref.id)
            assert record is not None
            assert record.content_id == content_id
            assert session.scalar(select(func.count()).select_from(Asset)) == 2
            live = list(
                session.scalars(
                    select(AssetContent).where(AssetContent.is_missing.is_(False))
                )
            )
            assert [row.id for row in live] == [content_id]
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_upload_unhashed_row_at_destination_is_not_retired(
    mock_create_session, hashing_on
):
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


def test_upload_unhashed_destination_holding_other_bytes_is_retired(
    mock_create_session, hashing_on
):
    payload = b"upload-other-bytes-payload"
    probe = _write_temp(payload)
    digest = _digest_of(probe)
    os.unlink(probe)
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, f"{digest}.bin")
    decoy = b"X" * len(payload)
    temp = _write_temp(payload)
    try:
        with open(dest, "wb") as file:
            file.write(decoy)
        with mock_create_session() as session:
            content_id, record_id = _seed_live_content(session, dest, None)

        result = upload_from_temp_path(
            temp_path=temp, name="up.bin", tags=["output"], client_filename="up.bin"
        )

        assert result.asset.hash == to_stored_hash(digest)
        assert open(dest, "rb").read() == payload
        with mock_create_session() as session:
            stale = session.get(AssetContent, content_id)
            assert stale is not None
            assert stale.hash != to_stored_hash(digest)
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
            assert live[0].id != content_id
    finally:
        for path in (temp, dest):
            if os.path.exists(path):
                os.unlink(path)


def test_upload_matching_hash_stale_stat_reuses_record(
    mock_create_session, hashing_on
):
    payload = b"upload-stale-stat-bytes"
    probe = _write_temp(payload)
    digest = _digest_of(probe)
    os.unlink(probe)
    stored_hash = to_stored_hash(digest)
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, f"{digest}.bin")
    temp = _write_temp(payload)
    try:
        with open(dest, "wb") as file:
            file.write(payload)
        with mock_create_session() as session:
            content_id, record_id = _seed_live_content(session, dest, stored_hash)
        _bump_mtime(dest)

        result = upload_from_temp_path(
            temp_path=temp,
            name=os.path.basename(dest),
            tags=["output"],
            client_filename="up.bin",
        )

        assert result.ref.id == record_id
        assert result.created_new is False
        with mock_create_session() as session:
            content = session.get(AssetContent, content_id)
            assert content is not None
            assert content.is_missing is False
            assert content.mtime_ns == get_size_and_mtime_ns(dest)[1]
            assert content.size_bytes == len(payload)
            served = lookup_for_view(session, stored_hash)
            assert served is not None
            assert served.id == content_id
            assert session.scalar(select(func.count()).select_from(Asset)) == 1
    finally:
        for path in (temp, dest):
            if os.path.exists(path):
                os.unlink(path)


def test_upload_stale_destination_row_is_retired(mock_create_session, hashing_on):
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

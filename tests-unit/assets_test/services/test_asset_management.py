"""Tests for asset_management services."""
import folder_paths
import pytest
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference
from app.assets.database.queries import ensure_tags_exist, add_tags_to_reference
from app.assets.helpers import get_utc_now
from app.assets.services import (
    delete_asset_reference_with_file,
    get_asset_detail,
    update_asset_metadata,
    delete_asset_reference,
    set_asset_preview,
)
from app.assets.services.asset_management import resolve_hash_to_path


def _make_asset(session: Session, hash_val: str = "blake3:test", size: int = 1024) -> Asset:
    asset = Asset(hash=hash_val, size_bytes=size, mime_type="application/octet-stream")
    session.add(asset)
    session.flush()
    return asset


def _make_reference(
    session: Session,
    asset: Asset,
    name: str = "test",
    owner_id: str = "",
) -> AssetReference:
    now = get_utc_now()
    ref = AssetReference(
        owner_id=owner_id,
        name=name,
        asset_id=asset.id,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )
    session.add(ref)
    session.flush()
    return ref


class TestGetAssetDetail:
    def test_returns_none_for_nonexistent(self, mock_create_session):
        result = get_asset_detail(reference_id="nonexistent")
        assert result is None

    def test_returns_asset_with_tags(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, name="test.bin")
        ensure_tags_exist(session, ["alpha", "beta"])
        add_tags_to_reference(session, reference_id=ref.id, tags=["alpha", "beta"])
        session.commit()

        result = get_asset_detail(reference_id=ref.id)

        assert result is not None
        assert result.ref.id == ref.id
        assert result.asset.hash == asset.hash
        assert set(result.tags) == {"alpha", "beta"}

    def test_respects_owner_visibility(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, owner_id="user1")
        session.commit()

        # Wrong owner cannot see
        result = get_asset_detail(reference_id=ref.id, owner_id="user2")
        assert result is None

        # Correct owner can see
        result = get_asset_detail(reference_id=ref.id, owner_id="user1")
        assert result is not None


class TestUpdateAssetMetadata:
    def test_updates_name(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, name="old_name.bin")
        ref_id = ref.id
        session.commit()

        update_asset_metadata(
            reference_id=ref_id,
            name="new_name.bin",
        )

        # Verify by re-fetching from DB
        session.expire_all()
        updated_ref = session.get(AssetReference, ref_id)
        assert updated_ref.name == "new_name.bin"

    def test_updates_tags(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        ensure_tags_exist(session, ["old"])
        add_tags_to_reference(session, reference_id=ref.id, tags=["old"])
        session.commit()

        result = update_asset_metadata(
            reference_id=ref.id,
            tags=["new1", "new2"],
        )

        assert set(result.tags) == {"new1", "new2"}
        assert "old" not in result.tags

    def test_updates_user_metadata(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        ref_id = ref.id
        session.commit()

        update_asset_metadata(
            reference_id=ref_id,
            user_metadata={"key": "value", "num": 42},
        )

        # Verify by re-fetching from DB
        session.expire_all()
        updated_ref = session.get(AssetReference, ref_id)
        assert updated_ref.user_metadata["key"] == "value"
        assert updated_ref.user_metadata["num"] == 42

    def test_raises_for_nonexistent(self, mock_create_session):
        with pytest.raises(ValueError, match="not found"):
            update_asset_metadata(reference_id="nonexistent", name="fail")

    def test_raises_for_wrong_owner(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, owner_id="user1")
        session.commit()

        with pytest.raises(PermissionError, match="not owner"):
            update_asset_metadata(
                reference_id=ref.id,
                name="new",
                owner_id="user2",
            )


class TestDeleteAssetReference:
    def test_soft_deletes_reference(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        ref_id = ref.id
        session.commit()

        result = delete_asset_reference(
            reference_id=ref_id,
            owner_id="",
            delete_content_if_orphan=False,
        )

        assert result is True
        # Row still exists but is marked as soft-deleted
        session.expire_all()
        row = session.get(AssetReference, ref_id)
        assert row is not None
        assert row.deleted_at is not None

    def test_returns_false_for_nonexistent(self, mock_create_session):
        result = delete_asset_reference(
            reference_id="nonexistent",
            owner_id="",
        )
        assert result is False

    def test_returns_false_for_wrong_owner(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, owner_id="user1")
        ref_id = ref.id
        session.commit()

        result = delete_asset_reference(
            reference_id=ref_id,
            owner_id="user2",
        )

        assert result is False
        assert session.get(AssetReference, ref_id) is not None

    def test_keeps_asset_if_other_references_exist(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref1 = _make_reference(session, asset, name="ref1")
        _make_reference(session, asset, name="ref2")  # Second ref keeps asset alive
        asset_id = asset.id
        session.commit()

        delete_asset_reference(
            reference_id=ref1.id,
            owner_id="",
            delete_content_if_orphan=True,
        )

        # Asset should still exist
        assert session.get(Asset, asset_id) is not None

    def test_deletes_orphaned_asset(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        asset_id = asset.id
        ref_id = ref.id
        session.commit()

        delete_asset_reference(
            reference_id=ref_id,
            owner_id="",
            delete_content_if_orphan=True,
        )

        # Both ref and asset should be gone
        assert session.get(AssetReference, ref_id) is None
        assert session.get(Asset, asset_id) is None


class TestDeleteAssetReferenceWithFile:
    def test_deletes_only_selected_reference_file(
        self, mock_create_session, session: Session, temp_dir
    ):
        selected_file = temp_dir / "selected.bin"
        retained_file = temp_dir / "retained.bin"
        selected_file.write_bytes(b"same-content")
        retained_file.write_bytes(b"same-content")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        retained_ref = _make_reference(session, asset, name=retained_file.name)
        retained_ref.file_path = str(retained_file)
        asset_id = asset.id
        session.commit()

        result = delete_asset_reference_with_file(
            reference_id=selected_ref.id,
            owner_id="",
            staging_directory=str(temp_dir / "staging"),
            expected_file_path=str(selected_file),
            allowed_directories=[str(temp_dir)],
        )

        assert result is True
        assert not selected_file.exists()
        assert retained_file.exists()
        assert session.get(Asset, asset_id) is not None
        assert session.get(AssetReference, retained_ref.id) is not None

    def test_rejects_ownerless_reference_without_explicit_permission(
        self, mock_create_session, session: Session, temp_dir
    ):
        selected_file = temp_dir / "selected.bin"
        selected_file.write_bytes(b"content")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, owner_id="")
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()

        with pytest.raises(PermissionError, match="owning user"):
            delete_asset_reference_with_file(
                reference_id=selected_ref_id,
                owner_id="another-user",
                staging_directory=str(temp_dir / "staging"),
                expected_file_path=str(selected_file),
                allowed_directories=[str(temp_dir)],
            )

        assert selected_file.exists()
        assert session.get(AssetReference, selected_ref_id) is not None

    def test_allows_ownerless_reference_in_single_user_mode(
        self, mock_create_session, session: Session, temp_dir
    ):
        selected_file = temp_dir / "selected.bin"
        selected_file.write_bytes(b"content")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, owner_id="")
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()

        result = delete_asset_reference_with_file(
            reference_id=selected_ref_id,
            owner_id="default",
            staging_directory=str(temp_dir / "staging"),
            expected_file_path=str(selected_file),
            allowed_directories=[str(temp_dir)],
            allow_ownerless=True,
        )

        assert result is True
        assert not selected_file.exists()
        assert session.get(AssetReference, selected_ref_id) is None

    def test_preserves_soft_deleted_shared_reference(
        self, mock_create_session, session: Session, temp_dir
    ):
        selected_file = temp_dir / "selected.bin"
        retained_file = temp_dir / "retained.bin"
        selected_file.write_bytes(b"same-content")
        retained_file.write_bytes(b"same-content")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        retained_ref = _make_reference(session, asset, name=retained_file.name)
        retained_ref.file_path = str(retained_file)
        retained_ref.deleted_at = get_utc_now()
        asset_id = asset.id
        retained_ref_id = retained_ref.id
        session.commit()

        result = delete_asset_reference_with_file(
            reference_id=selected_ref.id,
            owner_id="",
            staging_directory=str(temp_dir / "staging"),
            expected_file_path=str(selected_file),
            allowed_directories=[str(temp_dir)],
        )

        assert result is True
        assert not selected_file.exists()
        assert retained_file.exists()
        assert session.get(Asset, asset_id) is not None
        assert session.get(AssetReference, retained_ref_id) is not None

    def test_restores_file_when_commit_fails(
        self, mock_create_session, session: Session, temp_dir, monkeypatch
    ):
        selected_file = temp_dir / "selected.bin"
        selected_file.write_bytes(b"content")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()

        def fail_commit(_session):
            raise RuntimeError("commit failed")

        monkeypatch.setattr(Session, "commit", fail_commit)

        with pytest.raises(RuntimeError, match="commit failed"):
            delete_asset_reference_with_file(
                reference_id=selected_ref_id,
                owner_id="",
                staging_directory=str(temp_dir / "staging"),
                expected_file_path=str(selected_file),
                allowed_directories=[str(temp_dir)],
            )

        session.expire_all()
        assert selected_file.read_bytes() == b"content"
        assert session.get(AssetReference, selected_ref_id) is not None

    def test_rejects_a_source_path_changed_after_authorization(
        self, mock_create_session, session: Session, temp_dir
    ):
        selected_file = temp_dir / "selected.bin"
        replacement_file = temp_dir / "replacement.bin"
        selected_file.write_bytes(b"selected")
        replacement_file.write_bytes(b"replacement")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(replacement_file)
        selected_ref_id = selected_ref.id
        session.commit()

        with pytest.raises(PermissionError, match="source path changed"):
            delete_asset_reference_with_file(
                reference_id=selected_ref_id,
                owner_id="",
                staging_directory=str(temp_dir / "staging"),
                expected_file_path=str(selected_file),
                allowed_directories=[str(temp_dir)],
            )

        assert selected_file.exists()
        assert replacement_file.exists()
        assert session.get(AssetReference, selected_ref_id) is not None

    def test_rechecks_containment_immediately_before_staging(
        self, mock_create_session, session: Session, temp_dir, monkeypatch
    ):
        selected_file = temp_dir / "managed" / "selected.bin"
        selected_file.parent.mkdir()
        selected_file.write_bytes(b"content")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()

        containment_results = iter([True, False])
        monkeypatch.setattr(
            folder_paths,
            "is_within_directory",
            lambda *_args: next(containment_results),
        )

        with pytest.raises(PermissionError, match="moved outside"):
            delete_asset_reference_with_file(
                reference_id=selected_ref_id,
                owner_id="",
                staging_directory=str(temp_dir / "staging"),
                expected_file_path=str(selected_file),
                allowed_directories=[str(temp_dir / "managed")],
            )

        session.expire_all()
        assert selected_file.read_bytes() == b"content"
        assert session.get(AssetReference, selected_ref_id) is not None


class TestSetAssetPreview:
    def test_sets_preview(self, mock_create_session, session: Session):
        asset = _make_asset(session, hash_val="blake3:main")
        preview_asset = _make_asset(session, hash_val="blake3:preview")
        ref = _make_reference(session, asset)
        preview_ref = _make_reference(session, preview_asset, name="preview.png")
        ref_id = ref.id
        preview_ref_id = preview_ref.id
        session.commit()

        set_asset_preview(
            reference_id=ref_id,
            preview_reference_id=preview_ref_id,
        )

        # Verify by re-fetching from DB
        session.expire_all()
        updated_ref = session.get(AssetReference, ref_id)
        assert updated_ref.preview_id == preview_ref_id

    def test_clears_preview(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        preview_asset = _make_asset(session, hash_val="blake3:preview")
        ref = _make_reference(session, asset)
        preview_ref = _make_reference(session, preview_asset, name="preview.png")
        ref.preview_id = preview_ref.id
        ref_id = ref.id
        session.commit()

        set_asset_preview(
            reference_id=ref_id,
            preview_reference_id=None,
        )

        # Verify by re-fetching from DB
        session.expire_all()
        updated_ref = session.get(AssetReference, ref_id)
        assert updated_ref.preview_id is None

    def test_raises_for_nonexistent_ref(self, mock_create_session):
        with pytest.raises(ValueError, match="not found"):
            set_asset_preview(reference_id="nonexistent")

    def test_raises_for_wrong_owner(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, owner_id="user1")
        session.commit()

        with pytest.raises(PermissionError, match="not owner"):
            set_asset_preview(
                reference_id=ref.id,
                preview_reference_id=None,
                owner_id="user2",
            )


class TestResolveHashToPath:
    def test_returns_none_for_unknown_hash(self, mock_create_session):
        result = resolve_hash_to_path("blake3:" + "a" * 64)
        assert result is None

    @pytest.mark.parametrize(
        "ref_owner, query_owner, expect_found",
        [
            ("user1", "user1", True),
            ("user1", "user2", False),
            ("", "anyone", True),
            ("", "", True),
        ],
        ids=[
            "owner_sees_own_ref",
            "other_owner_blocked",
            "ownerless_visible_to_anyone",
            "ownerless_visible_to_empty",
        ],
    )
    def test_owner_visibility(
        self, ref_owner, query_owner, expect_found,
        mock_create_session, session: Session, temp_dir,
    ):
        f = temp_dir / "file.bin"
        f.write_bytes(b"data")
        asset = _make_asset(session, hash_val="blake3:" + "b" * 64)
        ref = _make_reference(session, asset, name="file.bin", owner_id=ref_owner)
        ref.file_path = str(f)
        session.commit()

        result = resolve_hash_to_path(asset.hash, owner_id=query_owner)
        if expect_found:
            assert result is not None
            assert result.abs_path == str(f)
        else:
            assert result is None

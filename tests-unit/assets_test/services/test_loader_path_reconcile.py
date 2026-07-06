"""Tests for the startup loader_path reconciliation sweep.

``loader_path`` is a pure function of (file_path, folder registry) and the
registry only changes across restarts, so the sweep recomputes every
filesystem-backed row once per scan. Responses then read the column verbatim
with no read-time fallback.
"""

import uuid
from pathlib import Path
from unittest.mock import patch

from sqlalchemy.orm import Session

from app.assets.database.models import AssetReference
from app.assets.scanner import reconcile_loader_paths


def _add_ref(session: Session, file_path: str | None, loader_path: str | None) -> str:
    ref_id = str(uuid.uuid4())
    session.add(
        AssetReference(
            id=ref_id,
            asset_id=str(uuid.uuid4()),
            name="ref",
            file_path=file_path,
            loader_path=loader_path,
        )
    )
    return ref_id


def _loader_paths(session: Session, ids: list[str]) -> list[str | None]:
    return [session.get(AssetReference, ref_id).loader_path for ref_id in ids]


def test_reconcile_covers_the_full_transition_matrix(db_engine, tmp_path: Path):
    models = tmp_path / "models"
    ckpt = models / "checkpoints"
    ckpt.mkdir(parents=True)

    with Session(db_engine) as session:
        legacy_null = _add_ref(session, str(ckpt / "sub" / "a.safetensors"), None)
        stale_value = _add_ref(session, str(ckpt / "b.safetensors"), "old/wrong.safetensors")
        bucket_gone = _add_ref(session, str(tmp_path / "gone" / "c.safetensors"), "c.safetensors")
        orphan = _add_ref(session, str(models / "junk" / "x.bin"), None)
        already_ok = _add_ref(session, str(ckpt / "d.safetensors"), "d.safetensors")
        hash_only = _add_ref(session, None, None)
        session.commit()
        ids = [legacy_null, stale_value, bucket_gone, orphan, already_ok]

    from contextlib import contextmanager

    @contextmanager
    def _session():
        with Session(db_engine) as sess:
            yield sess

    with patch("app.assets.scanner.create_session", _session), patch(
        "app.assets.services.path_utils.folder_paths"
    ) as mock_fp, patch(
        "app.assets.services.path_utils.get_comfy_models_folders",
        return_value=[("checkpoints", [str(ckpt)], {".safetensors"})],
    ):
        mock_fp.get_input_directory.return_value = str(tmp_path / "in")
        mock_fp.get_output_directory.return_value = str(tmp_path / "out")
        mock_fp.get_temp_directory.return_value = str(tmp_path / "tmp")
        mock_fp.models_dir = str(models)

        updated = reconcile_loader_paths()

    # legacy NULL healed, stale value corrected, vanished bucket nulled;
    # orphan and already-correct rows untouched.
    assert updated == 3
    with Session(db_engine) as session:
        assert _loader_paths(session, ids) == [
            "sub/a.safetensors",
            "b.safetensors",
            None,
            None,
            "d.safetensors",
        ]
        # file-less references are outside the sweep entirely
        assert session.get(AssetReference, hash_only).loader_path is None


def test_reconcile_is_idempotent(db_engine, tmp_path: Path):
    models = tmp_path / "models"
    ckpt = models / "checkpoints"
    ckpt.mkdir(parents=True)

    with Session(db_engine) as session:
        _add_ref(session, str(ckpt / "a.safetensors"), None)
        session.commit()

    from contextlib import contextmanager

    @contextmanager
    def _session():
        with Session(db_engine) as sess:
            yield sess

    with patch("app.assets.scanner.create_session", _session), patch(
        "app.assets.services.path_utils.folder_paths"
    ) as mock_fp, patch(
        "app.assets.services.path_utils.get_comfy_models_folders",
        return_value=[("checkpoints", [str(ckpt)], {".safetensors"})],
    ):
        mock_fp.get_input_directory.return_value = str(tmp_path / "in")
        mock_fp.get_output_directory.return_value = str(tmp_path / "out")
        mock_fp.get_temp_directory.return_value = str(tmp_path / "tmp")
        mock_fp.models_dir = str(models)

        assert reconcile_loader_paths() == 1
        assert reconcile_loader_paths() == 0

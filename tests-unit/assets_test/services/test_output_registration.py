import os
from pathlib import Path

import pytest
from sqlalchemy import event, select

from app.assets.database.models import Asset, AssetContent
from app.assets.services.ingest import register_executed_output, register_output_files
from app.assets.services.output_registration import (
    OutputExecution,
    OutputFileRegistration,
)


def _write_output_file(name: str, data: bytes) -> Path:
    import folder_paths

    path = Path(folder_paths.get_output_directory()) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_executed_registration_replaces_existing_live_content(mock_create_session):
    path = _write_output_file("executed-registration-replacement.png", b"original")
    try:
        original_record = register_executed_output(str(path), job_id="original-job")
        path.write_bytes(b"replacement")
        registration = OutputFileRegistration(
            path=str(path), execution=OutputExecution.EXECUTED
        )

        registered = register_output_files((registration,), job_id="replacement-job")

        with mock_create_session() as session:
            contents = list(
                session.scalars(
                    select(AssetContent).where(
                        AssetContent.path == os.path.abspath(path)
                    )
                )
            )
            original_content = session.get(AssetContent, original_record.content_id)
            replacement_content = next(
                content
                for content in contents
                if content.id != original_record.content_id
            )
            replacement_record = session.scalars(
                select(Asset).where(Asset.job_id == "replacement-job")
            ).one()
            assert registered == 1
            assert len(contents) == 2
            assert original_content.is_missing is True
            assert replacement_content.is_missing is False
            assert replacement_record.content_id == replacement_content.id
    finally:
        path.unlink(missing_ok=True)


def test_cached_registration_reuses_existing_content_without_update(
    mock_create_session, db_engine
):
    path = _write_output_file("cached-registration-replay.png", b"cached")
    update_statements: list[str] = []

    def capture_updates(_, __, statement, ___, ____, _____):
        if statement.lstrip().upper().startswith("UPDATE"):
            update_statements.append(statement)

    try:
        original_record = register_executed_output(str(path), job_id="original-job")
        event.listen(db_engine, "before_cursor_execute", capture_updates)
        registration = OutputFileRegistration(
            path=str(path), execution=OutputExecution.CACHED
        )

        registered = register_output_files((registration,), job_id="cached-job")

        with mock_create_session() as session:
            contents = list(
                session.scalars(
                    select(AssetContent).where(
                        AssetContent.path == os.path.abspath(path)
                    )
                )
            )
            records = list(
                session.scalars(
                    select(Asset).where(
                        Asset.content_id == original_record.content_id
                    )
                )
            )
            assert registered == 1
            assert len(contents) == 1
            assert {record.job_id for record in records} == {
                "original-job",
                "cached-job",
            }
            assert update_statements == []
    finally:
        event.remove(db_engine, "before_cursor_execute", capture_updates)
        path.unlink(missing_ok=True)


def test_cached_registration_without_live_content_is_nonevent(
    mock_create_session,
):
    """S10.4: a CACHED disposition with no live content creates nothing.

    Previously this fell back to a fresh executed registration; the cached
    primitive now treats missing live content as a logged non-event, so no
    content row and no delivery record are created. (The dispatch still counts
    it as handled; that wart is removed with the dispatch in a later todo.)
    """
    path = _write_output_file("cached-registration-missing.png", b"new content")
    try:
        registration = OutputFileRegistration(
            path=str(path), execution=OutputExecution.CACHED
        )

        register_output_files((registration,), job_id="fallback-job")

        with mock_create_session() as session:
            contents = list(
                session.scalars(
                    select(AssetContent).where(
                        AssetContent.path == os.path.abspath(path)
                    )
                )
            )
            records = list(
                session.scalars(
                    select(Asset).where(Asset.job_id == "fallback-job")
                )
            )
            assert contents == []
            assert records == []
    finally:
        path.unlink(missing_ok=True)


def test_invalid_execution_disposition_propagates(mock_create_session):
    path = _write_output_file("invalid-execution-disposition.png", b"invalid")
    try:
        registration = OutputFileRegistration(
            path=str(path), execution=OutputExecution.EXECUTED
        )
        object.__setattr__(registration, "execution", "invalid")

        with pytest.raises(AssertionError):
            register_output_files((registration,), job_id="invalid-job")
    finally:
        path.unlink(missing_ok=True)


def test_positional_metadata_and_job_id_compatibility(mock_create_session):
    path = _write_output_file("positional-registration.png", b"positional")
    try:
        registration = OutputFileRegistration(
            path=str(path), execution=OutputExecution.EXECUTED
        )

        registered = register_output_files(
            (registration,), {"source": "positional"}, "positional-job"
        )

        with mock_create_session() as session:
            record = session.scalars(
                select(Asset).where(Asset.job_id == "positional-job")
            ).one()
            assert registered == 1
            assert record.job_id == "positional-job"
    finally:
        path.unlink(missing_ok=True)

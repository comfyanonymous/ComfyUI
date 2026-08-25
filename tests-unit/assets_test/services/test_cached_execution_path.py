import os
from pathlib import Path

from sqlalchemy import event, select

from app.assets.database.models import Asset, AssetContent
from app.assets.services.ingest import register_output_file_b, register_output_files
from app.assets.services.output_registration import (
    OutputExecution,
    OutputFileRegistration,
)


def _write_output_file(name: str) -> Path:
    import folder_paths

    path = Path(folder_paths.get_output_directory()) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"cached output")
    return path


def test_cached_execution_binds_new_record_to_existing_content(mock_create_session):
    path = _write_output_file("cached-execution-bind.png")
    try:
        original_record = register_output_file_b(str(path), job_id="original-job")
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
            assert len(records) == 2
            assert original_record.id in {record.id for record in records}
            assert {record.job_id for record in records} == {
                "original-job",
                "cached-job",
            }
    finally:
        path.unlink(missing_ok=True)


def test_cached_execution_does_not_mutate_existing_content(
    mock_create_session, db_engine
):
    path = _write_output_file("cached-execution-no-update.png")
    update_statements: list[str] = []

    def capture_updates(_, __, statement, ___, ____, _____):
        if statement.lstrip().upper().startswith("UPDATE"):
            update_statements.append(statement)

    try:
        original_record = register_output_file_b(str(path), job_id="original-job")
        with mock_create_session() as session:
            original_content = session.get(AssetContent, original_record.content_id)
            original_state = (
                original_content.hash,
                original_content.size_bytes,
                original_content.path,
                original_content.mtime_ns,
                original_content.is_missing,
                original_content.created_at,
            )
        event.listen(db_engine, "before_cursor_execute", capture_updates)
        registration = OutputFileRegistration(
            path=str(path), execution=OutputExecution.CACHED
        )

        register_output_files((registration,), job_id="cached-job")

        with mock_create_session() as session:
            content = session.get(AssetContent, original_record.content_id)
            current_state = (
                content.hash,
                content.size_bytes,
                content.path,
                content.mtime_ns,
                content.is_missing,
                content.created_at,
            )
            assert current_state == original_state
            assert update_statements == []
    finally:
        event.remove(db_engine, "before_cursor_execute", capture_updates)
        path.unlink(missing_ok=True)

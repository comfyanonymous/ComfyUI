from pathlib import Path

from sqlalchemy import select

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, create_record
from app.assets.services.ingest import register_output_files


def test_register_output_files_uses_cached_path_for_existing_content(
    session, mock_create_session
):
    import folder_paths

    path = Path(folder_paths.get_output_directory()) / "cached-output-path-test.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(b"cached output")
        content = create_content(session, str(path))
        original_record = create_record(session, content.id, path.name, job_id="job1")
        session.commit()

        registered = register_output_files([str(path)], job_id="job2")

        session.expire_all()
        contents = list(
            session.scalars(select(AssetContent).where(AssetContent.path == str(path)))
        )
        records = list(session.scalars(select(Asset).where(Asset.content_id == content.id)))
        assert registered == 1
        assert len(contents) == 1
        assert len(records) == 2
        assert original_record.id in {record.id for record in records}
        assert {record.job_id for record in records} == {"job1", "job2"}
    finally:
        path.unlink(missing_ok=True)


def test_register_output_files_uses_fresh_path_for_new_content(
    session, mock_create_session
):
    import folder_paths

    path = Path(folder_paths.get_output_directory()) / "fresh-output-path-test.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(b"fresh output")
        registered = register_output_files([str(path)], job_id="job1")

        session.expire_all()
        contents = list(
            session.scalars(select(AssetContent).where(AssetContent.path == str(path)))
        )
        assert registered == 1
        assert len(contents) == 1
        records = list(
            session.scalars(select(Asset).where(Asset.content_id == contents[0].id))
        )
        assert len(records) == 1
        assert records[0].job_id == "job1"
    finally:
        path.unlink(missing_ok=True)

from sqlalchemy import select

from app.assets.database.models import Asset
from app.assets.database.queries.records import create_content
from app.assets.helpers import to_stored_hash
from app.assets.services.ingest import create_from_hash


def test_create_from_hash_with_prefixed_hash_finds_existing_content(
    mock_create_session, monkeypatch, temp_dir
):
    digest = "a" * 64
    path = temp_dir / "existing.bin"
    path.write_bytes(b"existing bytes")
    monkeypatch.setattr("app.assets.mode.hashing_enabled", lambda: True)

    with mock_create_session() as session:
        content = create_content(session, str(path), to_stored_hash(digest), path.stat().st_size)
        content_id = content.id
        session.commit()

    result = create_from_hash(f"blake3:{digest}", "derived.bin")

    assert result is not None
    with mock_create_session() as session:
        records = list(session.scalars(select(Asset)))
        created_record = session.get(Asset, result.ref.id)
    assert created_record is not None
    assert created_record.content_id == content_id
    assert [record.content_id for record in records] == [content_id]


def test_create_from_hash_with_bare_hash_also_works(
    mock_create_session, monkeypatch, temp_dir
):
    digest = "b" * 64
    path = temp_dir / "existing.bin"
    path.write_bytes(b"existing bytes")
    monkeypatch.setattr("app.assets.mode.hashing_enabled", lambda: True)

    with mock_create_session() as session:
        content = create_content(session, str(path), to_stored_hash(digest), path.stat().st_size)
        content_id = content.id
        session.commit()

    result = create_from_hash(digest, "derived.bin")

    assert result is not None
    with mock_create_session() as session:
        records = list(session.scalars(select(Asset)))
        created_record = session.get(Asset, result.ref.id)
    assert created_record is not None
    assert created_record.content_id == content_id
    assert [record.content_id for record in records] == [content_id]

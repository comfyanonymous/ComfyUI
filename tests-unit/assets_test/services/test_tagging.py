from sqlalchemy.orm import Session

from app.assets.database.queries import create_content, create_record, fetch_record_tags


def test_tags_are_birth_facts_of_a_record(session: Session) -> None:
    content = create_content(session, "/models/model.safetensors")
    record = create_record(session, content.id, "model", tags=["model", "checkpoint"])

    assert fetch_record_tags(session, record.id) == ["checkpoint", "model"]

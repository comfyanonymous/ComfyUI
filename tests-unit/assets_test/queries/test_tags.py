from sqlalchemy.orm import Session

from app.assets.database.queries import (
    create_content,
    create_record,
    fetch_record_tags,
    list_records_page,
)


def test_record_tags_filter_records_without_affecting_content(session: Session) -> None:
    content = create_content(session, "/models/shared.safetensors")
    tagged = create_record(session, content.id, "tagged", tags=["model"])
    untagged = create_record(session, content.id, "untagged")

    records, _ = list_records_page(session, include_tags=["model"])

    assert [record.id for record in records] == [tagged.id]
    assert fetch_record_tags(session, untagged.id) == []

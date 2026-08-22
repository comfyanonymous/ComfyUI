from sqlalchemy.orm import Session

from app.assets.database.queries import create_content, create_record, list_records_page


def test_record_keyset_cursor_pages_in_creation_order(session: Session) -> None:
    records = [
        create_record(session, create_content(session, f"/output/{name}").id, name)
        for name in ("one.png", "two.png", "three.png")
    ]
    for index, record in enumerate(records, start=1):
        record.id = f"00000000-0000-0000-0000-{index:012d}"
    session.flush()

    first_page, cursor = list_records_page(session, limit=2)
    second_page, next_cursor = list_records_page(session, cursor=cursor, limit=2)

    assert [record.id for record in first_page + second_page] == [record.id for record in records]
    assert next_cursor is None

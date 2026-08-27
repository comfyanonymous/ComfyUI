from sqlalchemy.orm import Session

from app.assets.database.models import AssetTag, Tag
from app.assets.database.queries import create_content, create_record, fetch_record_tags
from app.assets.services.tagging import remove_tags


def test_tags_are_birth_facts_of_a_record(session: Session) -> None:
    content = create_content(session, "/models/model.safetensors")
    record = create_record(session, content.id, "model", tags=["model", "checkpoint"])

    assert fetch_record_tags(session, record.id) == ["checkpoint", "model"]


def _attach_automatic_tag(session: Session, record_id: str, tag_name: str) -> None:
    if session.get(Tag, tag_name) is None:
        session.add(Tag(name=tag_name))
        session.flush()
    session.add(
        AssetTag(asset_id=record_id, tag_name=tag_name, origin="automatic")
    )


def test_removing_present_automatic_tag_reports_it_protected_not_absent(
    session: Session, mock_create_session
) -> None:
    """Given a record carrying an origin='automatic' tag, When remove_tags is
    asked to remove that tag, Then it is reported in the ``protected`` bucket —
    present but not removable — and NOT as ``not_present``. A caller must be able
    to tell 'the tag wasn't there' apart from 'the tag is there but protected'."""
    content = create_content(session, "/output/protected.png")
    record = create_record(session, content.id, "protected-fixture")
    _attach_automatic_tag(session, record.id, "auto")
    session.commit()

    result = remove_tags(record.id, ["auto"])

    assert result.protected == ["auto"]
    assert result.not_present == []
    assert result.removed == []


def test_remove_tags_separates_removed_protected_and_absent(
    session: Session, mock_create_session
) -> None:
    """A single remove_tags call across a manual (removable) tag, an automatic
    (protected) tag, and a name that was never applied lands each in its own
    bucket — removed / protected / not_present respectively."""
    content = create_content(session, "/output/mixed.png")
    record = create_record(session, content.id, "mixed-fixture", tags=["keep-me"])
    _attach_automatic_tag(session, record.id, "auto")
    session.commit()

    result = remove_tags(record.id, ["keep-me", "auto", "ghost"])

    assert result.removed == ["keep-me"]
    assert result.protected == ["auto"]
    assert result.not_present == ["ghost"]
    assert fetch_record_tags(session, record.id) == ["auto"]

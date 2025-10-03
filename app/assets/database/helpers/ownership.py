import sqlalchemy as sa

from ..models import AssetInfo


def visible_owner_clause(owner_id: str) -> sa.sql.ClauseElement:
    """Build owner visibility predicate for reads. Owner-less rows are visible to everyone."""

    owner_id = (owner_id or "").strip()
    if owner_id == "":
        return AssetInfo.owner_id == ""
    return AssetInfo.owner_id.in_(["", owner_id])

import json
from collections.abc import Callable

import pytest
from aiohttp.test_utils import make_mocked_request
from sqlalchemy.orm import Session

from app.assets import mode
from app.assets.api import routes
from app.assets.database.queries.records import (
    create_content,
    create_record,
    mark_content_missing,
)


class _JsonRequest:
    def __init__(self, payload: dict[str, str]) -> None:
        self._payload = payload

    async def json(self) -> dict[str, str]:
        return self._payload


def _session_factory(engine) -> Callable[[], Session]:
    return lambda: Session(engine)


@pytest.mark.asyncio
async def test_unfiltered_listing_contains_missing_entity(
    db_engine, session, temp_dir, monkeypatch
):
    content = create_content(session, path=str(temp_dir / "missing.png"))
    record = create_record(
        session,
        content_id=content.id,
        name="missing.png",
        mime_type="image/png",
        tags=["input"],
    )
    mark_content_missing(session, content.id)
    session.commit()

    monkeypatch.setattr(routes, "create_session", _session_factory(db_engine))

    response = await routes.list_assets_route.__wrapped__(
        make_mocked_request("GET", "/api/assets")
    )

    body = json.loads(response.body)
    asset = next(item for item in body["assets"] if item["id"] == record.id)
    assert "missing" in asset["tags"]


@pytest.mark.asyncio
async def test_exclude_tags_missing_hides_it(db_engine, session, temp_dir, monkeypatch):
    content = create_content(session, path=str(temp_dir / "missing.png"))
    record = create_record(
        session,
        content_id=content.id,
        name="missing.png",
        mime_type="image/png",
    )
    mark_content_missing(session, content.id)
    session.commit()

    monkeypatch.setattr(routes, "create_session", _session_factory(db_engine))

    response = await routes.list_assets_route.__wrapped__(
        make_mocked_request("GET", "/api/assets?exclude_tags=missing")
    )

    body = json.loads(response.body)
    assert record.id not in {item["id"] for item in body["assets"]}


@pytest.mark.asyncio
async def test_from_hash_off_mode_returns_400(monkeypatch):
    monkeypatch.setattr(mode, "hashing_enabled", lambda: False)

    response = await routes.create_asset_from_hash_route.__wrapped__(
        _JsonRequest({"hash": f"blake3:{'a' * 64}"})
    )

    assert response.status == 400
    assert json.loads(response.body)["error"]["code"] == "FEATURE_DISABLED"

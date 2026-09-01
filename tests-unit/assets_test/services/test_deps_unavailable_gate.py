import json

import pytest

from app.assets import lifecycle, mode
from app.assets.api import routes

_VALID_HASH = "blake3:" + "a" * 64


class _JsonRequest:
    def __init__(self, payload: dict) -> None:
        self.match_info: dict[str, str] = {}
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


@pytest.fixture
def registered_but_uninitialised(monkeypatch):
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)
    monkeypatch.setattr(mode, "_args", None)


@pytest.mark.asyncio
async def test_uninitialised_mode_serves_the_disabled_envelope_not_a_crash(
    registered_but_uninitialised,
):
    routes.disable_assets_routes()

    response = await routes.create_asset_from_hash_route(
        _JsonRequest({"hash": _VALID_HASH, "name": "x.bin"})
    )

    assert response.status == 503
    body = json.loads(response.body)
    assert body["error"]["code"] == "SERVICE_DISABLED", (
        "with its database dependencies missing the asset system is disabled, which is a "
        "known state the envelope already describes — not an unexpected server error"
    )


@pytest.mark.asyncio
async def test_route_crashes_while_enabled_with_mode_uninitialised(
    registered_but_uninitialised,
):
    with pytest.raises(RuntimeError, match="was not called before hashing_enabled"):
        await routes.create_asset_from_hash_route(
            _JsonRequest({"hash": _VALID_HASH, "name": "x.bin"})
        )


def test_missing_dependencies_disable_the_asset_routes(monkeypatch):
    monkeypatch.setattr(lifecycle, "dependencies_available", lambda: False)
    calls: list[int] = []
    monkeypatch.setattr(routes, "disable_assets_routes", lambda: calls.append(1))

    ready = lifecycle.assets_dependencies_ready()

    assert ready is False
    assert calls == [1], (
        "the deps-unavailable branch must actually disable the routes, exactly once"
    )


def test_available_dependencies_leave_the_routes_alone(monkeypatch):
    monkeypatch.setattr(lifecycle, "dependencies_available", lambda: True)
    calls: list[int] = []
    monkeypatch.setattr(routes, "disable_assets_routes", lambda: calls.append(1))

    ready = lifecycle.assets_dependencies_ready()

    assert ready is True
    assert calls == []

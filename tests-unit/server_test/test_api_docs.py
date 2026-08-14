"""Tests for the OpenAPI spec and API docs routes."""

import os

import pytest
import pytest_asyncio
import yaml
from aiohttp import web

from app.api_docs import SPEC_PATH, SPEC_URL, add_api_docs_routes

pytestmark = pytest.mark.asyncio


def _build_app():
    """Mirror how PromptServer mounts these routes, including the /api prefix.

    add_routes() walks the route table and re-registers every RouteDef under an
    /api prefix, so both the bare and prefixed forms are served. Reproducing
    that here keeps the prefix behaviour covered by tests.
    """
    app = web.Application()
    routes = web.RouteTableDef()
    add_api_docs_routes(routes)

    api_routes = web.RouteTableDef()
    for route in routes:
        if isinstance(route, web.RouteDef):
            api_routes.route(route.method, "/api" + route.path)(
                route.handler, **route.kwargs
            )
    app.add_routes(api_routes)
    app.add_routes(routes)
    return app


@pytest_asyncio.fixture
async def client(aiohttp_client):
    return await aiohttp_client(_build_app())


async def test_spec_path_points_at_the_repo_spec():
    """SPEC_PATH must resolve from __file__, not the cwd."""
    assert os.path.isfile(SPEC_PATH)
    assert os.path.basename(SPEC_PATH) == "openapi.yaml"


async def test_get_spec_returns_yaml(client):
    resp = await client.get("/openapi.yaml")
    assert resp.status == 200
    assert resp.headers["Content-Type"] == "application/yaml"
    assert resp.headers["Cache-Control"] == "no-store, must-revalidate"


async def test_spec_body_is_valid_openapi_3(client):
    resp = await client.get("/openapi.yaml")
    spec = yaml.safe_load(await resp.text())
    assert spec["openapi"].startswith("3.")
    assert spec["paths"]


async def test_spec_is_also_served_under_the_api_prefix(client):
    """Documents the prefix duplication so a refactor cannot silently break it."""
    resp = await client.get("/api/openapi.yaml")
    assert resp.status == 200
    assert resp.headers["Content-Type"] == "application/yaml"


async def test_missing_spec_returns_404(client, monkeypatch):
    monkeypatch.setattr("app.api_docs.SPEC_PATH", "/nonexistent/openapi.yaml")
    resp = await client.get("/openapi.yaml")
    assert resp.status == 404


async def test_docs_page_returns_html_referencing_the_spec(client):
    resp = await client.get("/api-docs")
    assert resp.status == 200
    assert resp.content_type == "text/html"
    body = await resp.text()
    assert SPEC_URL in body


async def test_docs_page_spec_url_is_relative(client):
    """A relative URL resolves correctly from both /api-docs and /api/api-docs."""
    assert not SPEC_URL.startswith("/")
    resp = await client.get("/api/api-docs")
    assert resp.status == 200


async def test_docs_page_cannot_execute_requests(client):
    """The local server is unauthenticated, so the docs UI must not fire requests.

    Redoc has no request execution at all. Guard against a swap to Swagger UI,
    whose "Try it out" would give one-click access to destructive endpoints.
    """
    body = (await (await client.get("/api-docs")).text()).lower()
    assert "<redoc" in body
    assert "swagger" not in body


async def test_docs_page_degrades_when_the_cdn_is_unreachable(client):
    """Offline installs must get the fallback notice, not a blank page.

    The viewer bundle is the only part that needs network, so the page carries
    an onerror hook that reveals a notice pointing at the locally served spec.
    """
    body = await (await client.get("/api-docs")).text()
    assert 'onerror=' in body
    assert "getElementById('fallback')" in body
    assert 'id="fallback"' in body
    # The notice has to link the spec, which is served locally.
    assert f'<a href="{SPEC_URL}">' in body


async def test_routes_survive_the_static_catch_all(aiohttp_client, tmp_path):
    """web.static('/') is registered last and matches everything.

    Registering on PromptServer's route table is what keeps these paths
    reachable; this fails if they are ever moved after the catch-all.
    """
    (tmp_path / "index.html").write_text("frontend")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "node.json").write_text("{}")

    app = _build_app()
    app.add_routes([web.static("/docs", tmp_path / "docs")])
    app.add_routes([web.static("/", tmp_path)])
    client = await aiohttp_client(app)

    assert (await client.get("/openapi.yaml")).headers["Content-Type"] == (
        "application/yaml"
    )
    assert (await client.get("/api-docs")).content_type == "text/html"
    # Embedded node docs and the frontend bundle are untouched.
    assert await (await client.get("/docs/node.json")).text() == "{}"
    assert await (await client.get("/index.html")).text() == "frontend"


async def test_api_docs_flag_is_off_by_default():
    """Both routes are gated on this flag in PromptServer.add_routes()."""
    from comfy.cli_args import parser

    assert parser.parse_args([]).enable_api_docs is False
    assert parser.parse_args(["--enable-api-docs"]).enable_api_docs is True

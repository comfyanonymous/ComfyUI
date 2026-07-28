"""aiohttp routes for the download manager.

Endpoint surface (all under ``/api/download``), mirroring the response
envelope used by ``app/assets/api/routes.py``:

  POST   /api/download/enqueue
  GET    /api/download
  POST   /api/download/availability
  POST   /api/download/clear
  GET    /api/download/auth
  POST   /api/download/auth/{provider}/login
  POST   /api/download/auth/{provider}/logout
  GET    /api/download/{id}
  DELETE /api/download/{id}
  POST   /api/download/{id}/pause
  POST   /api/download/{id}/resume
  POST   /api/download/{id}/cancel
  POST   /api/download/{id}/priority

Note on ordering: the static ``auth`` routes are registered before the dynamic
``/api/download/{id}`` route so a request to ``.../auth`` is not captured as
``id == "auth"``.
"""

from __future__ import annotations

import json

from aiohttp import web
from pydantic import BaseModel, ValidationError

from app.model_downloader.api import schemas_in, schemas_out
from app.model_downloader.auth.oauth import LoginInProgress, OAuthNotConfigured
from app.model_downloader.auth.providers import PROVIDERS
from app.model_downloader.auth.store import AUTH_STORE
from app.model_downloader.manager import DOWNLOAD_MANAGER, DownloadError

ROUTES = web.RouteTableDef()


def register_routes(app: web.Application) -> None:
    """Wire the download-manager routes into the running aiohttp app."""
    app.add_routes(ROUTES)


# ----- envelope helpers (same shape as app/assets/api/routes.py) -----


def _error(status: int, code: str, message: str, details: dict | None = None) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message, "details": details or {}}},
        status=status,
    )


def _ok(payload, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status)


async def _parse(request: web.Request, model: type[BaseModel]):
    try:
        raw = await request.json()
    except json.JSONDecodeError:
        return _error(400, "INVALID_JSON", "Request body must be valid JSON.")
    try:
        return model.model_validate(raw)
    except ValidationError as ve:
        return _error(400, "INVALID_BODY", "Validation failed.", {"errors": json.loads(ve.json())})


def _from_download_error(e: DownloadError) -> web.Response:
    return _error(e.http_status, e.code, e.message)


# ----- downloads: collection + enqueue + availability -----


@ROUTES.post("/api/download/enqueue")
async def enqueue(request: web.Request) -> web.Response:
    parsed = await _parse(request, schemas_in.EnqueueRequest)
    if isinstance(parsed, web.Response):
        return parsed
    try:
        download_id = await DOWNLOAD_MANAGER.enqueue(
            parsed.url,
            parsed.model_id,
            priority=parsed.priority,
            expected_sha256=parsed.expected_sha256,
            allow_any_extension=parsed.allow_any_extension,
        )
    except DownloadError as e:
        return _from_download_error(e)
    return _ok({"download_id": download_id, "accepted": True}, status=202)


@ROUTES.get("/api/download")
async def list_downloads(request: web.Request) -> web.Response:
    return _ok({"downloads": await DOWNLOAD_MANAGER.list()})


@ROUTES.post("/api/download/availability")
async def availability(request: web.Request) -> web.Response:
    parsed = await _parse(request, schemas_in.AvailabilityRequest)
    if isinstance(parsed, web.Response):
        return parsed
    return _ok({"models": await DOWNLOAD_MANAGER.availability(parsed.models)})


@ROUTES.post("/api/download/clear")
async def clear(request: web.Request) -> web.Response:
    deleted = await DOWNLOAD_MANAGER.clear()
    return _ok({"deleted": deleted})


# ----- auth (OAuth login + env-key status) — must precede /{id} -----


@ROUTES.get("/api/download/auth")
async def auth_status(request: web.Request) -> web.Response:
    return _ok({"providers": schemas_out.auth_status()})


@ROUTES.post("/api/download/auth/{provider}/login")
async def auth_login(request: web.Request) -> web.Response:
    provider = PROVIDERS.get(request.match_info["provider"])
    if provider is None:
        return _error(400, "UNKNOWN_PROVIDER", "No such auth provider.")
    try:
        authorize_url = await AUTH_STORE.begin_login(provider)
    except OAuthNotConfigured as e:
        return _error(400, "OAUTH_NOT_CONFIGURED", str(e))
    except LoginInProgress as e:
        return _error(409, "LOGIN_IN_PROGRESS", str(e))
    return _ok({"authorize_url": authorize_url})


@ROUTES.post("/api/download/auth/{provider}/logout")
async def auth_logout(request: web.Request) -> web.Response:
    provider = PROVIDERS.get(request.match_info["provider"])
    if provider is None:
        return _error(400, "UNKNOWN_PROVIDER", "No such auth provider.")
    AUTH_STORE.clear(provider.name)
    return _ok({"logged_out": True})


# ----- single download by id (dynamic; registered last) -----


@ROUTES.get("/api/download/{id}")
async def get_download(request: web.Request) -> web.Response:
    view = await DOWNLOAD_MANAGER.status(request.match_info["id"])
    if view is None:
        return _error(404, "NOT_FOUND", "No such download.")
    return _ok(view)


@ROUTES.delete("/api/download/{id}")
async def delete_download(request: web.Request) -> web.Response:
    try:
        await DOWNLOAD_MANAGER.delete(request.match_info["id"])
    except DownloadError as e:
        return _from_download_error(e)
    return _ok({"deleted": True})


@ROUTES.post("/api/download/{id}/pause")
async def pause(request: web.Request) -> web.Response:
    try:
        await DOWNLOAD_MANAGER.pause(request.match_info["id"])
    except DownloadError as e:
        return _from_download_error(e)
    return _ok({"ok": True})


@ROUTES.post("/api/download/{id}/resume")
async def resume(request: web.Request) -> web.Response:
    try:
        await DOWNLOAD_MANAGER.resume(request.match_info["id"])
    except DownloadError as e:
        return _from_download_error(e)
    return _ok({"ok": True})


@ROUTES.post("/api/download/{id}/cancel")
async def cancel(request: web.Request) -> web.Response:
    try:
        await DOWNLOAD_MANAGER.cancel(request.match_info["id"])
    except DownloadError as e:
        return _from_download_error(e)
    return _ok({"ok": True})


@ROUTES.post("/api/download/{id}/priority")
async def set_priority(request: web.Request) -> web.Response:
    parsed = await _parse(request, schemas_in.PriorityRequest)
    if isinstance(parsed, web.Response):
        return parsed
    try:
        await DOWNLOAD_MANAGER.set_priority(request.match_info["id"], parsed.priority)
    except DownloadError as e:
        return _from_download_error(e)
    return _ok({"ok": True})

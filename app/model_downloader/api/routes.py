"""Aiohttp routes for the server-side model download subsystem.

Endpoint surface (all under ``/api/``, all kebab-case):

  - ``POST /api/models-availability-status``     — bulk status + metadata query.
  - ``POST /api/download-models``                — start a batch of downloads.
  - ``POST /api/cancel-model-download-session``  — cancel a single in-flight one.
  - ``GET  /api/hf-auth-token-status``           — current HF login state.
  - ``POST /api/hf-auth-login-start``            — begin the HF OAuth flow.
  - ``POST /api/hf-auth-logout``                 — drop the stored HF token.

The contract is intentionally narrow: only model_ids of the form
``<directory>/<filename>`` (validated via ``app.model_downloader.paths``)
are accepted, and only URLs on the same allowlist the frontend already
uses (HuggingFace, Civitai, localhost) can be fetched. Both are required
to keep the server out of the SSRF business for this feature.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal, Optional

from aiohttp import web
from pydantic import BaseModel, ValidationError

from app.model_downloader.allowlist import is_url_allowed
from app.model_downloader.download_server import (
    DOWNLOAD_SERVER,
    DownloadSession,
)
from app.model_downloader.downloader import schedule_batch
from app.model_downloader.gated_detection import probe_url
from app.model_downloader.hf_auth.auth_store import HF_AUTH_STORE
from app.model_downloader.hf_auth.eligibility import is_hf_auth_eligible
from app.model_downloader.hf_auth.oauth import (
    OAuthCallbackError,
    OAuthInProgressError,
    start_login_flow,
)
from app.model_downloader.paths import (
    InvalidModelId,
    parse_model_id,
    resolve_existing,
)
from app.model_downloader.api import schemas_in, schemas_out

ROUTES = web.RouteTableDef()


def register_routes(app: web.Application) -> None:
    """Wire the model-downloader routes into the running aiohttp app.

    Called once from ``server.py`` during ``PromptServer`` startup.
    """
    app.add_routes(ROUTES)


# ----- response helpers (same envelope as app/assets/api/routes.py) -----


ErrorCode = Literal[
    "INVALID_JSON",
    "INVALID_BODY",
    "EMPTY_REQUEST",
    "INVALID_MODEL_ID",
    "URL_NOT_ALLOWED",
    "ALREADY_AVAILABLE",
    "ALREADY_DOWNLOADING",
    "MODEL_NOT_DOWNLOADABLE",
    "NOT_DOWNLOADING",
    "HF_AUTH_NOT_ELIGIBLE",
    "HF_AUTH_IN_PROGRESS",
    "HF_AUTH_START_FAILED",
]


def _error(status: int, code: ErrorCode, message: str, details: dict | None = None) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message, "details": details or {}}},
        status=status,
    )


def _validation_error(code: ErrorCode, ve: ValidationError) -> web.Response:
    return _error(400, code, "Validation failed.", {"errors": json.loads(ve.json())})


def _ok(payload: BaseModel, status: int = 200) -> web.Response:
    return web.json_response(
        payload.model_dump(mode="json", exclude_none=False),
        status=status,
    )


async def _parse_body(request: web.Request, model: type[BaseModel]) -> Any:
    """Parse a JSON body into a pydantic model or raise a 400 response."""
    try:
        raw = await request.json()
    except json.JSONDecodeError:
        return _error(400, "INVALID_JSON", "Request body must be valid JSON.")
    try:
        return model.model_validate(raw)
    except ValidationError as ve:
        return _validation_error("INVALID_BODY", ve)


# ----- 1. availability status (unified: state + metadata per id) -----


@ROUTES.post("/api/models-availability-status")
async def models_availability_status(request: web.Request) -> web.Response:
    """Return per-id ``{state, progress, file_size, is_hf_downloadable}``.

    State (``available`` / ``missing`` / ``downloading``) is cheap to
    recompute per call. ``file_size`` and ``is_gated`` are cached
    server-side per URL. ``is_hf_downloadable`` is recomputed every
    call from the current token state — that's what makes login + license
    acceptance show up in the UI within one poll cycle without any
    frontend cache plumbing.
    """
    parsed = await _parse_body(request, schemas_in.AvailabilityStatusRequest)
    if isinstance(parsed, web.Response):
        return parsed

    items = list(parsed.models.items())

    # Run all probes concurrently; each is internally cached per URL.
    probes = await asyncio.gather(*(probe_url(url) for _, url in items))

    response_models: dict[str, schemas_out.ModelStatusEntry] = {}
    for (model_id, _url), probe in zip(items, probes):
        try:
            parse_model_id(model_id)
        except InvalidModelId:
            # Ill-formed identifier: report as missing without 400-ing the
            # whole batch — the workflow author probably typo'd.
            response_models[model_id] = schemas_out.ModelStatusEntry(
                state="missing",
                file_size=probe.file_size,
                is_hf_downloadable=probe.is_hf_downloadable,
            )
            continue

        active = DOWNLOAD_SERVER.get(model_id)
        if active is not None:
            response_models[model_id] = schemas_out.ModelStatusEntry(
                state="downloading",
                progress=schemas_out.DownloadProgress(
                    bytes_downloaded=active.bytes_downloaded,
                    total_bytes=active.total_bytes,
                    progress=active.progress,
                ),
                file_size=probe.file_size,
                is_hf_downloadable=probe.is_hf_downloadable,
            )
            continue

        state: schemas_out.ModelState = (
            "available" if resolve_existing(model_id) is not None else "missing"
        )
        response_models[model_id] = schemas_out.ModelStatusEntry(
            state=state,
            file_size=probe.file_size,
            is_hf_downloadable=probe.is_hf_downloadable,
        )

    return _ok(schemas_out.AvailabilityStatusResponse(
        models=response_models,
        hf_auth=schemas_out.HfAuthStatus(
            token_available=HF_AUTH_STORE.has_token(),
            eligible=is_hf_auth_eligible(),
        ),
    ))


# ----- 2. start downloads -----


@ROUTES.post("/api/download-models")
async def download_models(request: web.Request) -> web.Response:
    parsed = await _parse_body(request, schemas_in.DownloadModelsRequest)
    if isinstance(parsed, web.Response):
        return parsed

    if not parsed.models:
        return _error(400, "EMPTY_REQUEST", "No models supplied.")

    # ----- precondition pass: validate everything BEFORE registering anything -----
    # Atomic semantics: if any model fails any precondition (invalid id,
    # not allow-listed URL, already on disk, already downloading, or gated),
    # the entire request fails and no state is changed.
    requested = list(parsed.models.items())

    for model_id, url in requested:
        try:
            parse_model_id(model_id)
        except InvalidModelId as e:
            return _error(400, "INVALID_MODEL_ID", str(e),
                          {"model_id": model_id})

        if not is_url_allowed(url):
            return _error(
                400, "URL_NOT_ALLOWED",
                "Server-side downloads only accept HuggingFace, Civitai, "
                "or localhost URLs ending in a known model extension.",
                {"model_id": model_id, "url": url},
            )

        if resolve_existing(model_id) is not None:
            return _error(409, "ALREADY_AVAILABLE",
                          f"Model already exists on disk: {model_id}",
                          {"model_id": model_id})

        if DOWNLOAD_SERVER.is_downloading(model_id):
            return _error(409, "ALREADY_DOWNLOADING",
                          f"A download for {model_id} is already in progress.",
                          {"model_id": model_id})

    # Reachability check last — it's the only one that talks to the
    # network. Concurrent probes. For HF URLs ``is_hf_downloadable``
    # reflects current token access; for non-HF URLs it's None, and we
    # treat that as "no info, proceed".
    probes = await asyncio.gather(*(probe_url(url) for _, url in requested))
    for (model_id, url), probe in zip(requested, probes):
        if probe.is_hf_downloadable is False:
            return _error(
                400, "MODEL_NOT_DOWNLOADABLE",
                f"Model {model_id} is gated on HuggingFace and the current "
                f"server token (if any) does not grant access.",
                {"model_id": model_id, "url": url},
            )

    # ----- registration pass: try_register is atomic per model_id -----
    # Defensive: another request might have raced past our pre-check
    # between the loop above and here. try_register handles that.
    sessions: list[DownloadSession] = []
    for model_id, url in requested:
        session = DOWNLOAD_SERVER.try_register(model_id, url)
        if session is None:
            # Race: someone else got in. Roll back what we registered.
            for s in sessions:
                DOWNLOAD_SERVER.cancel(s.model_id)
            return _error(409, "ALREADY_DOWNLOADING",
                          f"A download for {model_id} is already in progress (race).",
                          {"model_id": model_id})
        sessions.append(session)

    DOWNLOAD_SERVER.sweep_orphan_tmp_files()
    schedule_batch(sessions)
    logging.info(
        "[model_downloader] scheduled %d downloads: %s",
        len(sessions), [s.model_id for s in sessions],
    )

    return _ok(schemas_out.DownloadModelsResponse(
        accepted=True,
        scheduled=[s.model_id for s in sessions],
    ), status=202)


# ----- 3. cancel a session -----


@ROUTES.post("/api/cancel-model-download-session")
async def cancel_model_download_session(request: web.Request) -> web.Response:
    parsed = await _parse_body(request, schemas_in.CancelDownloadSessionRequest)
    if isinstance(parsed, web.Response):
        return parsed

    try:
        parse_model_id(parsed.model_id)
    except InvalidModelId as e:
        return _error(400, "INVALID_MODEL_ID", str(e), {"model_id": parsed.model_id})

    cancelled = DOWNLOAD_SERVER.cancel(parsed.model_id)
    if not cancelled:
        return _error(404, "NOT_DOWNLOADING",
                      f"No active download for {parsed.model_id}.",
                      {"model_id": parsed.model_id})

    return _ok(schemas_out.CancelDownloadSessionResponse(cancelled=True))


# ----- 4. HuggingFace OAuth status / login start / logout -----


@ROUTES.get("/api/hf-auth-token-status")
async def hf_auth_token_status(request: web.Request) -> web.Response:
    """Return whether the server holds a usable HF token + its username.

    Used by the settings UI and (out-of-band) by the frontend on
    login completion. ``token_available`` is true even if the cached
    access_token is expired — as long as a refresh_token exists, the
    user is "logged in" from their perspective.
    """
    token_present = HF_AUTH_STORE.has_token()
    username: Optional[str] = None
    if token_present:
        # Resolve the username via whoami. Done in a worker thread because
        # huggingface_hub's whoami is synchronous + blocks on a network call.
        tok = await HF_AUTH_STORE.get_valid_token()
        if tok is not None:
            try:
                username = await asyncio.to_thread(_whoami_username, tok.access_token)
            except Exception as e:
                logging.debug("[hf_auth] whoami failed: %s", e)
    return _ok(schemas_out.HfAuthTokenStatusResponse(
        token_available=token_present,
        username=username,
    ))


def _whoami_username(token: str) -> Optional[str]:
    """Sync helper: ask HF for the user name attached to a token."""
    from huggingface_hub import HfApi
    info = HfApi().whoami(token=token)
    if isinstance(info, dict):
        return info.get("name") or info.get("fullname")
    return None


@ROUTES.post("/api/hf-auth-login-start")
async def hf_auth_login_start(request: web.Request) -> web.Response:
    """Begin one OAuth attempt: bind the callback port, return the URL.

    Rejected outright if this deployment isn't eligible (we don't
    surface the option on multi-tenant / public-IP installs).
    """
    if not is_hf_auth_eligible():
        return _error(
            403, "HF_AUTH_NOT_ELIGIBLE",
            "This server is not eligible for interactive HuggingFace login. "
            "It must be bound to a loopback address and not running in "
            "--multi-user mode.",
        )
    try:
        url = await start_login_flow()
    except OAuthInProgressError:
        return _error(
            409, "HF_AUTH_IN_PROGRESS",
            "Another HuggingFace login attempt is in progress. Try again "
            "after it completes or times out.",
        )
    except OAuthCallbackError as e:
        return _error(
            503, "HF_AUTH_START_FAILED",
            f"Could not start the HuggingFace login flow: {e}",
        )
    return _ok(schemas_out.HfAuthLoginStartResponse(authorize_url=url))


@ROUTES.post("/api/hf-auth-logout")
async def hf_auth_logout(request: web.Request) -> web.Response:
    """Drop the in-memory + on-disk HF token."""
    HF_AUTH_STORE.clear()
    return _ok(schemas_out.HfAuthLogoutResponse(logged_out=True))

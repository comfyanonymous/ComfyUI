import logging
import uuid
from aiohttp import web

from pydantic import ValidationError

import app.assets.manager as manager
from app import user_manager
from app.assets.api import schemas_in
from app.assets.helpers import get_query_dict

ROUTES = web.RouteTableDef()
USER_MANAGER: user_manager.UserManager | None = None

# UUID regex (canonical hyphenated form, case-insensitive)
UUID_RE = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

def register_assets_system(app: web.Application, user_manager_instance: user_manager.UserManager) -> None:
    global USER_MANAGER
    USER_MANAGER = user_manager_instance
    app.add_routes(ROUTES)

def _error_response(status: int, code: str, message: str, details: dict | None = None) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message, "details": details or {}}}, status=status)


def _validation_error_response(code: str, ve: ValidationError) -> web.Response:
    return _error_response(400, code, "Validation failed.", {"errors": ve.json()})


@ROUTES.head("/api/assets/hash/{hash}")
async def head_asset_by_hash(request: web.Request) -> web.Response:
    """
    HEAD request to check if an asset with hash exists; does not care about type of asset.
    """
    # get hash value from url path
    hash_str = request.match_info.get("hash", "").strip().lower()
    if not hash_str or ":" not in hash_str:
        return _error_response(400, "INVALID_HASH", "hash must be in format 'blake3:<hex>'")
    algo, digest = hash_str.split(":", 1)
    # ensure hash type + uuid is valid
    if algo != "blake3" or not digest or any(c for c in digest if c not in "0123456789abcdef"):
        return _error_response(400, "INVALID_HASH", "hash must be in format 'blake3:<hex>'")

    exists = manager.asset_exists(asset_hash=hash_str)
    return web.Response(status=200 if exists else 404)


@ROUTES.get("/api/assets")
async def list_assets(request: web.Request) -> web.Response:
    """
    GET request to list assets.
    """
    query_dict = get_query_dict(request)
    try:
        q = schemas_in.ListAssetsQuery.model_validate(query_dict)
    except ValidationError as ve:
        return _validation_error_response("INVALID_QUERY", ve)

    payload = manager.list_assets(
        include_tags=q.include_tags,
        exclude_tags=q.exclude_tags,
        name_contains=q.name_contains,
        metadata_filter=q.metadata_filter,
        limit=q.limit,
        offset=q.offset,
        sort=q.sort,
        order=q.order,
        owner_id=USER_MANAGER.get_request_user_id(request),
    )
    return web.json_response(payload.model_dump(mode="json"))


@ROUTES.get(f"/api/assets/{{id:{UUID_RE}}}")
async def get_asset(request: web.Request) -> web.Response:
    """
    GET request to get an asset's info as JSON.
    """
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
    try:
        result = manager.get_asset(
            asset_info_id=asset_info_id,
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
    except ValueError as e:
        return _error_response(404, "ASSET_NOT_FOUND", str(e), {"id": asset_info_id})
    except Exception:
        logging.exception(
            "get_asset failed for asset_info_id=%s, owner_id=%s",
            asset_info_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.put(f"/api/assets/{{id:{UUID_RE}}}/preview")
async def set_asset_preview(request: web.Request) -> web.Response:
    """
    PUT request to set or clear an asset's preview image.
    """
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
    try:
        body = schemas_in.SetPreviewBody.model_validate(await request.json())
    except ValidationError as e:
        return _validation_error_response("INVALID_BODY", e)
    except Exception:
        return _error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    try:
        result = manager.set_asset_preview(
            asset_info_id=asset_info_id,
            preview_asset_id=body.preview_id,
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
    except (PermissionError, ValueError) as e:
        return _error_response(404, "ASSET_NOT_FOUND", str(e), {"id": asset_info_id})
    except Exception:
        logging.exception(
            "set_asset_preview failed for asset_info_id=%s, owner_id=%s",
            asset_info_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.get("/api/tags")
async def get_tags(request: web.Request) -> web.Response:
    """
    GET request to list all tags based on query parameters.
    """
    query_map = dict(request.rel_url.query)

    try:
        query = schemas_in.TagsListQuery.model_validate(query_map)
    except ValidationError as e:
        return web.json_response(
            {"error": {"code": "INVALID_QUERY", "message": "Invalid query parameters", "details": e.errors()}},
            status=400,
        )

    result = manager.list_tags(
        prefix=query.prefix,
        limit=query.limit,
        offset=query.offset,
        order=query.order,
        include_zero=query.include_zero,
        owner_id=USER_MANAGER.get_request_user_id(request),
    )
    return web.json_response(result.model_dump(mode="json"))

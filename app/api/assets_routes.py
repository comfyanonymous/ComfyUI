import json
from typing import Sequence
from aiohttp import web

from app import assets_manager


ROUTES = web.RouteTableDef()


@ROUTES.get("/api/assets")
async def list_assets(request: web.Request) -> web.Response:
    q = request.rel_url.query

    include_tags: Sequence[str] = _parse_csv_tags(q.get("include_tags"))
    exclude_tags: Sequence[str] = _parse_csv_tags(q.get("exclude_tags"))
    name_contains = q.get("name_contains")

    # Optional JSON metadata filter (top-level key equality only for now)
    metadata_filter = None
    raw_meta = q.get("metadata_filter")
    if raw_meta:
        try:
            metadata_filter = json.loads(raw_meta)
            if not isinstance(metadata_filter, dict):
                metadata_filter = None
        except Exception:
            # Silently ignore malformed JSON for first iteration; could 400 in future
            metadata_filter = None

    limit = _parse_int(q.get("limit"), default=20, lo=1, hi=100)
    offset = _parse_int(q.get("offset"), default=0, lo=0, hi=10_000_000)
    sort = q.get("sort", "created_at")
    order = q.get("order", "desc")

    payload = await assets_manager.list_assets(
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        name_contains=name_contains,
        metadata_filter=metadata_filter,
        limit=limit,
        offset=offset,
        sort=sort,
        order=order,
    )
    return web.json_response(payload)


@ROUTES.put("/api/assets/{id}")
async def update_asset(request: web.Request) -> web.Response:
    asset_info_id_raw = request.match_info.get("id")
    try:
        asset_info_id = int(asset_info_id_raw)
    except Exception:
        return _error_response(400, "INVALID_ID", f"AssetInfo id '{asset_info_id_raw}' is not a valid integer.")

    try:
        payload = await request.json()
    except Exception:
        return _error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    name = payload.get("name", None)
    tags = payload.get("tags", None)
    user_metadata = payload.get("user_metadata", None)

    if name is None and tags is None and user_metadata is None:
        return _error_response(400, "NO_FIELDS", "Provide at least one of: name, tags, user_metadata.")

    if tags is not None and (not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)):
        return _error_response(400, "INVALID_TAGS", "Field 'tags' must be an array of strings.")

    if user_metadata is not None and not isinstance(user_metadata, dict):
        return _error_response(400, "INVALID_METADATA", "Field 'user_metadata' must be an object.")

    try:
        result = await assets_manager.update_asset(
            asset_info_id=asset_info_id,
            name=name,
            tags=tags,
            user_metadata=user_metadata,
        )
    except ValueError as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        return _error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(result, status=200)


def register_assets_routes(app: web.Application) -> None:
    app.add_routes(ROUTES)


def _parse_csv_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_int(qval: str | None, default: int, lo: int, hi: int) -> int:
    if not qval:
        return default
    try:
        v = int(qval)
    except Exception:
        return default
    return max(lo, min(hi, v))


def _error_response(status: int, code: str, message: str, details: dict | None = None) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message, "details": details or {}}}, status=status)

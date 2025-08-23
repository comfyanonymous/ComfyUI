from typing import Optional

from aiohttp import web
from pydantic import ValidationError

from .. import assets_manager
from .schemas_in import ListAssetsQuery, UpdateAssetBody


ROUTES = web.RouteTableDef()


@ROUTES.get("/api/assets")
async def list_assets(request: web.Request) -> web.Response:
    query_dict = dict(request.rel_url.query)

    try:
        q = ListAssetsQuery.model_validate(query_dict)
    except ValidationError as ve:
        return _validation_error_response("INVALID_QUERY", ve)

    payload = await assets_manager.list_assets(
        include_tags=q.include_tags,
        exclude_tags=q.exclude_tags,
        name_contains=q.name_contains,
        metadata_filter=q.metadata_filter,
        limit=q.limit,
        offset=q.offset,
        sort=q.sort,
        order=q.order,
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
        body = UpdateAssetBody.model_validate(await request.json())
    except ValidationError as ve:
        return _validation_error_response("INVALID_BODY", ve)
    except Exception:
        return _error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    try:
        result = await assets_manager.update_asset(
            asset_info_id=asset_info_id,
            name=body.name,
            tags=body.tags,
            user_metadata=body.user_metadata,
        )
    except ValueError as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        return _error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(result, status=200)


def register_assets_routes(app: web.Application) -> None:
    app.add_routes(ROUTES)


def _error_response(status: int, code: str, message: str, details: Optional[dict] = None) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message, "details": details or {}}}, status=status)


def _validation_error_response(code: str, ve: ValidationError) -> web.Response:
    return _error_response(400, code, "Validation failed.", {"errors": ve.errors()})

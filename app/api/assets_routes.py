import os
import uuid
import urllib.parse
from typing import Optional

from aiohttp import web
from pydantic import ValidationError

import folder_paths

from .. import assets_manager, assets_scanner
from . import schemas_in, schemas_out


ROUTES = web.RouteTableDef()


@ROUTES.head("/api/assets/hash/{hash}")
async def head_asset_by_hash(request: web.Request) -> web.Response:
    hash_str = request.match_info.get("hash", "").strip().lower()
    if not hash_str or ":" not in hash_str:
        return _error_response(400, "INVALID_HASH", "hash must be like 'blake3:<hex>'")
    algo, digest = hash_str.split(":", 1)
    if algo != "blake3" or not digest or any(c for c in digest if c not in "0123456789abcdef"):
        return _error_response(400, "INVALID_HASH", "hash must be like 'blake3:<hex>'")
    exists = await assets_manager.asset_exists(asset_hash=hash_str)
    return web.Response(status=200 if exists else 404)


@ROUTES.get("/api/assets")
async def list_assets(request: web.Request) -> web.Response:
    query_dict = dict(request.rel_url.query)

    try:
        q = schemas_in.ListAssetsQuery.model_validate(query_dict)
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
    return web.json_response(payload.model_dump(mode="json"))


@ROUTES.get("/api/assets/{id:\\d+}/content")
async def download_asset_content(request: web.Request) -> web.Response:
    asset_info_id_raw = request.match_info.get("id")
    try:
        asset_info_id = int(asset_info_id_raw)
    except Exception:
        return _error_response(400, "INVALID_ID", f"AssetInfo id '{asset_info_id_raw}' is not a valid integer.")

    disposition = request.query.get("disposition", "attachment").lower().strip()
    if disposition not in {"inline", "attachment"}:
        disposition = "attachment"

    try:
        abs_path, content_type, filename = await assets_manager.resolve_asset_content_for_download(
            asset_info_id=asset_info_id
        )
    except ValueError as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve))
    except NotImplementedError as nie:
        return _error_response(501, "BACKEND_UNSUPPORTED", str(nie))
    except FileNotFoundError:
        return _error_response(404, "FILE_NOT_FOUND", "Underlying file not found on disk.")

    quoted = (filename or "").replace("\r", "").replace("\n", "").replace('"', "'")
    cd = f'{disposition}; filename="{quoted}"; filename*=UTF-8\'\'{urllib.parse.quote(filename)}'

    resp = web.FileResponse(abs_path)
    resp.content_type = content_type
    resp.headers["Content-Disposition"] = cd
    return resp


@ROUTES.post("/api/assets/from-hash")
async def create_asset_from_hash(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        body = schemas_in.CreateFromHashBody.model_validate(payload)
    except ValidationError as ve:
        return _validation_error_response("INVALID_BODY", ve)
    except Exception:
        return _error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    result = await assets_manager.create_asset_from_hash(
        hash_str=body.hash,
        name=body.name,
        tags=body.tags,
        user_metadata=body.user_metadata,
    )
    if result is None:
        return _error_response(404, "ASSET_NOT_FOUND", f"Asset content {body.hash} does not exist")
    return web.json_response(result.model_dump(mode="json"), status=201)


@ROUTES.post("/api/assets")
async def upload_asset(request: web.Request) -> web.Response:
    """Multipart/form-data endpoint for Asset uploads."""

    if not (request.content_type or "").lower().startswith("multipart/"):
        return _error_response(415, "UNSUPPORTED_MEDIA_TYPE", "Use multipart/form-data for uploads.")

    reader = await request.multipart()

    file_field = None
    file_client_name: Optional[str] = None
    tags_raw: list[str] = []
    provided_name: Optional[str] = None
    user_metadata_raw: Optional[str] = None
    file_written = 0

    tmp_path: Optional[str] = None
    while True:
        field = await reader.next()
        if field is None:
            break

        fname = getattr(field, "name", None) or ""
        if fname == "file":
            # Save to temp
            uploads_root = os.path.join(folder_paths.get_temp_directory(), "uploads")
            unique_dir = os.path.join(uploads_root, uuid.uuid4().hex)
            os.makedirs(unique_dir, exist_ok=True)
            tmp_path = os.path.join(unique_dir, ".upload.part")

            file_field = field
            file_client_name = (field.filename or "").strip()
            try:
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = await field.read_chunk(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        file_written += len(chunk)
            except Exception:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                finally:
                    return _error_response(500, "UPLOAD_IO_ERROR", "Failed to receive and store uploaded file.")
        elif fname == "tags":
            tags_raw.append((await field.text()) or "")
        elif fname == "name":
            provided_name = (await field.text()) or None
        elif fname == "user_metadata":
            user_metadata_raw = (await field.text()) or None

    if file_field is None:
        return _error_response(400, "MISSING_FILE", "Form must include a 'file' part.")

    if file_written == 0:
        try:
            os.remove(tmp_path)
        finally:
            return _error_response(400, "EMPTY_UPLOAD", "Uploaded file is empty.")

    try:
        spec = schemas_in.UploadAssetSpec.model_validate({
            "tags": tags_raw,
            "name": provided_name,
            "user_metadata": user_metadata_raw,
        })
    except ValidationError as ve:
        try:
            os.remove(tmp_path)
        finally:
            return _validation_error_response("INVALID_BODY", ve)

    if spec.tags[0] == "models" and spec.tags[1] not in folder_paths.folder_names_and_paths:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return _error_response(400, "INVALID_BODY", f"unknown models category '{spec.tags[1]}'")

    try:
        created = await assets_manager.upload_asset_from_temp_path(
            spec,
            temp_path=tmp_path,
            client_filename=file_client_name,
        )
        return web.json_response(created.model_dump(mode="json"), status=201)
    except ValueError:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return _error_response(400, "BAD_REQUEST", "Invalid inputs.")
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return _error_response(500, "INTERNAL", "Unexpected server error.")


@ROUTES.get("/api/assets/{id:\\d+}")
async def get_asset(request: web.Request) -> web.Response:
    asset_info_id_raw = request.match_info.get("id")
    try:
        asset_info_id = int(asset_info_id_raw)
    except Exception:
        return _error_response(400, "INVALID_ID", f"AssetInfo id '{asset_info_id_raw}' is not a valid integer.")

    try:
        result = await assets_manager.get_asset(asset_info_id=asset_info_id)
    except ValueError as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        return _error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.put("/api/assets/{id:\\d+}")
async def update_asset(request: web.Request) -> web.Response:
    asset_info_id_raw = request.match_info.get("id")
    try:
        asset_info_id = int(asset_info_id_raw)
    except Exception:
        return _error_response(400, "INVALID_ID", f"AssetInfo id '{asset_info_id_raw}' is not a valid integer.")

    try:
        body = schemas_in.UpdateAssetBody.model_validate(await request.json())
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
    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.delete("/api/assets/{id:\\d+}")
async def delete_asset(request: web.Request) -> web.Response:
    asset_info_id_raw = request.match_info.get("id")
    try:
        asset_info_id = int(asset_info_id_raw)
    except Exception:
        return _error_response(400, "INVALID_ID", f"AssetInfo id '{asset_info_id_raw}' is not a valid integer.")

    try:
        deleted = await assets_manager.delete_asset_reference(asset_info_id=asset_info_id)
    except Exception:
        return _error_response(500, "INTERNAL", "Unexpected server error.")

    if not deleted:
        return _error_response(404, "ASSET_NOT_FOUND", f"AssetInfo {asset_info_id} not found.")
    return web.Response(status=204)


@ROUTES.get("/api/tags")
async def get_tags(request: web.Request) -> web.Response:
    query_map = dict(request.rel_url.query)

    try:
        query = schemas_in.TagsListQuery.model_validate(query_map)
    except ValidationError as ve:
        return web.json_response(
            {"error": {"code": "INVALID_QUERY", "message": "Invalid query parameters", "details": ve.errors()}},
            status=400,
        )

    result = await assets_manager.list_tags(
        prefix=query.prefix,
        limit=query.limit,
        offset=query.offset,
        order=query.order,
        include_zero=query.include_zero,
    )
    return web.json_response(result.model_dump(mode="json"))


@ROUTES.post("/api/assets/{id:\\d+}/tags")
async def add_asset_tags(request: web.Request) -> web.Response:
    asset_info_id_raw = request.match_info.get("id")
    try:
        asset_info_id = int(asset_info_id_raw)
    except Exception:
        return _error_response(400, "INVALID_ID", f"AssetInfo id '{asset_info_id_raw}' is not a valid integer.")

    try:
        payload = await request.json()
        data = schemas_in.TagsAdd.model_validate(payload)
    except ValidationError as ve:
        return _error_response(400, "INVALID_BODY", "Invalid JSON body for tags add.", {"errors": ve.errors()})
    except Exception:
        return _error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    try:
        result = await assets_manager.add_tags_to_asset(
            asset_info_id=asset_info_id,
            tags=data.tags,
            origin="manual",
            added_by=None,
        )
    except ValueError as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        return _error_response(500, "INTERNAL", "Unexpected server error.")

    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.delete("/api/assets/{id:\\d+}/tags")
async def delete_asset_tags(request: web.Request) -> web.Response:
    asset_info_id_raw = request.match_info.get("id")
    try:
        asset_info_id = int(asset_info_id_raw)
    except Exception:
        return _error_response(400, "INVALID_ID", f"AssetInfo id '{asset_info_id_raw}' is not a valid integer.")

    try:
        payload = await request.json()
        data = schemas_in.TagsRemove.model_validate(payload)
    except ValidationError as ve:
        return _error_response(400, "INVALID_BODY", "Invalid JSON body for tags remove.", {"errors": ve.errors()})
    except Exception:
        return _error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    try:
        result = await assets_manager.remove_tags_from_asset(
            asset_info_id=asset_info_id,
            tags=data.tags,
        )
    except ValueError as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        return _error_response(500, "INTERNAL", "Unexpected server error.")

    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.post("/api/assets/scan/schedule")
async def schedule_asset_scan(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    try:
        body = schemas_in.ScheduleAssetScanBody.model_validate(payload)
    except ValidationError as ve:
        return _validation_error_response("INVALID_BODY", ve)

    states = await assets_scanner.schedule_scans(body.roots)
    return web.json_response(states.model_dump(mode="json"), status=202)


@ROUTES.get("/api/assets/scan")
async def get_asset_scan_status(request: web.Request) -> web.Response:
    root = request.query.get("root", "").strip().lower()
    states = assets_scanner.current_statuses()
    if root in {"models", "input", "output"}:
        states = [s for s in states.scans if s.root == root]  # type: ignore
        states = schemas_out.AssetScanStatusResponse(scans=states)
    return web.json_response(states.model_dump(mode="json"), status=200)


def register_assets_routes(app: web.Application) -> None:
    app.add_routes(ROUTES)


def _error_response(status: int, code: str, message: str, details: Optional[dict] = None) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message, "details": details or {}}}, status=status)


def _validation_error_response(code: str, ve: ValidationError) -> web.Response:
    return _error_response(400, code, "Validation failed.", {"errors": ve.errors()})

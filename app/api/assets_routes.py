import contextlib
import os
import urllib.parse
import uuid
from typing import Optional

from aiohttp import web
from pydantic import ValidationError

import folder_paths

from .. import assets_manager, assets_scanner, user_manager
from . import schemas_in, schemas_out

ROUTES = web.RouteTableDef()
UserManager: Optional[user_manager.UserManager] = None

# UUID regex (canonical hyphenated form, case-insensitive)
UUID_RE = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"


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
    qp = request.rel_url.query
    query_dict = {}
    if "include_tags" in qp:
        query_dict["include_tags"] = qp.getall("include_tags")
    if "exclude_tags" in qp:
        query_dict["exclude_tags"] = qp.getall("exclude_tags")
    for k in ("name_contains", "metadata_filter", "limit", "offset", "sort", "order"):
        v = qp.get(k)
        if v is not None:
            query_dict[k] = v

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
        owner_id=UserManager.get_request_user_id(request),
    )
    return web.json_response(payload.model_dump(mode="json"))


@ROUTES.get(f"/api/assets/{{id:{UUID_RE}}}/content")
async def download_asset_content(request: web.Request) -> web.Response:
    disposition = request.query.get("disposition", "attachment").lower().strip()
    if disposition not in {"inline", "attachment"}:
        disposition = "attachment"

    try:
        abs_path, content_type, filename = await assets_manager.resolve_asset_content_for_download(
            asset_info_id=str(uuid.UUID(request.match_info["id"])),
            owner_id=UserManager.get_request_user_id(request),
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
        owner_id=UserManager.get_request_user_id(request),
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

    file_present = False
    file_client_name: Optional[str] = None
    tags_raw: list[str] = []
    provided_name: Optional[str] = None
    user_metadata_raw: Optional[str] = None
    provided_hash: Optional[str] = None
    provided_hash_exists: Optional[bool] = None

    file_written = 0
    tmp_path: Optional[str] = None
    while True:
        field = await reader.next()
        if field is None:
            break

        fname = getattr(field, "name", "") or ""

        if fname == "hash":
            try:
                s = ((await field.text()) or "").strip().lower()
            except Exception:
                return _error_response(400, "INVALID_HASH", "hash must be like 'blake3:<hex>'")

            if s:
                if ":" not in s:
                    return _error_response(400, "INVALID_HASH", "hash must be like 'blake3:<hex>'")
                algo, digest = s.split(":", 1)
                if algo != "blake3" or not digest or any(c for c in digest if c not in "0123456789abcdef"):
                    return _error_response(400, "INVALID_HASH", "hash must be like 'blake3:<hex>'")
                provided_hash = f"{algo}:{digest}"
                try:
                    provided_hash_exists = await assets_manager.asset_exists(asset_hash=provided_hash)
                except Exception:
                    provided_hash_exists = None  # do not fail the whole request here

        elif fname == "file":
            file_present = True
            file_client_name = (field.filename or "").strip()

            if provided_hash and provided_hash_exists is True:
                # If client supplied a hash that we know exists, drain but do not write to disk
                try:
                    while True:
                        chunk = await field.read_chunk(8 * 1024 * 1024)
                        if not chunk:
                            break
                        file_written += len(chunk)
                except Exception:
                    return _error_response(500, "UPLOAD_IO_ERROR", "Failed to receive uploaded file.")
                continue  # Do not create temp file; we will create AssetInfo from the existing content

            # Otherwise, store to temp for hashing/ingest
            uploads_root = os.path.join(folder_paths.get_temp_directory(), "uploads")
            unique_dir = os.path.join(uploads_root, uuid.uuid4().hex)
            os.makedirs(unique_dir, exist_ok=True)
            tmp_path = os.path.join(unique_dir, ".upload.part")

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
                    if os.path.exists(tmp_path or ""):
                        os.remove(tmp_path)
                finally:
                    return _error_response(500, "UPLOAD_IO_ERROR", "Failed to receive and store uploaded file.")
        elif fname == "tags":
            tags_raw.append((await field.text()) or "")
        elif fname == "name":
            provided_name = (await field.text()) or None
        elif fname == "user_metadata":
            user_metadata_raw = (await field.text()) or None

    # If client did not send file, and we are not doing a from-hash fast path -> error
    if not file_present and not (provided_hash and provided_hash_exists):
        return _error_response(400, "MISSING_FILE", "Form must include a 'file' part or a known 'hash'.")

    if file_present and file_written == 0 and not (provided_hash and provided_hash_exists):
        # Empty upload is only acceptable if we are fast-pathing from existing hash
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            return _error_response(400, "EMPTY_UPLOAD", "Uploaded file is empty.")

    try:
        spec = schemas_in.UploadAssetSpec.model_validate({
            "tags": tags_raw,
            "name": provided_name,
            "user_metadata": user_metadata_raw,
            "hash": provided_hash,
        })
    except ValidationError as ve:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            return _validation_error_response("INVALID_BODY", ve)

    # Validate models category against configured folders (consistent with previous behavior)
    if spec.tags and spec.tags[0] == "models":
        if len(spec.tags) < 2 or spec.tags[1] not in folder_paths.folder_names_and_paths:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return _error_response(
                400, "INVALID_BODY", f"unknown models category '{spec.tags[1] if len(spec.tags) >= 2 else ''}'"
            )

    owner_id = UserManager.get_request_user_id(request)

    # Fast path: if a valid provided hash exists, create AssetInfo without writing anything
    if spec.hash and provided_hash_exists is True:
        try:
            result = await assets_manager.create_asset_from_hash(
                hash_str=spec.hash,
                name=spec.name or (spec.hash.split(":", 1)[1]),
                tags=spec.tags,
                user_metadata=spec.user_metadata or {},
                owner_id=owner_id,
            )
        except Exception:
            return _error_response(500, "INTERNAL", "Unexpected server error.")

        if result is None:
            return _error_response(404, "ASSET_NOT_FOUND", f"Asset content {spec.hash} does not exist")

        # Drain temp if we accidentally saved (e.g., hash field came after file)
        if tmp_path and os.path.exists(tmp_path):
            with contextlib.suppress(Exception):
                os.remove(tmp_path)

        status = 200 if (not result.created_new) else 201
        return web.json_response(result.model_dump(mode="json"), status=status)

    # Otherwise, we must have a temp file path to ingest
    if not tmp_path or not os.path.exists(tmp_path):
        # The only case we reach here without a temp file is: client sent a hash that does not exist and no file
        return _error_response(404, "ASSET_NOT_FOUND", "Provided hash not found and no file uploaded.")

    try:
        created = await assets_manager.upload_asset_from_temp_path(
            spec,
            temp_path=tmp_path,
            client_filename=file_client_name,
            owner_id=owner_id,
            expected_asset_hash=spec.hash,
        )
        status = 201 if created.created_new else 200
        return web.json_response(created.model_dump(mode="json"), status=status)
    except ValueError as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        msg = str(e)
        if "HASH_MISMATCH" in msg or msg.strip().upper() == "HASH_MISMATCH":
            return _error_response(
                400,
                "HASH_MISMATCH",
                "Uploaded file hash does not match provided hash.",
            )
        return _error_response(400, "BAD_REQUEST", "Invalid inputs.")
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return _error_response(500, "INTERNAL", "Unexpected server error.")


@ROUTES.get(f"/api/assets/{{id:{UUID_RE}}}")
async def get_asset(request: web.Request) -> web.Response:
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
    try:
        result = await assets_manager.get_asset(
            asset_info_id=asset_info_id,
            owner_id=UserManager.get_request_user_id(request),
        )
    except ValueError as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        return _error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.put(f"/api/assets/{{id:{UUID_RE}}}")
async def update_asset(request: web.Request) -> web.Response:
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
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
            owner_id=UserManager.get_request_user_id(request),
        )
    except (ValueError, PermissionError) as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        return _error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.put(f"/api/assets/{{id:{UUID_RE}}}/preview")
async def set_asset_preview(request: web.Request) -> web.Response:
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
    try:
        body = schemas_in.SetPreviewBody.model_validate(await request.json())
    except ValidationError as ve:
        return _validation_error_response("INVALID_BODY", ve)
    except Exception:
        return _error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    try:
        result = await assets_manager.set_asset_preview(
            asset_info_id=asset_info_id,
            preview_asset_id=body.preview_id,
            owner_id=UserManager.get_request_user_id(request),
        )
    except (PermissionError, ValueError) as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        return _error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.delete(f"/api/assets/{{id:{UUID_RE}}}")
async def delete_asset(request: web.Request) -> web.Response:
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
    delete_content = request.query.get("delete_content")
    delete_content = True if delete_content is None else delete_content.lower() not in {"0", "false", "no"}

    try:
        deleted = await assets_manager.delete_asset_reference(
            asset_info_id=asset_info_id,
            owner_id=UserManager.get_request_user_id(request),
            delete_content_if_orphan=delete_content,
        )
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
        owner_id=UserManager.get_request_user_id(request),
    )
    return web.json_response(result.model_dump(mode="json"))


@ROUTES.post(f"/api/assets/{{id:{UUID_RE}}}/tags")
async def add_asset_tags(request: web.Request) -> web.Response:
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
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
            owner_id=UserManager.get_request_user_id(request),
        )
    except (ValueError, PermissionError) as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        return _error_response(500, "INTERNAL", "Unexpected server error.")

    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.delete(f"/api/assets/{{id:{UUID_RE}}}/tags")
async def delete_asset_tags(request: web.Request) -> web.Response:
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
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
            owner_id=UserManager.get_request_user_id(request),
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


def register_assets_system(app: web.Application, user_manager_instance: user_manager.UserManager) -> None:
    global UserManager
    UserManager = user_manager_instance
    app.add_routes(ROUTES)


def _error_response(status: int, code: str, message: str, details: Optional[dict] = None) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message, "details": details or {}}}, status=status)


def _validation_error_response(code: str, ve: ValidationError) -> web.Response:
    return _error_response(400, code, "Validation failed.", {"errors": ve.json()})

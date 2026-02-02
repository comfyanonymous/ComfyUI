import logging
import uuid
import urllib.parse
import os
import contextlib
from aiohttp import web

from pydantic import ValidationError

import app.assets.manager as manager
from app import user_manager
from app.assets.api import schemas_in
from app.assets.helpers import get_query_dict
from app.assets.scanner import seed_assets

import folder_paths

ROUTES = web.RouteTableDef()
USER_MANAGER: user_manager.UserManager | None = None

# UUID regex (canonical hyphenated form, case-insensitive)
UUID_RE = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

# Note to any custom node developers reading this code:
# The assets system is not yet fully implemented, do not rely on the code in /app/assets remaining the same.

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
    hash_str = request.match_info.get("hash", "").strip().lower()
    if not hash_str or ":" not in hash_str:
        return _error_response(400, "INVALID_HASH", "hash must be like 'blake3:<hex>'")
    algo, digest = hash_str.split(":", 1)
    if algo != "blake3" or not digest or any(c for c in digest if c not in "0123456789abcdef"):
        return _error_response(400, "INVALID_HASH", "hash must be like 'blake3:<hex>'")
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
    return web.json_response(payload.model_dump(mode="json", exclude_none=True))


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


@ROUTES.get(f"/api/assets/{{id:{UUID_RE}}}/content")
async def download_asset_content(request: web.Request) -> web.Response:
    # question: do we need disposition? could we just stick with one of these?
    disposition = request.query.get("disposition", "attachment").lower().strip()
    if disposition not in {"inline", "attachment"}:
        disposition = "attachment"

    try:
        abs_path, content_type, filename = manager.resolve_asset_content_for_download(
            asset_info_id=str(uuid.UUID(request.match_info["id"])),
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
    except ValueError as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve))
    except NotImplementedError as nie:
        return _error_response(501, "BACKEND_UNSUPPORTED", str(nie))
    except FileNotFoundError:
        return _error_response(404, "FILE_NOT_FOUND", "Underlying file not found on disk.")

    quoted = (filename or "").replace("\r", "").replace("\n", "").replace('"', "'")
    cd = f'{disposition}; filename="{quoted}"; filename*=UTF-8\'\'{urllib.parse.quote(filename)}'

    file_size = os.path.getsize(abs_path)
    logging.info(
        "download_asset_content: path=%s, size=%d bytes (%.2f MB), content_type=%s, filename=%s",
        abs_path,
        file_size,
        file_size / (1024 * 1024),
        content_type,
        filename,
    )

    async def file_sender():
        chunk_size = 64 * 1024
        with open(abs_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    return web.Response(
        body=file_sender(),
        content_type=content_type,
        headers={
            "Content-Disposition": cd,
            "Content-Length": str(file_size),
        },
    )


@ROUTES.post("/api/assets/from-hash")
async def create_asset_from_hash(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        body = schemas_in.CreateFromHashBody.model_validate(payload)
    except ValidationError as ve:
        return _validation_error_response("INVALID_BODY", ve)
    except Exception:
        return _error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    result = manager.create_asset_from_hash(
        hash_str=body.hash,
        name=body.name,
        tags=body.tags,
        user_metadata=body.user_metadata,
        owner_id=USER_MANAGER.get_request_user_id(request),
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
    file_client_name: str | None = None
    tags_raw: list[str] = []
    provided_name: str | None = None
    user_metadata_raw: str | None = None
    provided_hash: str | None = None
    provided_hash_exists: bool | None = None

    file_written = 0
    tmp_path: str | None = None
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
                    provided_hash_exists = manager.asset_exists(asset_hash=provided_hash)
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

    owner_id = USER_MANAGER.get_request_user_id(request)

    # Fast path: if a valid provided hash exists, create AssetInfo without writing anything
    if spec.hash and provided_hash_exists is True:
        try:
            result = manager.create_asset_from_hash(
                hash_str=spec.hash,
                name=spec.name or (spec.hash.split(":", 1)[1]),
                tags=spec.tags,
                user_metadata=spec.user_metadata or {},
                owner_id=owner_id,
            )
        except Exception:
            logging.exception("create_asset_from_hash failed for hash=%s, owner_id=%s", spec.hash, owner_id)
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
        created = manager.upload_asset_from_temp_path(
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
        logging.exception("upload_asset_from_temp_path failed for tmp_path=%s, owner_id=%s", tmp_path, owner_id)
        return _error_response(500, "INTERNAL", "Unexpected server error.")


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
        result = manager.update_asset(
            asset_info_id=asset_info_id,
            name=body.name,
            user_metadata=body.user_metadata,
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
    except (ValueError, PermissionError) as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        logging.exception(
            "update_asset failed for asset_info_id=%s, owner_id=%s",
            asset_info_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.delete(f"/api/assets/{{id:{UUID_RE}}}")
async def delete_asset(request: web.Request) -> web.Response:
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
    delete_content = request.query.get("delete_content")
    delete_content = True if delete_content is None else delete_content.lower() not in {"0", "false", "no"}

    try:
        deleted = manager.delete_asset_reference(
            asset_info_id=asset_info_id,
            owner_id=USER_MANAGER.get_request_user_id(request),
            delete_content_if_orphan=delete_content,
        )
    except Exception:
        logging.exception(
            "delete_asset_reference failed for asset_info_id=%s, owner_id=%s",
            asset_info_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _error_response(500, "INTERNAL", "Unexpected server error.")

    if not deleted:
        return _error_response(404, "ASSET_NOT_FOUND", f"AssetInfo {asset_info_id} not found.")
    return web.Response(status=204)


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
        result = manager.add_tags_to_asset(
            asset_info_id=asset_info_id,
            tags=data.tags,
            origin="manual",
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
    except (ValueError, PermissionError) as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        logging.exception(
            "add_tags_to_asset failed for asset_info_id=%s, owner_id=%s",
            asset_info_id,
            USER_MANAGER.get_request_user_id(request),
        )
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
        result = manager.remove_tags_from_asset(
            asset_info_id=asset_info_id,
            tags=data.tags,
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
    except ValueError as ve:
        return _error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        logging.exception(
            "remove_tags_from_asset failed for asset_info_id=%s, owner_id=%s",
            asset_info_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _error_response(500, "INTERNAL", "Unexpected server error.")

    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.post("/api/assets/seed")
async def seed_assets_endpoint(request: web.Request) -> web.Response:
    """Trigger asset seeding for specified roots (models, input, output)."""
    try:
        payload = await request.json()
        roots = payload.get("roots", ["models", "input", "output"])
    except Exception:
        roots = ["models", "input", "output"]

    valid_roots = [r for r in roots if r in ("models", "input", "output")]
    if not valid_roots:
        return _error_response(400, "INVALID_BODY", "No valid roots specified")

    try:
        seed_assets(tuple(valid_roots))
    except Exception:
        logging.exception("seed_assets failed for roots=%s", valid_roots)
        return _error_response(500, "INTERNAL", "Seed operation failed")

    return web.json_response({"seeded": valid_roots}, status=200)

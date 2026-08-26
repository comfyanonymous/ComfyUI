"""Upload-vs-scanner-vs-output hashing behaviour in the B asset branch.

These tests pin the product decision (Q2, 2026-08-23): **uploads always hash**,
regardless of the ``--enable-asset-hashing`` flag, while the **scanner** and
**workflow-output** paths stay gated on that flag.

The shared ComfyUI subprocess (see ``conftest.comfy_url_and_proc``) is
session-scoped and its hashing mode is fixed once per pytest session: it comes
up with ``--enable-asset-hashing`` iff any collected test carries the
``hashing_on`` marker. This file carries no such marker, so an *isolated* run of
this file (``pytest tests-unit/assets_test/test_upload_hashing_modes.py``) boots
the server in **on mode** (assets enabled, hashing flag OFF) — the exact
configuration the upload assertions target. A *full-suite* run collects
``hashing_on`` tests from sibling files, so the shared server comes up in hash
mode; the always-hash upload assertions still hold there (uploads hash in both
modes), the scanner assertion skips (see ``server_hashing_enabled``), and the
output assertion is self-contained (it drives the ingest function in-process
with the flag forced off).

The authoritative hash signal is the ``asset_contents.hash`` column: a raw
64-char BLAKE3 hex digest when hashed, or ``NULL`` when not. (The HTTP layer
omits null hashes entirely via ``exclude_none=True`` and prefixes non-null ones
with ``blake3:``, so the DB column is the stable thing to assert on.)
"""
from __future__ import annotations

import os
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
import requests

# --------------------------------------------------------------------------- #
# DB access helpers (read the subprocess's sqlite file directly, read-only)
# --------------------------------------------------------------------------- #


def _db_path(comfy_tmp_base_dir: Path, request: pytest.FixtureRequest) -> str:
    url = request.config.getoption("--db-url")
    if url and url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    return str(comfy_tmp_base_dir / "assets-test.sqlite3")


def _query(db_path: str, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    last_err: Exception | None = None
    for _ in range(20):
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
            try:
                return list(con.execute(sql, params))
            finally:
                con.close()
        except sqlite3.OperationalError as e:
            last_err = e
            time.sleep(0.1)
    raise AssertionError(f"sqlite read failed: {last_err}")


def _content_hash_for_asset(db_path: str, asset_id: str) -> str | None:
    """Return the ``asset_contents.hash`` backing an asset record (raw hex or None)."""
    rows = _query(
        db_path,
        "SELECT c.hash FROM asset_contents c "
        "JOIN assets a ON a.content_id = c.id "
        "WHERE a.id = ?",
        (asset_id,),
    )
    assert rows, f"no content row found for asset {asset_id}"
    return rows[0][0]


def _is_blake3_hex(value: str | None) -> bool:
    """True iff ``value`` is a bare 64-char lowercase BLAKE3 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value.lower())
    )


# --------------------------------------------------------------------------- #
# Upload helpers
# --------------------------------------------------------------------------- #


def _upload_via_api(
    http: requests.Session, base: str, *, name: str, tags: list[str], data: bytes
) -> tuple[int, dict[str, Any]]:
    files = {"file": (name, data, "application/octet-stream")}
    form = {"tags": json.dumps(tags), "name": name, "user_metadata": json.dumps({})}
    r = http.post(base + "/api/assets", files=files, data=form, timeout=120)
    return r.status_code, r.json()


def _upload_via_image(
    http: requests.Session, base: str, *, name: str, data: bytes, upload_type: str = "input"
) -> tuple[int, dict[str, Any]]:
    files = {"image": (name, data, "image/png")}
    form = {"type": upload_type, "subfolder": ""}
    r = http.post(base + "/upload/image", files=files, data=form, timeout=120)
    return r.status_code, r.json()


def _unique_bytes(seed: str, size: int = 4096) -> bytes:
    return uuid.uuid4().bytes + seed.encode("utf-8").ljust(size, b"\0")[: size - 16]


@pytest.fixture(scope="session")
def server_hashing_enabled(request: pytest.FixtureRequest) -> bool:
    """Whether the shared subprocess booted with ``--enable-asset-hashing``.

    Mirrors the exact decision made in ``conftest.comfy_url_and_proc``. The
    server fixture is session-scoped, so its mode is fixed once per pytest
    session: ON iff the flag/markexpr is set or any collected test carries the
    ``hashing_on`` marker. An isolated run of this file has no such markers (so
    this returns False → on mode); a full-suite run collects ``hashing_on``
    tests from sibling files (so this returns True → hash mode).
    """
    markexpr = request.config.getoption("markexpr") or ""
    return bool(
        request.config.getoption("--enable-asset-hashing")
        or "hashing_on" in markexpr
        or any(item.get_closest_marker("hashing_on") for item in request.session.items)
    )


# --------------------------------------------------------------------------- #
# Uploads ALWAYS hash — even in on mode (hashing flag OFF). (Q2 decision)
# --------------------------------------------------------------------------- #


def test_upload_via_api_hashes_in_on_mode(http, api_base, comfy_tmp_base_dir, request):
    """POST /api/assets (client-bytes path) hashes its content in on mode."""
    db_path = _db_path(comfy_tmp_base_dir, request)
    data = _unique_bytes("api-hashes")
    status, body = _upload_via_api(
        http, api_base, name="api_hashes.bin", tags=["output", "unit-tests"], data=data
    )
    assert status in (200, 201), body
    digest = _content_hash_for_asset(db_path, body["id"])
    assert _is_blake3_hex(digest), f"expected a blake3 digest, got {digest!r}"


def test_upload_via_image_hashes_in_on_mode(http, api_base, comfy_tmp_base_dir, request):
    """POST /upload/image (in-place registration path) hashes its content in on mode."""
    db_path = _db_path(comfy_tmp_base_dir, request)
    data = _unique_bytes("image-hashes")
    status, body = _upload_via_image(http, api_base, name="image_hashes.png", data=data)
    assert status == 200, body
    asset = body.get("asset")
    assert asset and asset.get("id"), f"/upload/image did not register an asset: {body}"
    digest = _content_hash_for_asset(db_path, asset["id"])
    assert _is_blake3_hex(digest), f"expected a blake3 digest, got {digest!r}"


def test_upload_dedup_works_in_on_mode(http, api_base):
    """Same bytes + same name uploaded twice dedup to one entity, even in on mode."""
    data = _unique_bytes("dedup-on-mode")
    status1, first = _upload_via_api(
        http, api_base, name="dedup_on.bin", tags=["output", "unit-tests"], data=data
    )
    status2, second = _upload_via_api(
        http, api_base, name="dedup_on.bin", tags=["output", "unit-tests"], data=data
    )
    assert status1 in (200, 201), first
    assert status2 in (200, 201), second
    assert second["id"] == first["id"], "dedup should return the same entity id"
    assert second.get("created_new") is False


# --------------------------------------------------------------------------- #
# Scanner and workflow-output paths STAY gated on the hashing flag.
# --------------------------------------------------------------------------- #


def test_seeded_file_not_hashed_in_on_mode(
    http, api_base, comfy_tmp_base_dir, request, server_hashing_enabled
):
    """A scanned (seeded) on-disk file is NOT hashed while the flag is off.

    The scanner gate is deliberately untouched by the upload un-gating. It only
    exhibits the not-hashed outcome with the flag off, so this skips when the
    shared server booted in hash mode (a full-suite run).
    """
    if server_hashing_enabled:
        pytest.skip(
            "scanner hashing gate only observable with the hashing flag OFF; "
            "a full-suite run forces the shared server into hash mode"
        )
    from .helpers import trigger_sync_seed_assets

    db_path = _db_path(comfy_tmp_base_dir, request)
    ckpt_dir = comfy_tmp_base_dir / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    disk_path = ckpt_dir / f"seeded_{uuid.uuid4().hex}.safetensors"
    disk_path.write_bytes(uuid.uuid4().bytes + b"S" * (4096 - 16))

    row_hash: str | None = None
    for _ in range(5):
        trigger_sync_seed_assets(http, api_base)
        rows = _query(
            db_path,
            "SELECT hash FROM asset_contents WHERE path = ? AND is_missing = 0",
            (str(disk_path),),
        )
        if rows:
            row_hash = rows[0][0]
            break
    else:
        raise AssertionError(f"seeded file never produced a content row: {disk_path}")

    assert row_hash is None, "scanner must not hash seeded files while the flag is off"


def test_output_not_hashed_in_on_mode(monkeypatch):
    """A workflow output is NOT hashed while the flag is off (output gate held).

    Drives ``register_output_file_b`` in-process — the exact function main.py
    calls for each saved output — with the runtime hashing flag forced off and
    the DB pointed at an in-memory engine. This is deterministic regardless of
    the shared subprocess's mode.
    """
    from contextlib import contextmanager

    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session as SASession

    import app.assets.mode as mode_module
    import folder_paths
    from app.assets.database.models import AssetContent, Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    @contextmanager
    def _fake_create_session():
        with SASession(engine) as session:
            yield session

    monkeypatch.setattr(mode_module, "hashing_enabled", lambda: False)
    monkeypatch.setattr(
        "app.assets.services.ingest.create_session", _fake_create_session
    )

    from app.assets.services.ingest import register_output_file_b

    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"render_{uuid.uuid4().hex}.png")
    with open(out_path, "wb") as fh:
        fh.write(b"fake-output-bytes")

    try:
        register_output_file_b(out_path, job_id="job-out")
        with SASession(engine) as session:
            content = session.execute(
                select(AssetContent).where(
                    AssetContent.path == os.path.abspath(out_path)
                )
            ).scalar_one()
            assert content.hash is None, (
                "workflow output must not be hashed while the hashing flag is off"
            )
            assert content.is_missing is False
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)

"""Asset system startup/shutdown lifecycle: DB init, temp wipe, filesystem cleanup."""

from __future__ import annotations

import logging
import os
import shutil

import folder_paths
from sqlalchemy import select

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import delete_record
from app.assets.services.lookup import is_temp_path
from app.database.db import create_session, init_db

_excluded_scan_roots: set[str] = set()
_hash_mode_transition: str | None = None


def get_excluded_scan_roots() -> frozenset[str]:
    return frozenset(_excluded_scan_roots)


def record_hash_mode_transition_intent() -> None:
    global _hash_mode_transition
    from app.assets.services.hash_mode_state import record_transition_intent

    with create_session() as session:
        _hash_mode_transition = record_transition_intent(session)
        session.commit()


def enqueue_mode_transition_work() -> None:
    from app.assets.services.hash_mode_state import enqueue_transition_work

    with create_session() as session:
        enqueue_transition_work(session, _hash_mode_transition)
        session.commit()


def drain_mode_transition_work() -> None:
    from app.assets.services.hash_mode_state import drain_transition_queue

    with create_session() as session:
        drain_transition_queue(session)
        session.commit()


def init_db_and_state() -> None:
    init_db()
    record_hash_mode_transition_intent()


def wipe_temp_db_rows(session) -> tuple[int, int]:
    """Delete temp asset records, then temp content rows (records first — FK RESTRICT)."""
    temp_record_ids = [
        record.id
        for record in session.scalars(select(Asset)).all()
        if record.content is not None and is_temp_path(record.content.path)
    ]

    records_deleted = 0
    for record_id in temp_record_ids:
        delete_record(session, record_id)
        records_deleted += 1

    contents_deleted = 0
    for content in session.scalars(select(AssetContent)).all():
        if is_temp_path(content.path):
            session.delete(content)
            contents_deleted += 1

    session.flush()
    return records_deleted, contents_deleted


def cleanup_temp_filesystem() -> bool:
    """Remove the temp directory tree. On failure, exclude the root from scanning."""
    temp_dir = os.path.abspath(folder_paths.get_temp_directory())
    if not os.path.exists(temp_dir):
        return True
    try:
        shutil.rmtree(temp_dir)
        return True
    except OSError as exc:
        logging.warning(
            "Failed to remove temp directory %s: %s — excluding from scan for this process",
            temp_dir,
            exc,
        )
        _excluded_scan_roots.add(temp_dir)
        return False


def start_asset_seeder() -> bool:
    from app.assets.seeder import asset_seeder
    from comfy.cli_args import args

    started = asset_seeder.start(
        roots=("models", "input", "output"),
        prune_first=True,
        compute_hashes=args.enable_asset_hashing,
    )
    if started:
        logging.info("Background asset scan initiated for models, input, output")
    return started


def run_asset_startup() -> None:
    try:
        with create_session() as session:
            wipe_temp_db_rows(session)
            session.commit()
    except Exception:
        logging.exception("Temp DB row wipe failed; skipping filesystem cleanup")
        enqueue_mode_transition_work()
        drain_mode_transition_work()
        start_asset_seeder()
        return
    cleanup_temp_filesystem()
    enqueue_mode_transition_work()
    drain_mode_transition_work()
    start_asset_seeder()


def run_startup(*, enable_assets: bool) -> None:
    """Startup temp maintenance entry point, owning the enabled/disabled policy.

    Enabled: full asset startup (DB temp-row wipe -> filesystem sweep -> seeder), which
    intentionally orders row deletion before filesystem deletion and skips the sweep when
    the wipe fails. Disabled: filesystem sweep only (master parity -- master called
    cleanup_temp() unconditionally); with assets off there are no asset rows to wipe.
    """
    if enable_assets:
        run_asset_startup()
    else:
        cleanup_temp_filesystem()


def run_asset_shutdown_cleanup() -> None:
    with create_session() as session:
        wipe_temp_db_rows(session)
        session.commit()
    cleanup_temp_filesystem()

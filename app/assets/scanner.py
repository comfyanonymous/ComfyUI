import contextlib
import time
import logging
import os
import sqlalchemy

import folder_paths
from app.database.db import create_session, dependencies_available
from app.assets.helpers import (
    collect_models_files, compute_relative_filename, fast_asset_file_check, get_name_and_tags_from_asset_path,
    list_tree,prefixes_for_root, escape_like_prefix,
    RootType
)
from app.assets.database.tags import add_missing_tag_for_asset_id, ensure_tags_exist, remove_missing_tag_for_asset_id
from app.assets.database.bulk_ops import seed_from_paths_batch
from app.assets.database.models import Asset, AssetCacheState, AssetInfo


def seed_assets(roots: tuple[RootType, ...], enable_logging: bool = False) -> None:
    """
    Scan the given roots and seed the assets into the database.
    """
    if not dependencies_available():
        if enable_logging:
            logging.warning("Database dependencies not available, skipping assets scan")
        return
    t_start = time.perf_counter()
    created = 0
    skipped_existing = 0
    paths: list[str] = []
    try:
        existing_paths: set[str] = set()
        for r in roots:
            try:
                survivors: set[str] = _fast_db_consistency_pass(r, collect_existing_paths=True, update_missing_tags=True)
                if survivors:
                    existing_paths.update(survivors)
            except Exception as e:
                logging.exception("fast DB scan failed for %s: %s", r, e)

        if "models" in roots:
            paths.extend(collect_models_files())
        if "input" in roots:
            paths.extend(list_tree(folder_paths.get_input_directory()))
        if "output" in roots:
            paths.extend(list_tree(folder_paths.get_output_directory()))

        specs: list[dict] = []
        tag_pool: set[str] = set()
        for p in paths:
            abs_p = os.path.abspath(p)
            if abs_p in existing_paths:
                skipped_existing += 1
                continue
            try:
                stat_p = os.stat(abs_p, follow_symlinks=False)
            except OSError:
                continue
            # skip empty files
            if not stat_p.st_size:
                continue
            name, tags = get_name_and_tags_from_asset_path(abs_p)
            specs.append(
                {
                    "abs_path": abs_p,
                    "size_bytes": stat_p.st_size,
                    "mtime_ns": getattr(stat_p, "st_mtime_ns", int(stat_p.st_mtime * 1_000_000_000)),
                    "info_name": name,
                    "tags": tags,
                    "fname": compute_relative_filename(abs_p),
                }
            )
            for t in tags:
                tag_pool.add(t)
        # if no file specs, nothing to do
        if not specs:
            return
        with create_session() as sess:
            if tag_pool:
                ensure_tags_exist(sess, tag_pool, tag_type="user")

            result = seed_from_paths_batch(sess, specs=specs, owner_id="")
            created += result["inserted_infos"]
            sess.commit()
    finally:
        if enable_logging:
            logging.info(
                "Assets scan(roots=%s) completed in %.3fs (created=%d, skipped_existing=%d, total_seen=%d)",
                roots,
                time.perf_counter() - t_start,
                created,
                skipped_existing,
                len(paths),
            )


def _fast_db_consistency_pass(
    root: RootType,
    *,
    collect_existing_paths: bool = False,
    update_missing_tags: bool = False,
) -> set[str] | None:
    """Fast DB+FS pass for a root:
      - Toggle needs_verify per state using fast check
      - For hashed assets with at least one fast-ok state in this root: delete stale missing states
      - For seed assets with all states missing: delete Asset and its AssetInfos
      - Optionally add/remove 'missing' tags based on fast-ok in this root
      - Optionally return surviving absolute paths
    """
    prefixes = prefixes_for_root(root)
    if not prefixes:
        return set() if collect_existing_paths else None

    conds = []
    for p in prefixes:
        base = os.path.abspath(p)
        if not base.endswith(os.sep):
            base += os.sep
        escaped, esc = escape_like_prefix(base)
        conds.append(AssetCacheState.file_path.like(escaped + "%", escape=esc))

    with create_session() as sess:
        rows = (
            sess.execute(
                sqlalchemy.select(
                    AssetCacheState.id,
                    AssetCacheState.file_path,
                    AssetCacheState.mtime_ns,
                    AssetCacheState.needs_verify,
                    AssetCacheState.asset_id,
                    Asset.hash,
                    Asset.size_bytes,
                )
                .join(Asset, Asset.id == AssetCacheState.asset_id)
                .where(sqlalchemy.or_(*conds))
                .order_by(AssetCacheState.asset_id.asc(), AssetCacheState.id.asc())
            )
        ).all()

        by_asset: dict[str, dict] = {}
        for sid, fp, mtime_db, needs_verify, aid, a_hash, a_size in rows:
            acc = by_asset.get(aid)
            if acc is None:
                acc = {"hash": a_hash, "size_db": int(a_size or 0), "states": []}
                by_asset[aid] = acc

            fast_ok = False
            try:
                exists = True
                fast_ok = fast_asset_file_check(
                    mtime_db=mtime_db,
                    size_db=acc["size_db"],
                    stat_result=os.stat(fp, follow_symlinks=True),
                )
            except FileNotFoundError:
                exists = False
            except OSError:
                exists = False

            acc["states"].append({
                "sid": sid,
                "fp": fp,
                "exists": exists,
                "fast_ok": fast_ok,
                "needs_verify": bool(needs_verify),
            })

        to_set_verify: list[int] = []
        to_clear_verify: list[int] = []
        stale_state_ids: list[int] = []
        survivors: set[str] = set()

        for aid, acc in by_asset.items():
            a_hash = acc["hash"]
            states = acc["states"]
            any_fast_ok = any(s["fast_ok"] for s in states)
            all_missing = all(not s["exists"] for s in states)

            for s in states:
                if not s["exists"]:
                    continue
                if s["fast_ok"] and s["needs_verify"]:
                    to_clear_verify.append(s["sid"])
                if not s["fast_ok"] and not s["needs_verify"]:
                    to_set_verify.append(s["sid"])

            if a_hash is None:
                if states and all_missing:  # remove seed Asset completely, if no valid AssetCache exists
                    sess.execute(sqlalchemy.delete(AssetInfo).where(AssetInfo.asset_id == aid))
                    asset = sess.get(Asset, aid)
                    if asset:
                        sess.delete(asset)
                else:
                    for s in states:
                        if s["exists"]:
                            survivors.add(os.path.abspath(s["fp"]))
                continue

            if any_fast_ok:  # if Asset has at least one valid AssetCache record, remove any invalid AssetCache records
                for s in states:
                    if not s["exists"]:
                        stale_state_ids.append(s["sid"])
                if update_missing_tags:
                    with contextlib.suppress(Exception):
                        remove_missing_tag_for_asset_id(sess, asset_id=aid)
            elif update_missing_tags:
                with contextlib.suppress(Exception):
                    add_missing_tag_for_asset_id(sess, asset_id=aid, origin="automatic")

            for s in states:
                if s["exists"]:
                    survivors.add(os.path.abspath(s["fp"]))

        if stale_state_ids:
            sess.execute(sqlalchemy.delete(AssetCacheState).where(AssetCacheState.id.in_(stale_state_ids)))
        if to_set_verify:
            sess.execute(
                sqlalchemy.update(AssetCacheState)
                .where(AssetCacheState.id.in_(to_set_verify))
                .values(needs_verify=True)
            )
        if to_clear_verify:
            sess.execute(
                sqlalchemy.update(AssetCacheState)
                .where(AssetCacheState.id.in_(to_clear_verify))
                .values(needs_verify=False)
            )
        sess.commit()
        return survivors if collect_existing_paths else None

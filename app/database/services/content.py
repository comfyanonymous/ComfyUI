import contextlib
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Optional, Sequence, Union

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.dialects import postgresql as d_pg
from sqlalchemy.dialects import sqlite as d_sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload

from ..._assets_helpers import compute_relative_filename, normalize_tags
from ...storage import hashing as hashing_mod
from ..helpers import (
    ensure_tags_exist,
    escape_like_prefix,
    remove_missing_tag_for_asset_id,
)
from ..models import Asset, AssetCacheState, AssetInfo, AssetInfoMeta, AssetInfoTag, Tag
from ..timeutil import utcnow
from .info import replace_asset_info_metadata_projection
from .queries import list_cache_states_by_asset_id, pick_best_live_path


async def check_fs_asset_exists_quick(
    session: AsyncSession,
    *,
    file_path: str,
    size_bytes: Optional[int] = None,
    mtime_ns: Optional[int] = None,
) -> bool:
    """Returns True if we already track this absolute path with a HASHED asset and the cached mtime/size match."""
    locator = os.path.abspath(file_path)

    stmt = (
        sa.select(sa.literal(True))
        .select_from(AssetCacheState)
        .join(Asset, Asset.id == AssetCacheState.asset_id)
        .where(
            AssetCacheState.file_path == locator,
            Asset.hash.isnot(None),
            AssetCacheState.needs_verify.is_(False),
        )
        .limit(1)
    )

    conds = []
    if mtime_ns is not None:
        conds.append(AssetCacheState.mtime_ns == int(mtime_ns))
    if size_bytes is not None:
        conds.append(sa.or_(Asset.size_bytes == 0, Asset.size_bytes == int(size_bytes)))
    if conds:
        stmt = stmt.where(*conds)
    return (await session.execute(stmt)).first() is not None


async def seed_from_path(
    session: AsyncSession,
    *,
    abs_path: str,
    size_bytes: int,
    mtime_ns: int,
    info_name: str,
    tags: Sequence[str],
    owner_id: str = "",
    skip_tag_ensure: bool = False,
) -> None:
    """Creates Asset(hash=NULL), AssetCacheState(file_path), and AssetInfo exist for the path."""
    locator = os.path.abspath(abs_path)
    now = utcnow()
    dialect = session.bind.dialect.name

    new_asset_id = str(uuid.uuid4())
    new_info_id = str(uuid.uuid4())

    # 1) Insert Asset (hash=NULL) – no conflict expected
    asset_vals = {
        "id": new_asset_id,
        "hash": None,
        "size_bytes": size_bytes,
        "mime_type": None,
        "created_at": now,
    }
    if dialect == "sqlite":
        await session.execute(d_sqlite.insert(Asset).values(**asset_vals))
    elif dialect == "postgresql":
        await session.execute(d_pg.insert(Asset).values(**asset_vals))
    else:
        raise NotImplementedError(f"Unsupported database dialect: {dialect}")

    # 2) Try to claim file_path in AssetCacheState. Our concurrency gate.
    acs_vals = {
        "asset_id": new_asset_id,
        "file_path": locator,
        "mtime_ns": mtime_ns,
    }
    if dialect == "sqlite":
        ins_state = (
            d_sqlite.insert(AssetCacheState)
            .values(**acs_vals)
            .on_conflict_do_nothing(index_elements=[AssetCacheState.file_path])
        )
        state_inserted = int((await session.execute(ins_state)).rowcount or 0) > 0
    else:
        ins_state = (
            d_pg.insert(AssetCacheState)
            .values(**acs_vals)
            .on_conflict_do_nothing(index_elements=[AssetCacheState.file_path])
            .returning(AssetCacheState.id)
        )
        state_inserted = (await session.execute(ins_state)).scalar_one_or_none() is not None

    if not state_inserted:
        # Lost the race - clean up our orphan seed Asset and exit
        with contextlib.suppress(Exception):
            await session.execute(sa.delete(Asset).where(Asset.id == new_asset_id))
        return

    # 3) Create AssetInfo (unique(asset_id, owner_id, name)).
    fname = compute_relative_filename(locator)

    info_vals = {
        "id": new_info_id,
        "owner_id": owner_id,
        "name": info_name,
        "asset_id": new_asset_id,
        "preview_id": None,
        "user_metadata": {"filename": fname} if fname else None,
        "created_at": now,
        "updated_at": now,
        "last_access_time": now,
    }
    if dialect == "sqlite":
        ins_info = (
            d_sqlite.insert(AssetInfo)
            .values(**info_vals)
            .on_conflict_do_nothing(index_elements=[AssetInfo.asset_id, AssetInfo.owner_id, AssetInfo.name])
        )
        info_inserted = int((await session.execute(ins_info)).rowcount or 0) > 0
    else:
        ins_info = (
            d_pg.insert(AssetInfo)
            .values(**info_vals)
            .on_conflict_do_nothing(index_elements=[AssetInfo.asset_id, AssetInfo.owner_id, AssetInfo.name])
            .returning(AssetInfo.id)
        )
        info_inserted = (await session.execute(ins_info)).scalar_one_or_none() is not None

    # 4) If we actually inserted AssetInfo, attach tags and filename.
    if info_inserted:
        want = normalize_tags(tags)
        if want:
            if not skip_tag_ensure:
                await ensure_tags_exist(session, want, tag_type="user")
            tag_rows = [
                {
                    "asset_info_id": new_info_id,
                    "tag_name": t,
                    "origin": "automatic",
                    "added_at": now,
                }
                for t in want
            ]
            if dialect == "sqlite":
                ins_links = (
                    d_sqlite.insert(AssetInfoTag)
                    .values(tag_rows)
                    .on_conflict_do_nothing(index_elements=[AssetInfoTag.asset_info_id, AssetInfoTag.tag_name])
                )
            else:
                ins_links = (
                    d_pg.insert(AssetInfoTag)
                    .values(tag_rows)
                    .on_conflict_do_nothing(index_elements=[AssetInfoTag.asset_info_id, AssetInfoTag.tag_name])
                )
            await session.execute(ins_links)

        if fname:  # simple filename projection with single row
            meta_row = {
                "asset_info_id": new_info_id,
                "key": "filename",
                "ordinal": 0,
                "val_str": fname,
                "val_num": None,
                "val_bool": None,
                "val_json": None,
            }
            if dialect == "sqlite":
                await session.execute(d_sqlite.insert(AssetInfoMeta).values(**meta_row))
            else:
                await session.execute(d_pg.insert(AssetInfoMeta).values(**meta_row))


async def redirect_all_references_then_delete_asset(
    session: AsyncSession,
    *,
    duplicate_asset_id: str,
    canonical_asset_id: str,
) -> None:
    """
    Safely migrate all references from duplicate_asset_id to canonical_asset_id.

    - If an AssetInfo for (owner_id, name) already exists on the canonical asset,
      merge tags, metadata, times, and preview, then delete the duplicate AssetInfo.
    - Otherwise, simply repoint the AssetInfo.asset_id.
    - Always retarget AssetCacheState rows.
    - Finally delete the duplicate Asset row.
    """
    if duplicate_asset_id == canonical_asset_id:
        return

    # 1) Migrate AssetInfo rows one-by-one to avoid UNIQUE conflicts.
    dup_infos = (
        await session.execute(
            select(AssetInfo).options(noload(AssetInfo.tags)).where(AssetInfo.asset_id == duplicate_asset_id)
        )
    ).unique().scalars().all()

    for info in dup_infos:
        # Try to find an existing collision on canonical
        existing = (
            await session.execute(
                select(AssetInfo)
                .options(noload(AssetInfo.tags))
                .where(
                    AssetInfo.asset_id == canonical_asset_id,
                    AssetInfo.owner_id == info.owner_id,
                    AssetInfo.name == info.name,
                )
                .limit(1)
            )
        ).unique().scalars().first()

        if existing:
            merged_meta = dict(existing.user_metadata or {})
            other_meta = info.user_metadata or {}
            for k, v in other_meta.items():
                if k not in merged_meta:
                    merged_meta[k] = v
            if merged_meta != (existing.user_metadata or {}):
                await replace_asset_info_metadata_projection(
                    session,
                    asset_info_id=existing.id,
                    user_metadata=merged_meta,
                )

            existing_tags = {
                t for (t,) in (
                    await session.execute(
                        select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == existing.id)
                    )
                ).all()
            }
            from_tags = {
                t for (t,) in (
                    await session.execute(
                        select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == info.id)
                    )
                ).all()
            }
            to_add = sorted(from_tags - existing_tags)
            if to_add:
                await ensure_tags_exist(session, to_add, tag_type="user")
                now = utcnow()
                session.add_all([
                    AssetInfoTag(asset_info_id=existing.id, tag_name=t, origin="automatic", added_at=now)
                    for t in to_add
                ])
                await session.flush()

            if existing.preview_id is None and info.preview_id is not None:
                existing.preview_id = info.preview_id
            if info.last_access_time and (
                existing.last_access_time is None or info.last_access_time > existing.last_access_time
            ):
                existing.last_access_time = info.last_access_time
            existing.updated_at = utcnow()
            await session.flush()

            # Delete the duplicate AssetInfo (cascades will clean its tags/meta)
            await session.delete(info)
            await session.flush()
        else:
            # Simple retarget
            info.asset_id = canonical_asset_id
            info.updated_at = utcnow()
            await session.flush()

    # 2) Repoint cache states and previews
    await session.execute(
        sa.update(AssetCacheState)
        .where(AssetCacheState.asset_id == duplicate_asset_id)
        .values(asset_id=canonical_asset_id)
    )
    await session.execute(
        sa.update(AssetInfo)
        .where(AssetInfo.preview_id == duplicate_asset_id)
        .values(preview_id=canonical_asset_id)
    )

    # 3) Remove duplicate Asset
    dup = await session.get(Asset, duplicate_asset_id)
    if dup:
        await session.delete(dup)
    await session.flush()


async def compute_hash_and_dedup_for_cache_state(
    session: AsyncSession,
    *,
    state_id: int,
) -> Optional[str]:
    """
    Compute hash for the given cache state, deduplicate, and settle verify cases.

    Returns the asset_id that this state ends up pointing to, or None if file disappeared.
    """
    state = await session.get(AssetCacheState, state_id)
    if not state:
        return None

    path = state.file_path
    try:
        if not os.path.isfile(path):
            # File vanished: drop the state. If the Asset has hash=NULL and has no other states, drop the Asset too.
            asset = await session.get(Asset, state.asset_id)
            await session.delete(state)
            await session.flush()

            if asset and asset.hash is None:
                remaining = (
                    await session.execute(
                        sa.select(sa.func.count())
                        .select_from(AssetCacheState)
                        .where(AssetCacheState.asset_id == asset.id)
                    )
                ).scalar_one()
                if int(remaining or 0) == 0:
                    await session.delete(asset)
                    await session.flush()
                else:
                    await _recompute_and_apply_filename_for_asset(session, asset_id=asset.id)
            return None

        digest = await hashing_mod.blake3_hash(path)
        new_hash = f"blake3:{digest}"

        st = os.stat(path, follow_symlinks=True)
        new_size = int(st.st_size)
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))

        # Current asset of this state
        this_asset = await session.get(Asset, state.asset_id)

        # If the state got orphaned somehow (race), just reattach appropriately.
        if not this_asset:
            canonical = (
                await session.execute(sa.select(Asset).where(Asset.hash == new_hash).limit(1))
            ).scalars().first()
            if canonical:
                state.asset_id = canonical.id
            else:
                now = utcnow()
                new_asset = Asset(hash=new_hash, size_bytes=new_size, mime_type=None, created_at=now)
                session.add(new_asset)
                await session.flush()
                state.asset_id = new_asset.id
            state.mtime_ns = mtime_ns
            state.needs_verify = False
            with contextlib.suppress(Exception):
                await remove_missing_tag_for_asset_id(session, asset_id=state.asset_id)
            await session.flush()
            return state.asset_id

        # 1) Seed asset case (hash is NULL): claim or merge into canonical
        if this_asset.hash is None:
            canonical = (
                await session.execute(sa.select(Asset).where(Asset.hash == new_hash).limit(1))
            ).scalars().first()

            if canonical and canonical.id != this_asset.id:
                # Merge seed asset into canonical (safe, collision-aware)
                await redirect_all_references_then_delete_asset(
                    session,
                    duplicate_asset_id=this_asset.id,
                    canonical_asset_id=canonical.id,
                )
                state = await session.get(AssetCacheState, state_id)
                if state:
                    state.mtime_ns = mtime_ns
                    state.needs_verify = False
                    with contextlib.suppress(Exception):
                        await remove_missing_tag_for_asset_id(session, asset_id=canonical.id)
                    await _recompute_and_apply_filename_for_asset(session, asset_id=canonical.id)
                    await session.flush()
                return canonical.id

            # No canonical: try to claim the hash; handle races with a SAVEPOINT
            try:
                async with session.begin_nested():
                    this_asset.hash = new_hash
                    if int(this_asset.size_bytes or 0) == 0 and new_size > 0:
                        this_asset.size_bytes = new_size
                    await session.flush()
            except IntegrityError:
                # Someone else claimed it concurrently; fetch canonical and merge
                canonical = (
                    await session.execute(sa.select(Asset).where(Asset.hash == new_hash).limit(1))
                ).scalars().first()
                if canonical and canonical.id != this_asset.id:
                    await redirect_all_references_then_delete_asset(
                        session,
                        duplicate_asset_id=this_asset.id,
                        canonical_asset_id=canonical.id,
                    )
                    state = await session.get(AssetCacheState, state_id)
                    if state:
                        state.mtime_ns = mtime_ns
                        state.needs_verify = False
                        with contextlib.suppress(Exception):
                            await remove_missing_tag_for_asset_id(session, asset_id=canonical.id)
                        await _recompute_and_apply_filename_for_asset(session, asset_id=canonical.id)
                        await session.flush()
                    return canonical.id
                # If we got here, the integrity error was not about hash uniqueness
                raise

            # Claimed successfully
            state.mtime_ns = mtime_ns
            state.needs_verify = False
            with contextlib.suppress(Exception):
                await remove_missing_tag_for_asset_id(session, asset_id=this_asset.id)
            await _recompute_and_apply_filename_for_asset(session, asset_id=this_asset.id)
            await session.flush()
            return this_asset.id

        # 2) Verify case for hashed assets
        if this_asset.hash == new_hash:
            if int(this_asset.size_bytes or 0) == 0 and new_size > 0:
                this_asset.size_bytes = new_size
            state.mtime_ns = mtime_ns
            state.needs_verify = False
            with contextlib.suppress(Exception):
                await remove_missing_tag_for_asset_id(session, asset_id=this_asset.id)
            await _recompute_and_apply_filename_for_asset(session, asset_id=this_asset.id)
            await session.flush()
            return this_asset.id

        # Content changed on this path only: retarget THIS state, do not move AssetInfo rows
        canonical = (
            await session.execute(sa.select(Asset).where(Asset.hash == new_hash).limit(1))
        ).scalars().first()
        if canonical:
            target_id = canonical.id
        else:
            now = utcnow()
            new_asset = Asset(hash=new_hash, size_bytes=new_size, mime_type=None, created_at=now)
            session.add(new_asset)
            await session.flush()
            target_id = new_asset.id

        state.asset_id = target_id
        state.mtime_ns = mtime_ns
        state.needs_verify = False
        with contextlib.suppress(Exception):
            await remove_missing_tag_for_asset_id(session, asset_id=target_id)
        await _recompute_and_apply_filename_for_asset(session, asset_id=target_id)
        await session.flush()
        return target_id
    except Exception:
        raise


async def list_unhashed_candidates_under_prefixes(session: AsyncSession, *, prefixes: list[str]) -> list[int]:
    if not prefixes:
        return []

    conds = []
    for p in prefixes:
        base = os.path.abspath(p)
        if not base.endswith(os.sep):
            base += os.sep
        escaped, esc = escape_like_prefix(base)
        conds.append(AssetCacheState.file_path.like(escaped + "%", escape=esc))

    path_filter = sa.or_(*conds) if len(conds) > 1 else conds[0]
    if session.bind.dialect.name == "postgresql":
        stmt = (
            sa.select(AssetCacheState.id)
            .join(Asset, Asset.id == AssetCacheState.asset_id)
            .where(Asset.hash.is_(None), path_filter)
            .order_by(AssetCacheState.asset_id.asc(), AssetCacheState.id.asc())
            .distinct(AssetCacheState.asset_id)
        )
    else:
        first_id = sa.func.min(AssetCacheState.id).label("first_id")
        stmt = (
            sa.select(first_id)
            .join(Asset, Asset.id == AssetCacheState.asset_id)
            .where(Asset.hash.is_(None), path_filter)
            .group_by(AssetCacheState.asset_id)
            .order_by(first_id.asc())
        )
    return [int(x) for x in (await session.execute(stmt)).scalars().all()]


async def list_verify_candidates_under_prefixes(
    session: AsyncSession, *, prefixes: Sequence[str]
) -> Union[list[int], Sequence[int]]:
    if not prefixes:
        return []
    conds = []
    for p in prefixes:
        base = os.path.abspath(p)
        if not base.endswith(os.sep):
            base += os.sep
        escaped, esc = escape_like_prefix(base)
        conds.append(AssetCacheState.file_path.like(escaped + "%", escape=esc))

    return (
        await session.execute(
            sa.select(AssetCacheState.id)
            .where(AssetCacheState.needs_verify.is_(True))
            .where(sa.or_(*conds))
            .order_by(AssetCacheState.id.asc())
        )
    ).scalars().all()


async def ingest_fs_asset(
    session: AsyncSession,
    *,
    asset_hash: str,
    abs_path: str,
    size_bytes: int,
    mtime_ns: int,
    mime_type: Optional[str] = None,
    info_name: Optional[str] = None,
    owner_id: str = "",
    preview_id: Optional[str] = None,
    user_metadata: Optional[dict] = None,
    tags: Sequence[str] = (),
    tag_origin: str = "manual",
    require_existing_tags: bool = False,
) -> dict:
    """
    Idempotently upsert:
      - Asset by content hash (create if missing)
      - AssetCacheState(file_path) pointing to asset_id
      - Optionally AssetInfo + tag links and metadata projection
    Returns flags and ids.
    """
    locator = os.path.abspath(abs_path)
    now = utcnow()
    dialect = session.bind.dialect.name

    if preview_id:
        if not await session.get(Asset, preview_id):
            preview_id = None

    out: dict[str, Any] = {
        "asset_created": False,
        "asset_updated": False,
        "state_created": False,
        "state_updated": False,
        "asset_info_id": None,
    }

    # 1) Asset by hash
    asset = (
        await session.execute(select(Asset).where(Asset.hash == asset_hash).limit(1))
    ).scalars().first()
    if not asset:
        vals = {
            "hash": asset_hash,
            "size_bytes": int(size_bytes),
            "mime_type": mime_type,
            "created_at": now,
        }
        if dialect == "sqlite":
            res = await session.execute(
                d_sqlite.insert(Asset)
                .values(**vals)
                .on_conflict_do_nothing(index_elements=[Asset.hash])
            )
            if int(res.rowcount or 0) > 0:
                out["asset_created"] = True
            asset = (
                await session.execute(
                    select(Asset).where(Asset.hash == asset_hash).limit(1)
                )
            ).scalars().first()
        elif dialect == "postgresql":
            res = await session.execute(
                d_pg.insert(Asset)
                .values(**vals)
                .on_conflict_do_nothing(
                    index_elements=[Asset.hash],
                    index_where=Asset.__table__.c.hash.isnot(None),
                )
                .returning(Asset.id)
            )
            inserted_id = res.scalar_one_or_none()
            if inserted_id:
                out["asset_created"] = True
                asset = await session.get(Asset, inserted_id)
            else:
                asset = (
                    await session.execute(
                        select(Asset).where(Asset.hash == asset_hash).limit(1)
                    )
                ).scalars().first()
        else:
            raise NotImplementedError(f"Unsupported database dialect: {dialect}")
        if not asset:
            raise RuntimeError("Asset row not found after upsert.")
    else:
        changed = False
        if asset.size_bytes != int(size_bytes) and int(size_bytes) > 0:
            asset.size_bytes = int(size_bytes)
            changed = True
        if mime_type and asset.mime_type != mime_type:
            asset.mime_type = mime_type
            changed = True
        if changed:
            out["asset_updated"] = True

    # 2) AssetCacheState upsert by file_path (unique)
    vals = {
        "asset_id": asset.id,
        "file_path": locator,
        "mtime_ns": int(mtime_ns),
    }
    if dialect == "sqlite":
        ins = (
            d_sqlite.insert(AssetCacheState)
            .values(**vals)
            .on_conflict_do_nothing(index_elements=[AssetCacheState.file_path])
        )
    elif dialect == "postgresql":
        ins = (
            d_pg.insert(AssetCacheState)
            .values(**vals)
            .on_conflict_do_nothing(index_elements=[AssetCacheState.file_path])
        )
    else:
        raise NotImplementedError(f"Unsupported database dialect: {dialect}")

    res = await session.execute(ins)
    if int(res.rowcount or 0) > 0:
        out["state_created"] = True
    else:
        upd = (
            sa.update(AssetCacheState)
            .where(AssetCacheState.file_path == locator)
            .where(
                sa.or_(
                    AssetCacheState.asset_id != asset.id,
                    AssetCacheState.mtime_ns.is_(None),
                    AssetCacheState.mtime_ns != int(mtime_ns),
                )
            )
            .values(asset_id=asset.id, mtime_ns=int(mtime_ns))
        )
        res2 = await session.execute(upd)
        if int(res2.rowcount or 0) > 0:
            out["state_updated"] = True

    # 3) Optional AssetInfo + tags + metadata
    if info_name:
        try:
            async with session.begin_nested():
                info = AssetInfo(
                    owner_id=owner_id,
                    name=info_name,
                    asset_id=asset.id,
                    preview_id=preview_id,
                    created_at=now,
                    updated_at=now,
                    last_access_time=now,
                )
                session.add(info)
                await session.flush()
                out["asset_info_id"] = info.id
        except IntegrityError:
            pass

        existing_info = (
            await session.execute(
                select(AssetInfo)
                .where(
                    AssetInfo.asset_id == asset.id,
                    AssetInfo.name == info_name,
                    (AssetInfo.owner_id == owner_id),
                )
                .limit(1)
            )
        ).unique().scalar_one_or_none()
        if not existing_info:
            raise RuntimeError("Failed to update or insert AssetInfo.")

        if preview_id and existing_info.preview_id != preview_id:
            existing_info.preview_id = preview_id

        existing_info.updated_at = now
        if existing_info.last_access_time < now:
            existing_info.last_access_time = now
        await session.flush()
        out["asset_info_id"] = existing_info.id

        norm = [t.strip().lower() for t in (tags or []) if (t or "").strip()]
        if norm and out["asset_info_id"] is not None:
            if not require_existing_tags:
                await ensure_tags_exist(session, norm, tag_type="user")

            existing_tag_names = set(
                name for (name,) in (await session.execute(select(Tag.name).where(Tag.name.in_(norm)))).all()
            )
            missing = [t for t in norm if t not in existing_tag_names]
            if missing and require_existing_tags:
                raise ValueError(f"Unknown tags: {missing}")

            existing_links = set(
                tag_name
                for (tag_name,) in (
                    await session.execute(
                        select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == out["asset_info_id"])
                    )
                ).all()
            )
            to_add = [t for t in norm if t in existing_tag_names and t not in existing_links]
            if to_add:
                session.add_all(
                    [
                        AssetInfoTag(
                            asset_info_id=out["asset_info_id"],
                            tag_name=t,
                            origin=tag_origin,
                            added_at=now,
                        )
                        for t in to_add
                    ]
                )
                await session.flush()

        # metadata["filename"] hack
        if out["asset_info_id"] is not None:
            primary_path = pick_best_live_path(await list_cache_states_by_asset_id(session, asset_id=asset.id))
            computed_filename = compute_relative_filename(primary_path) if primary_path else None

            current_meta = existing_info.user_metadata or {}
            new_meta = dict(current_meta)
            if user_metadata is not None:
                for k, v in user_metadata.items():
                    new_meta[k] = v
            if computed_filename:
                new_meta["filename"] = computed_filename

            if new_meta != current_meta:
                await replace_asset_info_metadata_projection(
                    session,
                    asset_info_id=out["asset_info_id"],
                    user_metadata=new_meta,
                )

    try:
        await remove_missing_tag_for_asset_id(session, asset_id=asset.id)
    except Exception:
        logging.exception("Failed to clear 'missing' tag for asset %s", asset.id)
    return out


async def touch_asset_infos_by_fs_path(
    session: AsyncSession,
    *,
    file_path: str,
    ts: Optional[datetime] = None,
    only_if_newer: bool = True,
) -> int:
    locator = os.path.abspath(file_path)
    ts = ts or utcnow()

    stmt = sa.update(AssetInfo).where(
        sa.exists(
            sa.select(sa.literal(1))
            .select_from(AssetCacheState)
            .where(
                AssetCacheState.asset_id == AssetInfo.asset_id,
                AssetCacheState.file_path == locator,
            )
        )
    )

    if only_if_newer:
        stmt = stmt.where(
            sa.or_(
                AssetInfo.last_access_time.is_(None),
                AssetInfo.last_access_time < ts,
            )
        )

    stmt = stmt.values(last_access_time=ts)

    res = await session.execute(stmt)
    return int(res.rowcount or 0)


async def list_cache_states_with_asset_under_prefixes(
    session: AsyncSession,
    *,
    prefixes: Sequence[str],
) -> list[tuple[AssetCacheState, Optional[str], int]]:
    """Return (AssetCacheState, asset_hash, size_bytes) for rows under any prefix."""
    if not prefixes:
        return []

    conds = []
    for p in prefixes:
        if not p:
            continue
        base = os.path.abspath(p)
        if not base.endswith(os.sep):
            base = base + os.sep
        escaped, esc = escape_like_prefix(base)
        conds.append(AssetCacheState.file_path.like(escaped + "%", escape=esc))

    if not conds:
        return []

    rows = (
        await session.execute(
            select(AssetCacheState, Asset.hash, Asset.size_bytes)
            .join(Asset, Asset.id == AssetCacheState.asset_id)
            .where(sa.or_(*conds))
            .order_by(AssetCacheState.id.asc())
        )
    ).all()
    return [(r[0], r[1], int(r[2] or 0)) for r in rows]


async def _recompute_and_apply_filename_for_asset(session: AsyncSession, *, asset_id: str) -> None:
    """Compute filename from the first *existing* cache state path and apply it to all AssetInfo (if changed)."""
    try:
        primary_path = pick_best_live_path(await list_cache_states_by_asset_id(session, asset_id=asset_id))
        if not primary_path:
            return
        new_filename = compute_relative_filename(primary_path)
        if not new_filename:
            return
        infos = (
            await session.execute(select(AssetInfo).where(AssetInfo.asset_id == asset_id))
        ).scalars().all()
        for info in infos:
            current_meta = info.user_metadata or {}
            if current_meta.get("filename") == new_filename:
                continue
            updated = dict(current_meta)
            updated["filename"] = new_filename
            await replace_asset_info_metadata_projection(session, asset_info_id=info.id, user_metadata=updated)
    except Exception:
        logging.exception("Failed to recompute filename metadata for asset %s", asset_id)

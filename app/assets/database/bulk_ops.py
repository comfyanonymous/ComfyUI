import os
import uuid
import sqlalchemy
from typing import Iterable
from sqlalchemy.orm import Session
from sqlalchemy.dialects import sqlite

from app.assets.helpers import utcnow
from app.assets.database.models import Asset, AssetCacheState, AssetInfo, AssetInfoTag, AssetInfoMeta

MAX_BIND_PARAMS = 800

def _chunk_rows(rows: list[dict], cols_per_row: int, max_bind_params: int) -> Iterable[list[dict]]:
    if not rows:
        return []
    rows_per_stmt = max(1, max_bind_params // max(1, cols_per_row))
    for i in range(0, len(rows), rows_per_stmt):
        yield rows[i:i + rows_per_stmt]

def _iter_chunks(seq, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def _rows_per_stmt(cols: int) -> int:
    return max(1, MAX_BIND_PARAMS // max(1, cols))


def seed_from_paths_batch(
    session: Session,
    *,
    specs: list[dict],
    owner_id: str = "",
) -> dict:
    """Each spec is a dict with keys:
      - abs_path: str
      - size_bytes: int
      - mtime_ns: int
      - info_name: str
      - tags: list[str]
      - fname: Optional[str]
    """
    if not specs:
        return {"inserted_infos": 0, "won_states": 0, "lost_states": 0}

    now = utcnow()
    asset_rows: list[dict] = []
    state_rows: list[dict] = []
    path_to_asset: dict[str, str] = {}
    asset_to_info: dict[str, dict] = {}  # asset_id -> prepared info row
    path_list: list[str] = []

    for sp in specs:
        ap = os.path.abspath(sp["abs_path"])
        aid = str(uuid.uuid4())
        iid = str(uuid.uuid4())
        path_list.append(ap)
        path_to_asset[ap] = aid

        asset_rows.append(
            {
                "id": aid,
                "hash": None,
                "size_bytes": sp["size_bytes"],
                "mime_type": None,
                "created_at": now,
            }
        )
        state_rows.append(
            {
                "asset_id": aid,
                "file_path": ap,
                "mtime_ns": sp["mtime_ns"],
            }
        )
        asset_to_info[aid] = {
            "id": iid,
            "owner_id": owner_id,
            "name": sp["info_name"],
            "asset_id": aid,
            "preview_id": None,
            "user_metadata": {"filename": sp["fname"]} if sp["fname"] else None,
            "created_at": now,
            "updated_at": now,
            "last_access_time": now,
            "_tags": sp["tags"],
            "_filename": sp["fname"],
        }

    # insert all seed Assets (hash=NULL)
    ins_asset = sqlite.insert(Asset)
    for chunk in _iter_chunks(asset_rows, _rows_per_stmt(5)):
        session.execute(ins_asset, chunk)

    # try to claim AssetCacheState (file_path)
    # Insert with ON CONFLICT DO NOTHING, then query to find which paths were actually inserted
    ins_state = (
        sqlite.insert(AssetCacheState)
        .on_conflict_do_nothing(index_elements=[AssetCacheState.file_path])
    )
    for chunk in _iter_chunks(state_rows, _rows_per_stmt(3)):
        session.execute(ins_state, chunk)

    # Query to find which of our paths won (were actually inserted)
    winners_by_path: set[str] = set()
    for chunk in _iter_chunks(path_list, MAX_BIND_PARAMS):
        result = session.execute(
            sqlalchemy.select(AssetCacheState.file_path)
            .where(AssetCacheState.file_path.in_(chunk))
            .where(AssetCacheState.asset_id.in_([path_to_asset[p] for p in chunk]))
        )
        winners_by_path.update(result.scalars().all())

    all_paths_set = set(path_list)
    losers_by_path = all_paths_set - winners_by_path
    lost_assets = [path_to_asset[p] for p in losers_by_path]
    if lost_assets:  # losers get their Asset removed
        for id_chunk in _iter_chunks(lost_assets, MAX_BIND_PARAMS):
            session.execute(sqlalchemy.delete(Asset).where(Asset.id.in_(id_chunk)))

    if not winners_by_path:
        return {"inserted_infos": 0, "won_states": 0, "lost_states": len(losers_by_path)}

    # insert AssetInfo only for winners
    # Insert with ON CONFLICT DO NOTHING, then query to find which were actually inserted
    winner_info_rows = [asset_to_info[path_to_asset[p]] for p in winners_by_path]
    ins_info = (
        sqlite.insert(AssetInfo)
        .on_conflict_do_nothing(index_elements=[AssetInfo.asset_id, AssetInfo.owner_id, AssetInfo.name])
    )
    for chunk in _iter_chunks(winner_info_rows, _rows_per_stmt(9)):
        session.execute(ins_info, chunk)

    # Query to find which info rows were actually inserted (by matching our generated IDs)
    all_info_ids = [row["id"] for row in winner_info_rows]
    inserted_info_ids: set[str] = set()
    for chunk in _iter_chunks(all_info_ids, MAX_BIND_PARAMS):
        result = session.execute(
            sqlalchemy.select(AssetInfo.id).where(AssetInfo.id.in_(chunk))
        )
        inserted_info_ids.update(result.scalars().all())

    # build and insert tag + meta rows for the AssetInfo
    tag_rows: list[dict] = []
    meta_rows: list[dict] = []
    if inserted_info_ids:
        for row in winner_info_rows:
            iid = row["id"]
            if iid not in inserted_info_ids:
                continue
            for t in row["_tags"]:
                tag_rows.append({
                    "asset_info_id": iid,
                    "tag_name": t,
                    "origin": "automatic",
                    "added_at": now,
                })
            if row["_filename"]:
                meta_rows.append(
                    {
                        "asset_info_id": iid,
                        "key": "filename",
                        "ordinal": 0,
                        "val_str": row["_filename"],
                        "val_num": None,
                        "val_bool": None,
                        "val_json": None,
                    }
                )

    bulk_insert_tags_and_meta(session, tag_rows=tag_rows, meta_rows=meta_rows, max_bind_params=MAX_BIND_PARAMS)
    return {
        "inserted_infos": len(inserted_info_ids),
        "won_states": len(winners_by_path),
        "lost_states": len(losers_by_path),
    }


def bulk_insert_tags_and_meta(
    session: Session,
    *,
    tag_rows: list[dict],
    meta_rows: list[dict],
    max_bind_params: int,
) -> None:
    """Batch insert into asset_info_tags and asset_info_meta with ON CONFLICT DO NOTHING.
    - tag_rows keys: asset_info_id, tag_name, origin, added_at
    - meta_rows keys: asset_info_id, key, ordinal, val_str, val_num, val_bool, val_json
    """
    if tag_rows:
        ins_links = (
            sqlite.insert(AssetInfoTag)
            .on_conflict_do_nothing(index_elements=[AssetInfoTag.asset_info_id, AssetInfoTag.tag_name])
        )
        for chunk in _chunk_rows(tag_rows, cols_per_row=4, max_bind_params=max_bind_params):
            session.execute(ins_links, chunk)
    if meta_rows:
        ins_meta = (
            sqlite.insert(AssetInfoMeta)
            .on_conflict_do_nothing(
                index_elements=[AssetInfoMeta.asset_info_id, AssetInfoMeta.key, AssetInfoMeta.ordinal]
            )
        )
        for chunk in _chunk_rows(meta_rows, cols_per_row=7, max_bind_params=max_bind_params):
            session.execute(ins_meta, chunk)

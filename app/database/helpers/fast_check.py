import os
from typing import Optional


def fast_asset_file_check(
    *,
    mtime_db: Optional[int],
    size_db: Optional[int],
    stat_result: os.stat_result,
) -> bool:
    if mtime_db is None:
        return False
    actual_mtime_ns = getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000))
    if int(mtime_db) != int(actual_mtime_ns):
        return False
    sz = int(size_db or 0)
    if sz > 0:
        return int(stat_result.st_size) == sz
    return True

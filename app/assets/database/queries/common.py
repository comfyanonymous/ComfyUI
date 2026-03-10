"""Shared utilities for database query modules."""

import os
from typing import Iterable

import sqlalchemy as sa

from app.assets.database.models import AssetReference
from app.assets.helpers import escape_sql_like_string

MAX_BIND_PARAMS = 800


def calculate_rows_per_statement(cols: int) -> int:
    """Calculate how many rows can fit in one statement given column count."""
    return max(1, MAX_BIND_PARAMS // max(1, cols))


def iter_chunks(seq, n: int):
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def iter_row_chunks(rows: list[dict], cols_per_row: int) -> Iterable[list[dict]]:
    """Yield chunks of rows sized to fit within bind param limits."""
    if not rows:
        return
    yield from iter_chunks(rows, calculate_rows_per_statement(cols_per_row))


def build_visible_owner_clause(owner_id: str) -> sa.sql.ClauseElement:
    """Build owner visibility predicate for reads.

    Owner-less rows are visible to everyone.
    """
    owner_id = (owner_id or "").strip()
    if owner_id == "":
        return AssetReference.owner_id == ""
    return AssetReference.owner_id.in_(["", owner_id])


def build_prefix_like_conditions(
    prefixes: list[str],
) -> list[sa.sql.ColumnElement]:
    """Build LIKE conditions for matching file paths under directory prefixes."""
    conds = []
    for p in prefixes:
        base = os.path.abspath(p)
        if not base.endswith(os.sep):
            base += os.sep
        escaped, esc = escape_sql_like_string(base)
        conds.append(AssetReference.file_path.like(escaped + "%", escape=esc))
    return conds

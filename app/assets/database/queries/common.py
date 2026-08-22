"""Shared utilities for database query modules (B-schema stub)."""


MAX_BIND_PARAMS = 800


def calculate_rows_per_statement(cols: int) -> int:
    return max(1, MAX_BIND_PARAMS // max(1, cols))


def iter_chunks(seq, n: int):
    for index in range(0, len(seq), n):
        yield seq[index : index + n]


def iter_row_chunks(rows, cols_per_row: int):
    yield from iter_chunks(rows, calculate_rows_per_statement(cols_per_row))


def build_visibility_clause(*_args, **_kwargs):
    raise NotImplementedError("Asset-reference queries were removed in the B schema")


def build_prefix_like_conditions(*_args, **_kwargs):
    raise NotImplementedError("Asset-reference queries were removed in the B schema")


def apply_tag_filters(*_args, **_kwargs):
    raise NotImplementedError("Asset-reference queries were removed in the B schema")


def apply_metadata_filter(*_args, **_kwargs):
    raise NotImplementedError("Asset-reference queries were removed in the B schema")

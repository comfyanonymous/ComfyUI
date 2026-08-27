"""Shared case table for the SQL path-prefix predicate equivalence tests.

Both SQL prefix sites (``lifecycle.wipe_temp_db_rows`` and
``scanner.get_unenriched_assets_for_roots``) must reproduce
``scanner_changes.is_path_under_prefixes`` exactly. Rather than hand-pick
assertions per site, each test seeds this table and asserts the SQL result set
is identical to the Python predicate evaluated over the same paths.

The predicate normalizes only its *prefix*; the column is compared raw. That is
sound only because ``records.create_content`` normalizes every path it stores,
so tests compare against ``stored_path(case)`` — never the raw input.
"""

from __future__ import annotations

import os


def stored_path(path: str) -> str:
    """The path ``records.create_content`` actually stores for ``path``."""
    return os.path.abspath(path)


def prefix_case_paths(root: str) -> list[str]:
    """Every way a path handed to the write boundary can relate to ``root``.

    ``root``'s basename must have a distinct uppercase form — callers own the
    last component precisely so the case-difference case cannot degenerate.

    The last five entries are non-normalized inputs. Raw, they disagree with
    ``is_path_under_prefixes``: ``<root>/../escaped.png`` shares the ``<root>/``
    character prefix while resolving outside the root, so the SQL predicate
    admitted it and ``wipe_temp_db_rows`` hard-deleted an out-of-root row.
    """
    parent = os.path.dirname(root)
    name = os.path.basename(root)
    upper = name.upper()
    assert upper != name, f"root basename {name!r} has no distinct uppercase form"

    return [
        root,                                            # exact root
        os.path.join(root, "child.png"),                 # direct child
        os.path.join(root, "sub", "deep.png"),           # nested child
        os.path.join(root, "meta%_*?[].png"),            # child holding metacharacters
        os.path.join(parent, upper),                     # case-different root
        os.path.join(parent, upper, "case.png"),         # case-different child
        root + "-other" + os.sep + "lexical.png",        # lexical sibling
        root + "extra",                                  # lexical sibling, no separator
        os.path.join(parent, "unrelated.png"),           # outside entirely
        os.path.join(root, os.pardir, "escaped.png"),    # `..` resolving outside
        os.path.join(root, "sub", os.pardir, "in.png"),  # `..` resolving inside
        os.path.join(name, "relative.png"),              # relative to the cwd
        root + os.sep + os.sep + "doubled.png",          # repeated separators
        os.path.join(root, "trailing.png") + os.sep,     # trailing separator
    ]

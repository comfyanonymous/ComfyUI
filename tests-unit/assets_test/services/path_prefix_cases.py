"""Shared case table for the SQL path-prefix predicate equivalence tests.

Both SQL prefix sites (``lifecycle.wipe_temp_db_rows`` and
``scanner.get_unenriched_assets_for_roots``) must reproduce
``scanner_changes.is_path_under_prefixes`` exactly. Rather than hand-pick
assertions per site, each test seeds this table and asserts the SQL result set
is identical to the Python predicate evaluated over the same paths.
"""

from __future__ import annotations

import os


def prefix_case_paths(root: str) -> list[str]:
    """Every way a stored path can relate to ``root``, matching and not.

    ``root``'s basename must have a distinct uppercase form — callers own the
    last component precisely so the case-difference case cannot degenerate.
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
    ]

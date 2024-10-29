from __future__ import annotations

from pathlib import PurePosixPath, Path, PosixPath


def construct_path(*args) -> PurePosixPath | Path:
    if len(args) > 0 and args[0] is not None and isinstance(args[0], str) and args[0].startswith("/"):
        try:
            return PosixPath(*args)
        except Exception:
            return PurePosixPath(*args)
    else:
        return Path(*args)

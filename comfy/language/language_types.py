from __future__ import annotations

from typing import NamedTuple, Dict, Any


class ProcArgsRes(NamedTuple):
    seed: int
    generate_kwargs: Dict[str, Any]

from __future__ import annotations

import hashlib
import json
from typing_extensions import Buffer


def digest(data: dict | str | Buffer) -> str:
    hash_object = hashlib.sha256()
    if isinstance(data, Buffer):
        hash_object.update(data)
    else:
        if isinstance(data, str):
            json_str = data
        elif isinstance(data, dict):
            json_str = json.dumps(data, separators=(',', ':'))
        else:
            raise RuntimeError("invalid data type")
        hash_object.update(json_str.encode())
    return hash_object.hexdigest()

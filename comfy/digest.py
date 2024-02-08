from __future__ import annotations

import hashlib
import json


def digest(data: dict | str) -> str:
    json_str = data if isinstance(data, str) else json.dumps(data, separators=(',', ':'))
    hash_object = hashlib.sha256()
    hash_object.update(json_str.encode())
    return hash_object.hexdigest()

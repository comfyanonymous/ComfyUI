from typing import Any
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

def to_dict(obj: Any, include_none: bool = False) -> dict[str, Any]:
    fields = obj.__table__.columns.keys()
    out: dict[str, Any] = {}
    for field in fields:
        val = getattr(obj, field)
        if val is None and not include_none:
            continue
        if isinstance(val, datetime):
            out[field] = val.isoformat()
        else:
            out[field] = val
    return out

# TODO: Define models here

from __future__ import annotations

from typing import Mapping, Any


def _make_mutable(obj: Any) -> Any:
    if isinstance(obj, Mapping) and not isinstance(obj, dict) and not hasattr(obj, "__setitem__"):
        obj = dict(obj)
        for key, value in obj.items():
            obj[key] = make_mutable(value)
    if isinstance(obj, tuple):
        obj = list(obj)
        for i in range(len(obj)):
            obj[i] = make_mutable(obj[i])
    if isinstance(obj, frozenset):
        obj = set([make_mutable(x) for x in obj])
    return obj


def make_mutable(obj: Mapping) -> dict:
    """
    Makes a copy of an immutable dict into a mutable dict.

    If the object is already a mutable type or a value type like a string or integer, returns the value.

    Returns dict, set or tuple depending on its input, but you should not use it this way.

    :param obj: any object
    :return:
    """
    return _make_mutable(obj)

from typing import Mapping, Any


def make_mutable(obj: Any) -> dict:
    """
    Makes an immutable dict, frozenset or tuple mutable. Otherwise, returns the value.
    :param obj: any object
    :return:
    """
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

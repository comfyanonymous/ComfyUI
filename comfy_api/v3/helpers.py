from typing import Callable, Optional


def first_real_override(cls: type, name: str, *, base: type) -> Optional[Callable]:
    """Return the *callable* override of `name` visible on `cls`, or None if every
    implementation up to (and including) `base` is the placeholder defined on `base`.
    """
    base_attr = getattr(base, name, None)
    if base_attr is None:
        return None
    base_func = base_attr.__func__
    for c in cls.mro():                       # NodeB, NodeA, ComfyNodeV3, object …
        if c is base:                         # reached the placeholder – we're done
            break
        if name in c.__dict__:                # first class that *defines* the attr
            func = getattr(c, name).__func__
            if func is not base_func:         # real override
                return getattr(cls, name)     # bound to *cls*
    return None

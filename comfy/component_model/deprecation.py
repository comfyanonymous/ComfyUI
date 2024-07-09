from __future__ import annotations

import warnings
from functools import wraps
from typing import Optional


def _deprecate_method(*, version: str, message: Optional[str] = None):
    """Decorator to issue warnings when using a deprecated method.

    Args:
        version (`str`):
            The version when deprecated arguments will result in error.
        message (`str`, *optional*):
            Warning message that is raised. If not passed, a default warning message
            will be created.
    """

    def _inner_deprecate_method(f):
        name = f.__name__
        if name == "__init__":
            name = f.__qualname__.split(".")[0]  # class name instead of method name

        @wraps(f)
        def inner_f(*args, **kwargs):
            warning_message = (
                f"'{name}' (from '{f.__module__}') is deprecated and will be removed from version '{version}'."
            )
            if message is not None:
                warning_message += " " + message
            warnings.warn(warning_message, FutureWarning)
            return f(*args, **kwargs)

        return inner_f

    return _inner_deprecate_method

import os.path
import pathlib

import requests_cache
from contextlib import contextmanager


@contextmanager
def use_requests_caching(
        cache_name='http_cache',
        cache_control=True,
        **kwargs
):
    """
    A context manager to globally patch 'requests' with 'requests-cache'
    for all code executed within its scope.

    This implementation uses the 'filesystem' backend, which is ideal
    for large file responses.

    By default, it also sets 'use_cache_dir=True'. This automatically
    tells requests-cache to store its cache files in the standard
    user cache directory for your operating system.

    - On Linux, this respects the $XDG_CACHE_HOME environment variable.
    - On macOS, it uses ~/Library/Caches/<cache_name>/
    - On Windows, it uses %LOCALAPPDATA%\\<cache_name>\\Cache

    You do not need to populate a directory variable; this parameter
    handles it for you.
    """

    kwargs['backend'] = 'filesystem'
    path_provided = isinstance(cache_name, pathlib.PurePath) or os.path.sep in str(cache_name) or '.' == str(cache_name)[0]
    kwargs.setdefault('use_cache_dir', not path_provided)
    kwargs.setdefault('cache_control', cache_control)

    with requests_cache.enabled(cache_name, **kwargs):
        yield

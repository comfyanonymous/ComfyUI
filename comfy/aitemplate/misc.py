#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
miscellaneous utilities
"""
import hashlib
import logging
import os
import platform


def is_debug():
    logger = logging.getLogger("aitemplate")
    return logger.level == logging.DEBUG


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_windows() -> bool:
    return os.name == "nt"


def setup_logger(name):
    root_logger = logging.getLogger(name)
    info_handle = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s <%(name)s> %(message)s")
    info_handle.setFormatter(formatter)
    root_logger.addHandler(info_handle)
    root_logger.propagate = False

    DEFAULT_LOGLEVEL = logging.getLogger().level
    log_level_str = os.environ.get("LOGLEVEL", None)
    LOG_LEVEL = (
        getattr(logging, log_level_str.upper())
        if log_level_str is not None
        else DEFAULT_LOGLEVEL
    )
    root_logger.setLevel(LOG_LEVEL)
    return root_logger


def short_str(s, length=8) -> str:
    """
    Returns a hashed string, somewhat similar to URL shortener.
    """
    hash_str = hashlib.sha256(s.encode()).hexdigest()
    return hash_str[0:length]


def callstack_stats(enable=False):
    if enable:

        def decorator(f):
            import cProfile
            import io
            import pstats

            logger = logging.getLogger(__name__)

            def inner_function(*args, **kwargs):
                pr = cProfile.Profile()
                pr.enable()
                result = f(*args, **kwargs)
                pr.disable()
                s = io.StringIO()
                pstats.Stats(pr, stream=s).sort_stats(
                    pstats.SortKey.CUMULATIVE
                ).print_stats(30)
                logger.debug(s.getvalue())
                return result

            return inner_function

        return decorator
    else:

        def decorator(f):
            def inner_function(*args, **kwargs):
                return f(*args, **kwargs)

            return inner_function

        return decorator

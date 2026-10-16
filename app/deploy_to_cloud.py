import logging
import os

import folder_paths

HEADLESS_MARKER = "used_headlessly"


def headless_marker_path() -> str:
    return os.path.join(folder_paths.get_user_directory(), HEADLESS_MARKER)


def used_headlessly() -> bool:
    return os.path.exists(headless_marker_path())


def mark_used_headlessly() -> None:
    """Record, once and permanently, that a prompt was queued without a client_id."""
    path = headless_marker_path()
    if os.path.exists(path):
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "a").close()
    except OSError as e:
        logging.warning("Could not write headless marker %s: %s", path, e)

from __future__ import annotations

import logging
import os
import platform
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Optional

import filelock
import huggingface_hub
from huggingface_hub import hf_hub_download
from huggingface_hub import logging as hf_logging

hf_logging.set_verbosity_debug()
from pebble import ThreadPool

from .tqdm_watcher import TqdmWatcher

logger = logging.getLogger(__name__)

_VAR = "HF_HUB_ENABLE_HF_TRANSFER"
_XET_VAR = "HF_XET_HIGH_PERFORMANCE"


if platform.system() == "Windows":
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    logger.debug("Xet was disabled since it is currently not reliable")
    os.environ[_VAR] = "True"
else:
    os.environ[_XET_VAR] = "True"



def hf_hub_download_with_disable_fast(repo_id=None, filename=None, disable_fast=None, hf_env: dict[str, str] = None, **kwargs):
    for k, v in hf_env.items():
        os.environ[k] = v
    if disable_fast:
        if _VAR == _XET_VAR:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
        else:
            os.environ[_VAR] = "False"
    return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)


def hf_hub_download_with_retries(repo_id: str, filename: str, watcher: Optional[TqdmWatcher] = None, retries=2, stall_timeout=10, **kwargs):
    """
    Wraps hf_hub_download with stall detection and retries using a TqdmWatcher.
    Includes a monkey-patch for filelock to release locks from stalled downloads.
    """
    if watcher is None:
        logger.warning(f"called _hf_hub_download_with_retries without progress to watch")
        return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)

    xet_available = huggingface_hub.file_download.is_xet_available()
    hf_hub_disable_xet_prev_value = os.getenv("HF_HUB_DISABLE_XET")
    disable_fast = hf_hub_disable_xet_prev_value is not None

    instantiated_locks: list[filelock.FileLock] = []
    original_filelock_init = filelock.FileLock.__init__

    def new_filelock_init(self, *args, **kwargs):
        """A wrapper around FileLock.__init__ to capture lock instances."""
        original_filelock_init(self, *args, **kwargs)
        instantiated_locks.append(self)

    filelock.FileLock.__init__ = new_filelock_init

    try:
        with ThreadPool(max_workers=retries + 1) as executor:
            for attempt in range(retries):
                watcher.tick()
                hf_env = {k: v for k, v in os.environ.items() if k.upper().startswith("HF_")}

                if len(instantiated_locks) > 0:
                    logger.debug(f"Attempting to unlock {len(instantiated_locks)} captured file locks.")
                for lock in instantiated_locks:
                    path = lock.lock_file
                    if lock.is_locked:
                        lock.release(force=True)
                    else:
                        # something else went wrong
                        try:
                            lock._release()
                        except (AttributeError, TypeError):
                            pass
                    try:
                        Path(path).unlink(missing_ok=True)
                    except OSError:
                        # todo: obviously the process is holding this lock
                        pass
                    logger.debug(f"Released stalled lock: {lock.lock_file}")
                instantiated_locks.clear()
                future: Future[str] = executor.submit(hf_hub_download_with_disable_fast, repo_id=repo_id, filename=filename, disable_fast=disable_fast, hf_env=hf_env, **kwargs)

                try:
                    while not future.done():
                        if time.monotonic() - watcher.last_update_time > stall_timeout:
                            msg = f"Download of '{repo_id}/{filename}' stalled for >{stall_timeout}s. Retrying... (Attempt {attempt + 1}/{retries})"
                            if xet_available:
                                logger.warning(f"{msg}. Disabling xet for our retry.")
                                disable_fast = True
                            else:
                                logger.warning(msg)

                            future.cancel()  # Cancel the stalled future
                            break

                        time.sleep(0.5)

                    if future.done() and not future.cancelled():
                        return future.result()

                except Exception as e:
                    logger.error(f"Exception during download attempt {attempt + 1}: {e}", exc_info=True)

        raise RuntimeError(f"Failed to download '{repo_id}/{filename}' after {retries} attempts.")
    finally:
        filelock.FileLock.__init__ = original_filelock_init

        if hf_hub_disable_xet_prev_value is not None:
            os.environ["HF_HUB_DISABLE_XET"] = hf_hub_disable_xet_prev_value

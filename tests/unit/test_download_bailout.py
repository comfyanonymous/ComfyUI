import threading
import time

import pytest

from comfy.component_model.hf_hub_download_with_disable_xet import hf_hub_download_with_retries
from comfy.component_model.tqdm_watcher import TqdmWatcher

download_method_name = "comfy.component_model.hf_hub_download_with_disable_xet.hf_hub_download_with_disable_fast"

def mock_stalled_download(*args, **kwargs):
    """A mock for hf_hub_download that simulates a stall by sleeping indefinitely."""
    time.sleep(10)
    return "this_path_should_never_be_returned"


def test_download_stalls_and_fails(monkeypatch):
    """
    Verify that a stalled download triggers retries and eventually fails with an RuntimeError.
    """

    monkeypatch.setattr(download_method_name, mock_stalled_download)
    watcher = TqdmWatcher()
    repo_id = "test/repo-stall"
    filename = "stalled_file.safetensors"

    with pytest.raises(RuntimeError) as excinfo:
        hf_hub_download_with_retries(
            repo_id=repo_id,
            filename=filename,
            watcher=watcher,
            stall_timeout=0.2,
            retries=2,
        )

    assert f"Failed to download '{repo_id}/{filename}' after 2 attempts" in str(excinfo.value)


def mock_successful_slow_download(*args, **kwargs):
    """A mock for a download that is slow but not stalled."""
    time.sleep(1)

    return "expected/successful/path"


def _keep_watcher_alive(watcher: TqdmWatcher, stop_event: threading.Event):
    """Helper function to run in a thread and periodically tick the watcher."""
    while not stop_event.is_set():
        watcher.tick()
        time.sleep(0.1)


def test_download_progresses_and_succeeds(monkeypatch):
    """
    Verify that a download with periodic progress updates completes successfully.
    """
    monkeypatch.setattr(download_method_name, mock_successful_slow_download)

    watcher = TqdmWatcher()
    stop_event = threading.Event()
    ticker_thread = threading.Thread(
        target=_keep_watcher_alive,
        args=(watcher, stop_event),
        daemon=True

    )
    ticker_thread.start()

    try:
        result = hf_hub_download_with_retries(
            repo_id="test/repo-success",
            filename="good_file.safetensors",
            stall_timeout=0.3,
            watcher=watcher
        )
        assert result == "expected/successful/path"
    finally:
        stop_event.set()
        ticker_thread.join(timeout=1)

"""Regression tests for #15820 — a userdata path on another Windows drive.

`get_request_user_filepath()` joins the requested file onto the user root and
then compares them with `os.path.commonpath()`, which raises

    ValueError: Paths don't have the same drive

when the two paths sit on different Windows drives. Requesting
`/userdata/C:%5CWindows%5Ctemp%5Ctest.txt` from an install whose user
directory is on `D:` therefore produced an unhandled 500 rather than the
intended rejection. Measured on the pre-fix build, user directory on `D:`:

    'C:\\\\Windows\\\\temp\\\\test.txt'      -> ValueError: Paths don't have the same drive
    'C:%5CWindows%5Ctemp%5Ctest.txt'  -> ValueError: Paths don't have the same drive
    '../escape.json'                  -> None      (rejected, as it should be)
    'sub/ok.json'                     -> <path inside the user directory>

The cross-drive case is Windows-only; the tests below simulate the drive
mismatch on every platform by patching `os.path.commonpath` to raise the same
ValueError, and additionally run the real thing on Windows when a second drive
is actually available.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import folder_paths
from app.user_manager import UserManager


@pytest.fixture
def user_manager():
    with tempfile.TemporaryDirectory() as temp_dir:
        original = folder_paths.get_user_directory()
        folder_paths.set_user_directory(temp_dir)
        with patch("app.user_manager.args") as mock_args:
            mock_args.multi_user = False
            manager = UserManager()
            manager.users = {"default": "default"}
            yield manager
        folder_paths.set_user_directory(original)


@pytest.fixture
def request_():
    request = MagicMock()
    request.headers = {}
    return request


def test_cross_drive_path_is_rejected_not_raised(user_manager, request_):
    """A drive mismatch must return None, the same as any other escape."""
    real_commonpath = os.path.commonpath

    def commonpath_raising_on_mismatch(paths):
        first, second = paths
        if os.path.splitdrive(first)[0].lower() != os.path.splitdrive(second)[0].lower():
            raise ValueError("Paths don't have the same drive")
        return real_commonpath(paths)

    with patch("os.path.commonpath", side_effect=commonpath_raising_on_mismatch):
        result = user_manager.get_request_user_filepath(
            request_, "Z:\\Windows\\temp\\test.txt", create_dir=False
        )

    assert result is None


def test_url_encoded_cross_drive_path_is_rejected(user_manager, request_):
    """The reported request shape: the drive letter arrives percent-encoded."""
    real_commonpath = os.path.commonpath

    def commonpath_raising_on_mismatch(paths):
        first, second = paths
        if os.path.splitdrive(first)[0].lower() != os.path.splitdrive(second)[0].lower():
            raise ValueError("Paths don't have the same drive")
        return real_commonpath(paths)

    with patch("os.path.commonpath", side_effect=commonpath_raising_on_mismatch):
        result = user_manager.get_request_user_filepath(
            request_, "Z:%5CWindows%5Ctemp%5Ctest.txt", create_dir=False
        )

    assert result is None


def test_ordinary_paths_are_unaffected(user_manager, request_):
    """The guard must not change the two outcomes that already worked."""
    inside = user_manager.get_request_user_filepath(
        request_, "sub/workflow.json", create_dir=False
    )
    assert inside is not None
    assert inside.endswith(os.path.join("sub", "workflow.json"))

    assert (
        user_manager.get_request_user_filepath(request_, "../escape.json", create_dir=False)
        is None
    )
    assert user_manager.get_request_user_filepath(request_, None, create_dir=False) is not None


@pytest.mark.skipif(sys.platform != "win32", reason="drive letters are Windows-only")
def test_real_cross_drive_path_on_windows(user_manager, request_):
    """Same case again without patching, when a second real drive exists."""
    user_drive = os.path.splitdrive(folder_paths.get_user_directory())[0].upper()
    other = next(
        (
            f"{letter}:"
            for letter in "CDEFGH"
            if f"{letter}:" != user_drive and os.path.exists(f"{letter}:\\")
        ),
        None,
    )
    if other is None:
        pytest.skip("no second drive available on this machine")

    assert (
        user_manager.get_request_user_filepath(
            request_, f"{other}\\Windows\\temp\\test.txt", create_dir=False
        )
        is None
    )

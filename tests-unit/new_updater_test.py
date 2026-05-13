"""Tests for new_updater.

`update_windows_updater()` runs on every startup of the Windows standalone
build and silently rewrites the user's `update/` folder. The signature check
that guards that rewrite used to be a strict `bytes.startswith` against the
legacy updater command, which meant any user-edited bat (e.g. one that picked
up a `@echo off` prefix or a UTF-8 BOM from a text editor) was treated as
"foreign" and never patched again — leaving the portable's updater stuck on
the legacy format. The check is now tolerant of those prefixes, and previously
silenced I/O failures are now surfaced via `logging.warning`.
"""

import logging
import os

import pytest

import new_updater
from new_updater import update_windows_updater


LEGACY_BAT_FIRST_LINE = b"..\\python_embeded\\python.exe .\\update.py ..\\ComfyUI\\\r\n"
DEPS_BAT_LINE = b"..\\python_embeded\\python.exe .\\update.py ..\\ComfyUI\\ --depend\r\n"
NEW_UPDATER_SOURCE = b"# new updater source\n"


@pytest.fixture
def portable_layout(tmp_path, monkeypatch):
    """Lay out the directory structure that `update_windows_updater` expects:

        <top>/ComfyUI/.ci/update_windows/{update.py,update_comfyui.bat}   (sources)
        <top>/update/{update.py,update_comfyui.bat,update_comfyui_and_python_dependencies.bat}  (dest)

    `base_path` is rewritten to point at `<top>/ComfyUI` so `update_windows_updater`
    resolves both source and destination paths inside `tmp_path`.
    """
    top = tmp_path
    comfyui = top / "ComfyUI"
    ci_dir = comfyui / ".ci" / "update_windows"
    update_dir = top / "update"
    ci_dir.mkdir(parents=True)
    update_dir.mkdir(parents=True)

    (ci_dir / "update.py").write_bytes(NEW_UPDATER_SOURCE)
    (ci_dir / "update_comfyui.bat").write_bytes(b"@echo off\r\n" + LEGACY_BAT_FIRST_LINE)

    monkeypatch.setattr(new_updater, "base_path", str(comfyui))
    return {
        "top": top,
        "comfyui": comfyui,
        "ci_dir": ci_dir,
        "update_dir": update_dir,
        "dest_bat": update_dir / "update_comfyui.bat",
        "dest_updater": update_dir / "update.py",
        "dest_deps_bat": update_dir / "update_comfyui_and_python_dependencies.bat",
    }


class TestUpdateWindowsUpdater:
    def test_updates_plain_legacy_bat(self, portable_layout):
        portable_layout["dest_bat"].write_bytes(LEGACY_BAT_FIRST_LINE)
        portable_layout["dest_deps_bat"].write_bytes(DEPS_BAT_LINE)

        update_windows_updater()

        assert portable_layout["dest_updater"].read_bytes() == NEW_UPDATER_SOURCE
        assert portable_layout["dest_deps_bat"].read_bytes() == b"call update_comfyui.bat nopause --depend\r\n"

    def test_updates_legacy_bat_with_echo_off_prefix(self, portable_layout):
        # User's deployed bat picked up an `@echo off` prefix (e.g. from a text
        # editor saving with a different first line). The strict `startswith`
        # check used to skip this file; the new check recognises it and rewrites it.
        portable_layout["dest_bat"].write_bytes(b"@echo off\r\n" + LEGACY_BAT_FIRST_LINE)
        portable_layout["dest_deps_bat"].write_bytes(DEPS_BAT_LINE)

        update_windows_updater()

        assert portable_layout["dest_updater"].read_bytes() == NEW_UPDATER_SOURCE
        assert portable_layout["dest_deps_bat"].read_bytes() == b"call update_comfyui.bat nopause --depend\r\n"

    def test_updates_legacy_bat_with_utf8_bom(self, portable_layout):
        # Text editors on Windows sometimes prepend a UTF-8 BOM when saving the
        # bat file. The signature check must look past it.
        portable_layout["dest_bat"].write_bytes(b"\xef\xbb\xbf" + LEGACY_BAT_FIRST_LINE)

        update_windows_updater()

        assert portable_layout["dest_updater"].exists()

    def test_skips_foreign_bat(self, portable_layout):
        # A bat that doesn't look like our updater (e.g. user fully replaced it)
        # must not be overwritten.
        portable_layout["dest_bat"].write_bytes(b"echo something else\r\n")

        update_windows_updater()

        assert not portable_layout["dest_updater"].exists()
        assert portable_layout["dest_bat"].read_bytes() == b"echo something else\r\n"

    def test_skips_when_dest_bat_missing(self, portable_layout):
        # No dest bat file present → quietly return, do not raise.
        update_windows_updater()
        assert not portable_layout["dest_updater"].exists()

    def test_missing_deps_bat_is_not_fatal(self, portable_layout):
        # Some old portable layouts don't ship the deps bat. Updating the main
        # bat must still succeed.
        portable_layout["dest_bat"].write_bytes(LEGACY_BAT_FIRST_LINE)
        # Deliberately don't create dest_deps_bat

        update_windows_updater()

        assert portable_layout["dest_updater"].exists()

    def test_logs_warning_when_source_updater_missing(self, portable_layout, caplog):
        # The portable was somehow shipped without the canonical update.py source.
        # We must surface that as a logged warning, not swallow it silently like
        # the old bare `except` did.
        portable_layout["dest_bat"].write_bytes(LEGACY_BAT_FIRST_LINE)
        (portable_layout["ci_dir"] / "update.py").unlink()

        with caplog.at_level(logging.WARNING):
            update_windows_updater()

        assert any(
            "Failed to update" in r.message and "update.py" in r.message
            for r in caplog.records
        )
        # Main bat should not have been replaced if the updater copy failed.
        assert portable_layout["dest_bat"].read_bytes() == LEGACY_BAT_FIRST_LINE

    def test_logs_warning_when_dest_bat_read_fails(self, portable_layout, caplog, monkeypatch):
        # Permission errors etc. on the read used to fall through `except: pass`
        # with no signal. They now log.
        portable_layout["dest_bat"].write_bytes(LEGACY_BAT_FIRST_LINE)
        real_open = open

        def fake_open(path, *args, **kwargs):
            if os.path.basename(str(path)) == "update_comfyui.bat" and "rb" in args:
                raise PermissionError("denied")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        with caplog.at_level(logging.WARNING):
            update_windows_updater()

        assert any("Could not read" in r.message for r in caplog.records)

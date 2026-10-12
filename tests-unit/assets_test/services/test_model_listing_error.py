"""A model folder that can't be listed (gone, or raising) is skipped: the asset scan
completes and catalogues everything else. Run through the seeder's real scan loop on an
in-memory catalog."""

import errno
import os
import logging
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session as SASession, sessionmaker

import folder_paths
from app.assets import scanner as scanner_module
from app.assets import seeder as seeder_module
from app.assets.database.models import Asset, AssetContent
from app.assets.event_log import TAG
from app.assets.scanner_admission import _WATCH_LIST
from app.assets.services import file_utils

ALL_ROOTS = ("models", "input", "output")


def events_named(caplog: pytest.LogCaptureFixture, event: str) -> list[dict]:
    prefix = f"{TAG} {event}"
    out = []
    for record in caplog.records:
        message = record.getMessage()
        if message == prefix or message.startswith(prefix + " "):
            pairs = (pair.split("=", 1) for pair in message[len(prefix):].split())
            out.append({k: int(v) if v.isdigit() else v for k, v in pairs})
    return out


@pytest.fixture(autouse=True)
def isolated_state(db_engine):
    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    _WATCH_LIST.clear()
    with patch("app.assets.scanner.create_session", _create_session), \
         patch("app.assets.seeder.create_session", _create_session), \
         patch("app.database.db.WriteSession", sessionmaker(bind=db_engine)):
        yield
    _WATCH_LIST.clear()


@pytest.fixture
def layout(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """checkpoints in the install, an extra_model_paths-style shared folder, and one
    configured after it."""
    dirs = {
        "checkpoints": temp_dir / "models" / "checkpoints",
        "shared": temp_dir / "shared" / "models" / "checkpoints",
        "late": temp_dir / "late" / "checkpoints",
        "loras": temp_dir / "models" / "loras",
        "input": temp_dir / "input",
        "output": temp_dir / "output",
        "temp": temp_dir / "temp",
    }
    for path in dirs.values():
        path.mkdir(parents=True)
    exts = {".safetensors"}
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "checkpoints": ([str(dirs["checkpoints"]), str(dirs["shared"]), str(dirs["late"])], exts),
        "loras": ([str(dirs["loras"])], exts),
    })
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(dirs["input"]))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(dirs["output"]))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(dirs["temp"]))
    return dirs


def _write(path: Path, payload: bytes = b"model-bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _startup_scan(caplog: pytest.LogCaptureFixture, roots: tuple = ALL_ROOTS) -> dict:
    """One startup scan (prune first, then a fast scan); returns its scan_completed fields."""
    seeder = seeder_module._AssetSeeder()
    seeder._state = seeder_module.State.RUNNING
    seeder._scan_state = seeder_module._ScanState()
    seeder._roots = roots
    seeder._phase = seeder_module.ScanPhase.FAST
    seeder._prune_first = True
    seeder._run_gate.set()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        seeder._run_scan()
    assert events_named(caplog, "seeder.scan_failed") == []
    (completed,) = events_named(caplog, "seeder.scan_completed")
    return completed


def _missing(session) -> dict[str, bool]:
    """path -> missing, for every record."""
    session.expire_all()
    return dict(session.execute(
        sa.select(AssetContent.path, AssetContent.is_missing).join(
            Asset, Asset.content_id == AssetContent.id
        )
    ).all())


def _seed(layout: dict[str, Path], session, caplog) -> list[Path]:
    shared_files = [_write(layout["shared"] / f"shared_{i}.safetensors") for i in range(3)]
    _write(layout["checkpoints"] / "local.safetensors")
    _write(layout["input"] / "photo.png", b"png")
    _startup_scan(caplog)
    assert _missing(session) == {str(p): False for p in [
        *shared_files, layout["checkpoints"] / "local.safetensors", layout["input"] / "photo.png",
    ]}
    return shared_files


def _add_new_files(layout: dict[str, Path]) -> list[Path]:
    return [
        _write(layout["loras"] / "new_lora.safetensors"),
        _write(layout["output"] / "ComfyUI_00001_.png", b"png"),
    ]


def test_a_model_folder_that_vanishes_after_being_listed_does_not_abort_the_scan(
    layout, session, caplog
):
    """folder_paths' cached listing raises once a folder it recorded is gone (and checked
    before any folder that changed). The category's other folders are still catalogued."""
    shared_files = _seed(layout, session, caplog)
    layout["shared"].rename(layout["shared"].with_name("checkpoints-away"))
    added = [*_add_new_files(layout), _write(layout["late"] / "new_checkpoint.safetensors")]

    completed = _startup_scan(caplog)

    assert completed["created"] == 3
    missing = _missing(session)
    assert all(missing[str(p)] is False for p in added)
    assert missing[str(layout["checkpoints"] / "local.safetensors")] is False
    # Unchanged: the gone folder's rows are marked missing, as they are on master.
    assert all(missing[str(p)] is True for p in shared_files)

    # Every later scan still lists the category rather than skipping it.
    later = _write(layout["late"] / "later_checkpoint.safetensors")
    assert _startup_scan(caplog)["created"] == 1
    assert _missing(session)[str(later)] is False


def test_a_model_category_whose_listing_raises_an_oserror_is_skipped(
    layout, session, caplog, monkeypatch
):
    """Any OSError listing a folder, e.g. Windows' WinError 433 (device does not exist):
    that category's folders are skipped, its rows kept, and the rest catalogued."""
    shared_files = _seed(layout, session, caplog)
    checkpoint_dirs = tuple(str(layout[k]) for k in ("checkpoints", "shared", "late"))
    real_scandir = file_utils.os.scandir

    def raising(path):
        if str(path).startswith(checkpoint_dirs):
            raise OSError(errno.EINVAL, "A device which does not exist was specified")
        return real_scandir(path)

    monkeypatch.setattr(file_utils.os, "scandir", raising)
    added = _add_new_files(layout)
    unlisted = _write(layout["late"] / "unlisted.safetensors")

    completed = _startup_scan(caplog)

    assert completed["created"] == 2
    missing = _missing(session)
    assert all(missing[str(p)] is False for p in [*added, *shared_files])
    assert str(unlisted) not in missing


def test_a_model_folder_absent_at_startup_still_has_its_rows_marked_missing(
    layout, session, caplog
):
    """Unchanged from master: nothing raises here, and the sync retires the rows."""
    shared_files = _seed(layout, session, caplog)
    layout["shared"].rename(layout["shared"].with_name("checkpoints-away"))
    folder_paths.filename_list_cache.clear()

    completed = _startup_scan(caplog)

    assert completed["created"] == 0
    missing = _missing(session)
    assert all(missing[str(p)] is True for p in shared_files)
    assert missing[str(layout["checkpoints"] / "local.safetensors")] is False


def test_a_rescan_of_an_unchanged_library_stats_no_file(layout, session, caplog, monkeypatch):
    """The walk's stats answer for every catalogued file, so a rescan with nothing changed
    makes no per-file stat of its own, and still creates and retires nothing."""
    monkeypatch.setattr(seeder_module, "TRUST_LISTED_STATS", True)
    shared_files = _seed(layout, session, caplog)

    completed = _startup_scan(caplog)

    assert (completed["created"], completed["files_statted_count"]) == (0, 0)
    assert all(_missing(session)[str(p)] is False for p in shared_files)


@pytest.mark.parametrize("trusted,per_file", [(True, 2), (False, 3)])
def test_a_fresh_scan_skips_admissions_first_stat(layout, caplog, monkeypatch, trusted, per_file):
    """The walk's stat is admission's first where listings are trusted, so each new file
    costs admission's second stat and the seed's observation stat only; elsewhere (Windows)
    admission takes its own first stat as well."""
    monkeypatch.setattr(seeder_module, "TRUST_LISTED_STATS", trusted)
    _write(layout["shared"] / "a.safetensors")
    _write(layout["late"] / "b.safetensors")
    _write(layout["input"] / "photo.png", b"png")

    completed = _startup_scan(caplog)

    assert (completed["created"], completed["files_statted_count"]) == (3, 3 * per_file)


def test_the_output_rescan_stats_only_its_new_files(layout, session, caplog):
    """The rescan queued after a prompt doesn't check every row, so its walk takes no stats:
    each new file costs admission's two and the seed's one, and existing outputs none."""
    _seed(layout, session, caplog)
    for i in range(3):
        _write(layout["output"] / f"old_{i}.png", b"png")
    _startup_scan(caplog, ("output",))
    _write(layout["output"] / "new.png", b"png")

    completed = _startup_scan(caplog, ("output",))

    assert (completed["created"], completed["files_statted_count"]) == (1, 3)


def _live_size(session, path: Path) -> int | None:
    session.expire_all()
    return session.scalar(
        sa.select(AssetContent.size_bytes).where(
            AssetContent.path == str(path), AssetContent.is_missing.is_(False)
        )
    )


def test_a_model_changed_through_another_hard_link_is_seen_by_the_next_scan(
    layout, session, caplog, temp_dir
):
    """NTFS updates a directory entry only through the link a file was written by, so a
    listing can show a model changed through another link as unchanged; the rescan must
    still see the new size. This guards Windows itself; elsewhere the listing's stat is
    fresh, and the stale-listing test below covers the rule on every platform."""
    model = _write(layout["checkpoints"] / "linked.safetensors", b"x" * 64)
    _startup_scan(caplog)
    assert _live_size(session, model) == 64
    other = temp_dir / "elsewhere.safetensors"
    os.link(model, other)
    with open(other, "ab") as f:
        f.write(b"y" * 16)
    os.utime(other, ns=(0, os.stat(other).st_mtime_ns + 5_000_000_000))

    _startup_scan(caplog)

    assert _live_size(session, model) == 80


def test_where_listings_can_be_stale_a_rescan_stats_every_row(layout, session, caplog, monkeypatch):
    shared_files = _seed(layout, session, caplog)
    monkeypatch.setattr(seeder_module, "TRUST_LISTED_STATS", False)

    completed = _startup_scan(caplog)

    assert completed["created"] == 0
    assert completed["files_statted_count"] == len(shared_files) + 2  # every catalogued row


def test_where_listings_can_be_stale_a_stale_listing_does_not_hide_a_change(
    layout, session, caplog, monkeypatch
):
    """A listing that still shows a model's old size and mtime (as NTFS can) is ignored
    where listings aren't trusted: the rescan's own stat sees the change."""
    model = _write(layout["checkpoints"] / "stale.safetensors", b"x" * 64)
    _startup_scan(caplog)
    stale = os.stat(model)
    _write(model, b"x" * 80)
    os.utime(model, ns=(0, stale.st_mtime_ns + 5_000_000_000))
    real_walk = file_utils.walk_listings

    def stale_walk(base, *args, **kwargs):
        walk = real_walk(base, *args, **kwargs)
        if str(model) in walk.stats:
            walk.stats[str(model)] = stale
        return walk

    monkeypatch.setattr(scanner_module, "walk_listings", stale_walk)
    monkeypatch.setattr(seeder_module, "TRUST_LISTED_STATS", False)

    _startup_scan(caplog)

    assert _live_size(session, model) == 80

"""A walk with a cache reuses the listing of each directory whose mtime, inode and change
time are unchanged, on a filesystem trusted to keep directory mtimes current, and still
stats every directory, so a change deep in the tree is found."""

import os
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session as SASession, sessionmaker

from app.assets import scanner, seeder as seeder_module
from app.assets.database.models import AssetContent
from app.assets.scanner_admission import _WATCH_LIST
from app.assets.services import file_utils
from app.assets.services.file_utils import walk_listings


def _write(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _age(root: Path, seconds: int = 60) -> None:
    """Set every directory's mtime well outside the recent-change window."""
    when = time.time() - seconds
    for dirpath, _dirs, _files in os.walk(root):
        os.utime(dirpath, (when, when))


def _trust_all(_path: str) -> bool:
    return True


def _walk(root: Path, cache, **kwargs):
    return walk_listings(str(root), cache=cache, trusted=kwargs.pop("trusted", _trust_all), **kwargs)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "out"
    for rel in ("top.png", "a/one.png", "a/b/two.png", "a/b/c/three.png", "d/four.png"):
        _write(root / rel)
    _age(root)
    return root


def test_unchanged_directories_are_reused_with_the_same_files(tree: Path):
    first = _walk(tree, {})
    second = _walk(tree, first.cache)

    assert first.dirs_reused == 0
    assert (second.dirs_listed, second.dirs_reused) == (0, len(first.listings))
    assert sorted(second.files) == sorted(first.files)
    assert second.listings == first.listings


def test_a_deep_add_relists_only_its_directory_and_is_found(tree: Path, monkeypatch):
    first = _walk(tree, {})
    added = _write(tree / "a" / "b" / "c" / "new.png")
    listed: list[str] = []
    real = file_utils._list_visible_entries

    def spy(dirpath, interrupted=None, stat_files=False):
        listed.append(dirpath)
        return real(dirpath, interrupted, stat_files)

    monkeypatch.setattr(file_utils, "_list_visible_entries", spy)
    second = _walk(tree, first.cache)

    assert listed == [str(tree / "a" / "b" / "c")]
    assert str(added) in second.files
    assert second.dirs_reused == len(first.listings) - 1


def test_a_removed_and_a_renamed_subdirectory_relist_their_parent(tree: Path):
    first = _walk(tree, {})
    os.rename(tree / "d", tree / "e")
    (tree / "a" / "b" / "c" / "three.png").unlink()
    (tree / "a" / "b" / "c").rmdir()

    second = _walk(tree, first.cache)

    assert sorted(second.files) == sorted(
        str(tree / rel) for rel in ("top.png", "a/one.png", "a/b/two.png", "e/four.png")
    )


def test_a_directory_changed_within_the_window_of_its_listing_is_relisted(tree: Path, monkeypatch):
    mtime_ns = os.stat(tree).st_mtime_ns
    monkeypatch.setattr(file_utils.time, "time_ns", lambda: mtime_ns + 1_000_000_000)
    first = _walk(tree, {})
    monkeypatch.setattr(file_utils.time, "time_ns", lambda: mtime_ns + 10_000_000_000)

    second = _walk(tree, first.cache)

    # Listed one second after its mtime: a change in that tick could hide behind it.
    assert second.dirs_reused == 0
    assert _walk(tree, second.cache).dirs_listed == 0


def test_a_directory_with_a_future_mtime_is_never_reused(tree: Path):
    future = time.time() + 3600
    os.utime(tree, (future, future))
    first = _walk(tree, {})
    second = _walk(tree, first.cache)
    assert second.dirs_listed == 1


def test_a_directory_replaced_by_one_with_the_same_mtime_is_relisted(tree: Path):
    first = _walk(tree, {})
    old = os.stat(tree / "d")
    replacement = tree / "d.new"
    _write(replacement / "five.png")
    for path in (tree / "d" / "four.png", tree / "d"):
        path.unlink() if path.is_file() else path.rmdir()
    os.rename(replacement, tree / "d")
    os.utime(tree / "d", ns=(old.st_atime_ns, old.st_mtime_ns))
    os.utime(tree, ns=(os.stat(tree).st_atime_ns, first.cache[str(tree)].key[0]))

    second = _walk(tree, first.cache)

    assert str(tree / "d" / "five.png") in second.files
    assert str(tree / "d" / "four.png") not in second.files


def test_an_untrusted_filesystem_lists_every_directory_and_caches_nothing(tree: Path):
    first = _walk(tree, {}, trusted=lambda _p: False)
    second = _walk(tree, first.cache, trusted=lambda _p: False)
    assert second.dirs_reused == 0
    assert first.cache == {} and second.cache == {}


def test_trust_is_asked_at_the_root_and_through_links_only(tree: Path):
    elsewhere = tree.parent / "elsewhere"
    _write(elsewhere / "linked.png")
    (tree / "a" / "link").symlink_to(elsewhere, target_is_directory=True)
    asked: list[str] = []

    walk_listings(str(tree), cache={}, trusted=lambda p: asked.append(p) or True)

    assert asked == [str(tree), str(tree / "a" / "link")]


def test_a_file_links_target_deleted_drops_it_from_a_reused_listing(tree: Path):
    target = _write(tree.parent / "target.png")
    (tree / "d" / "link.png").symlink_to(target)
    _age(tree)
    first = _walk(tree, {})
    assert str(tree / "d" / "link.png") in first.files

    target.unlink()
    second = _walk(tree, first.cache)

    assert second.dirs_reused == len(first.listings)
    assert str(tree / "d" / "link.png") not in second.files
    assert "link.png" not in second.listings[str(tree / "d")][0]


def test_a_dangling_folder_link_whose_target_returns_is_walked(tree: Path):
    share = tree.parent / "share"
    (tree / "a" / "nas").symlink_to(share, target_is_directory=True)
    _age(tree)
    first = _walk(tree, {})
    assert not any("nas" in f for f in first.files)

    _write(share / "model.safetensors")
    second = _walk(tree, first.cache)

    assert str(tree / "a" / "nas" / "model.safetensors") in second.files


def test_files_in_a_reused_directory_are_known_regular(tree: Path):
    first = _walk(tree, {}, stat_files=True)
    second = _walk(tree, first.cache, stat_files=True)
    assert second.stats == {}
    assert second.reused_regular == frozenset(first.files)


def test_a_stopped_walk_is_incomplete(tree: Path):
    calls = iter([False, True])
    walk = _walk(tree, {}, should_stop=lambda: next(calls, True))
    assert walk.complete is False


# The scanner keeps each base's cache between scans; the output rescan's listing diff
# must stay exact with reused listings.


@pytest.fixture
def output(tmp_path: Path, monkeypatch) -> Path:
    out = tmp_path / "output"
    inp = tmp_path / "input"
    out.mkdir()
    inp.mkdir()
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(out))
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(inp))
    monkeypatch.setattr(scanner, "get_comfy_models_folders", lambda: [])
    monkeypatch.setattr(scanner, "directory_mtimes_trusted", _trust_all)
    monkeypatch.setattr(scanner, "_listing_caches", {})
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


def _scan(roots=("output",)):
    seeder = seeder_module._AssetSeeder()
    seeder._scan_state = seeder_module._ScanState()
    seeder._phase = seeder_module.ScanPhase.FAST
    seeder._run_gate.set()
    seeder._cancel_event.clear()
    return seeder._run_fast_phase(roots)


def _missing(session, path: Path) -> bool:
    session.expire_all()
    (row,) = session.scalars(sa.select(AssetContent).where(AssetContent.path == str(path)))
    return row.is_missing


@pytest.mark.parametrize("roots", [("output",), ("models", "input", "output")])
def test_deep_changes_under_reused_parents_are_found(output: Path, session, roots):
    files = [_write(output / rel) for rel in ("x.png", "p/y.png", "p/q/r/z.png")]
    _age(output)
    _scan(roots)
    _scan(roots)  # fills the cache from listings taken outside the window
    assert scanner._listing_caches

    files[2].unlink()
    added = _write(output / "p" / "q" / "new.png")
    _scan(roots)

    assert _missing(session, files[2]) is True
    assert _missing(session, files[1]) is False
    assert _missing(session, added) is False


def test_a_file_links_target_deleted_is_marked_missing_by_the_output_rescan(output: Path, session):
    target = _write(output.parent / "elsewhere" / "t.png")
    link = output / "sub" / "t.png"
    link.parent.mkdir()
    link.symlink_to(target)
    _age(output)
    _scan()
    _scan()

    target.unlink()
    _scan()

    assert _missing(session, link) is True


def test_a_folder_no_longer_configured_is_dropped_from_the_cache(output: Path, tmp_path: Path, monkeypatch):
    _write(output / "x.png")
    _scan()
    assert str(output) in scanner._listing_caches
    moved = tmp_path / "output2"
    moved.mkdir()
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(moved))
    _scan()
    assert set(scanner._listing_caches) == {str(moved)}


def test_a_model_in_a_reused_folder_still_shadows_a_later_folders_duplicate(output: Path, tmp_path: Path, monkeypatch):
    first, second = tmp_path / "models1", tmp_path / "models2"
    kept = _write(first / "a.safetensors")
    _write(second / "a.safetensors")
    _age(first)
    folders = [("checkpoints", [str(first), str(second)], {".safetensors"})]
    monkeypatch.setattr(scanner, "get_comfy_models_folders", lambda: folders)
    assert scanner.list_root("models").files == [str(kept)]

    added = _write(second / "z.safetensors")
    walk = scanner.list_root("models")

    assert walk.dirs_reused == 1
    assert sorted(walk.files) == [str(kept), str(added)]


def test_trust_is_asked_again_where_the_walk_enters_another_device(tree: Path, monkeypatch):
    real_stat = os.stat
    mount = str(tree / "a" / "b")

    def stat(path, *args, **kwargs):
        st = real_stat(path, *args, **kwargs)
        if str(path) == mount or str(path).startswith(mount + os.sep):
            fields = list(st)
            fields[2] += 1  # st_dev
            return os.stat_result(fields, {"st_mtime_ns": st.st_mtime_ns, "st_ctime_ns": st.st_ctime_ns})
        return st

    monkeypatch.setattr(file_utils.os, "stat", stat)
    asked: list[str] = []
    walk = walk_listings(str(tree), cache={}, trusted=lambda p: asked.append(p) or p != mount)

    assert asked == [str(tree), mount]
    assert str(tree / "a") in walk.cache and mount not in walk.cache
    assert str(tree / "a" / "b" / "c") not in walk.cache  # inherits the mount's verdict


def test_a_scan_stopped_part_way_keeps_the_previous_cache(output: Path):
    _write(output / "p" / "x.png")
    _age(output)
    scanner.list_root("output")
    before = dict(scanner._listing_caches)

    calls = iter([False, True])
    walk = scanner.list_root("output", should_stop=lambda: next(calls, True))

    assert walk.complete is False
    assert scanner._listing_caches == before


def test_a_directory_no_longer_trusted_is_listed_despite_its_cached_listing(tree: Path):
    first = _walk(tree, {})
    second = _walk(tree, first.cache, trusted=lambda _p: False)
    assert second.dirs_reused == 0

"""Models are listed by the scan walker, one walk per folder, and its stats stand in for
the scan's own: a catalogued file listed unchanged isn't stat'ed again, and admission
takes the listed stat as its first."""

import logging
import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import folder_paths
from app.assets import scanner
from app.assets.database.models import Base
from app.assets.database.queries import create_content
from app.assets.scanner import build_asset_specs, observe_references_on_filesystem
from app.assets.scanner_admission import _WATCH_LIST
from app.assets.services import file_utils
from app.assets.services.path_utils import get_comfy_models_folders


def _master_collect_models_files() -> list[str]:
    """master's collect_models_files (folder_paths listing, then get_full_path per file),
    frozen as the parity reference."""
    out: list[str] = []
    for folder_name, bases, _exts in get_comfy_models_folders():
        try:
            rel_files = folder_paths.get_filename_list(folder_name) or []
        except OSError:
            try:
                rel_files = folder_paths.get_filename_list_(folder_name)[0]
            except OSError as e:
                logging.warning("skipping model category %s: %s", folder_name, e)
                continue
        for rel_path in rel_files:
            if not all(file_utils.is_visible(part) for part in Path(rel_path).parts):
                continue
            abs_path = folder_paths.get_full_path(folder_name, rel_path)
            if not abs_path:
                continue
            abs_path = os.path.abspath(abs_path)
            if any(Path(abs_path).is_relative_to(os.path.abspath(b)) for b in bases):
                out.append(abs_path)
    return out


@pytest.fixture(autouse=True)
def _clear_watch_list():
    _WATCH_LIST.clear()
    yield
    _WATCH_LIST.clear()


def _write(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture
def models(tmp_path, monkeypatch) -> dict[str, Path]:
    """Two checkpoint folders holding the same name, a folder shared by two categories,
    an extensionless category, and the files each category must leave out."""
    dirs = {name: tmp_path / name for name in ("ckpt_a", "ckpt_b", "shared", "classifiers")}
    for path in dirs.values():
        path.mkdir()
    _write(dirs["ckpt_a"] / "same.safetensors", b"from a")
    _write(dirs["ckpt_a"] / "sub" / "deep.safetensors")
    _write(dirs["ckpt_a"] / "notes.txt")
    _write(dirs["ckpt_a"] / ".hidden" / "secret.safetensors")
    _write(dirs["ckpt_a"] / ".dotfile.safetensors")
    _write(dirs["ckpt_b"] / "same.safetensors", b"from b")
    _write(dirs["ckpt_b"] / "only_b.safetensors")
    _write(dirs["shared"] / "same.safetensors", b"shared")
    _write(dirs["shared"] / "lora.pt")
    _write(dirs["shared"] / "readme.md")
    _write(dirs["classifiers"] / "noext")
    _write(dirs["classifiers"] / "with.pt")
    exts = {".safetensors", ".pt"}
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "checkpoints": ([str(dirs["ckpt_a"]), str(dirs["ckpt_b"])], exts),
        "loras": ([str(dirs["shared"]), str(dirs["ckpt_b"])], exts),
        "vae": ([str(dirs["shared"])], {".safetensors"}),
        "classifiers": ([str(dirs["classifiers"])], {""}),
        "diffusers": ([str(dirs["shared"])], ["folder"]),
        "missing": ([str(tmp_path / "not-there")], exts),
    })
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    return dirs


def test_the_walk_finds_exactly_the_files_master_resolves(models):
    walk = scanner.list_root("models")
    assert sorted(walk.files) == sorted(set(_master_collect_models_files()))
    assert len(walk.files) == len(set(walk.files))
    assert str(models["ckpt_b"] / "only_b.safetensors") in walk.files
    assert str(models["classifiers"] / "noext") in walk.files
    assert str(models["classifiers"] / "with.pt") not in walk.files
    assert str(models["shared"] / "readme.md") not in walk.files


def test_a_name_in_an_earlier_folder_of_the_category_shadows_the_later_one(models):
    files = scanner.list_root("models").files
    assert str(models["ckpt_a"] / "same.safetensors") in files  # checkpoints' first folder
    assert str(models["shared"] / "same.safetensors") in files  # loras' first folder
    assert str(models["ckpt_b"] / "same.safetensors") not in files  # second for both


def test_each_folder_is_walked_once_and_every_listed_file_has_its_stat(models, monkeypatch):
    walked: list[str] = []
    real = scanner.walk_listings

    def counting(base, *args):
        walked.append(base)
        return real(base, *args)

    monkeypatch.setattr(scanner, "walk_listings", counting)
    walk = scanner.list_root("models")
    folders = [*models.values(), models["ckpt_a"].parent / "not-there"]
    assert sorted(walked) == sorted(os.path.abspath(p) for p in folders)
    for path in walk.files:
        assert walk.stats[path].st_size == os.stat(path).st_size


def test_the_walk_records_no_stat_for_a_file_it_could_not_stat(tmp_path, monkeypatch):
    good = _write(tmp_path / "good.png")
    bad = _write(tmp_path / "bad.png")
    real_scandir = file_utils.os.scandir

    class _Entry:
        def __init__(self, entry):
            self._entry = entry
            self.name, self.path = entry.name, entry.path

        def is_dir(self):
            return self._entry.is_dir()

        def is_symlink(self):
            return self._entry.is_symlink()

        def stat(self):
            if self.name == "bad.png":
                raise OSError(5, "I/O error")
            return self._entry.stat()

    class _Listing:
        def __init__(self, path):
            self._it = real_scandir(path)

        def __enter__(self):
            return (_Entry(e) for e in self._it)

        def __exit__(self, *exc):
            self._it.close()

    monkeypatch.setattr(file_utils.os, "scandir", _Listing)
    walk = file_utils.walk_listings(str(tmp_path))
    assert sorted(walk.files) == sorted([str(good), str(bad)])
    assert set(walk.stats) == {str(good)}


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session


class _Counter:
    files_statted = 0
    permission_denied = 0

    def mark_emitted(self, key: str) -> bool:
        return True


def _row(session, path: Path):
    st = path.stat()
    create_content(session, str(path), size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def test_a_row_listed_unchanged_is_not_stated_again(tmp_path, session):
    unchanged = _write(tmp_path / "unchanged.png")
    changed = _write(tmp_path / "changed.png")
    gone = _write(tmp_path / "gone.png")
    for path in (unchanged, changed, gone):
        _row(session, path)
    os.utime(changed, ns=(0, changed.stat().st_mtime_ns + 5_000_000_000))
    gone.unlink()
    walk = file_utils.walk_listings(str(tmp_path))
    counter = _Counter()

    observations, survivors = observe_references_on_filesystem(
        session, [str(tmp_path)], counter, walk.stats
    )

    assert survivors == {str(unchanged), str(changed)}
    assert counter.files_statted == 2  # changed and gone; unchanged came from the listing
    assert {o.stat_result is None for o in observations} == {True, False}
    assert len(observations) == 2


def test_a_listed_mtime_that_differs_is_decided_by_a_fresh_stat(tmp_path, session):
    """A listing can carry a file's size and mtime from before a writer opened it (NTFS,
    and shares served from it). The scan's change decision is the fresh stat's."""
    path = _write(tmp_path / "model.safetensors")
    _row(session, path)
    stale = os.stat_result((0,) * 6 + (0, 1, 1, 1))  # an mtime the catalogue doesn't have
    counter = _Counter()

    observations, survivors = observe_references_on_filesystem(
        session, [str(tmp_path)], counter, {str(path): stale}
    )

    assert counter.files_statted == 1
    assert survivors == {str(path)}
    assert observations == []  # the fresh stat matches the catalogue


def test_admission_takes_the_listed_stat_as_its_first(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    paths = [str(_write(tmp_path / f"f{i}.png")) for i in range(3)]
    walk = file_utils.walk_listings(str(tmp_path))
    counter = _Counter()

    specs, _tags, _skipped = build_asset_specs(
        paths, set(), enable_metadata_extraction=False, progress=counter, listed_stats=walk.stats
    )

    assert sorted(s["abs_path"] for s in specs) == sorted(paths)
    assert counter.files_statted == 3  # the second stats only


def test_a_listing_frozen_while_a_writer_holds_the_file_sends_it_to_the_watch_list(tmp_path):
    """The listing kept the size and mtime from before the writer opened the file, while
    the file itself has grown: the second stat disagrees, so it isn't admitted."""
    path = _write(tmp_path / "resumed.safetensors", b"x" * 10)
    frozen = os.stat(path)
    _write(path, b"x" * 20)

    specs, _tags, _skipped = build_asset_specs(
        [str(path)], set(), enable_metadata_extraction=False, listed_stats={str(path): frozen}
    )

    assert specs == []

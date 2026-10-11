"""Models are listed by the scan walker, one walk per folder, and its stats stand in for
the scan's own: a catalogued file listed unchanged isn't stat'ed again, and admission
takes the listed stat as its first."""

import logging
import os
import sys
from pathlib import Path

import pytest
import folder_paths
from app.assets import scanner
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
    _write(dirs["ckpt_a"] / "UPPER.SAFETENSORS")
    if hasattr(os, "mkfifo"):
        os.mkfifo(dirs["ckpt_a"] / "piped.safetensors")  # not a file: the later folder's one counts
    _write(dirs["ckpt_b"] / "piped.safetensors")
    _write(tmp_path / "datasets" / "anything.bin")
    _write(tmp_path / "mixed" / "weights.bin")  # a category mixing "folder" with a real extension
    _write(tmp_path / "diffusers" / "unet" / "model.safetensors")  # a folder-only category's files
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
        "diffusers": ([str(dirs["shared"]), str(tmp_path / "diffusers")], ["folder"]),
        "missing": ([str(tmp_path / "not-there")], exts),
        "datasets": ([str(tmp_path / "datasets")], set()),
        "mixed": ([str(tmp_path / "mixed")], ["folder", ".bin"]),
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
    assert str(models["ckpt_a"] / "UPPER.SAFETENSORS") in walk.files
    assert str(models["ckpt_b"] / "piped.safetensors") in walk.files
    assert str(models["ckpt_a"].parent / "datasets" / "anything.bin") in walk.files
    assert str(models["ckpt_a"].parent / "mixed" / "weights.bin") in walk.files


def test_a_name_in_an_earlier_folder_of_the_category_shadows_the_later_one(models):
    files = scanner.list_root("models").files
    assert str(models["ckpt_a"] / "same.safetensors") in files  # checkpoints' first folder
    assert str(models["shared"] / "same.safetensors") in files  # loras' first folder
    assert str(models["ckpt_b"] / "same.safetensors") not in files  # second for both


def test_each_folder_is_walked_once_and_every_listed_file_has_its_stat(models, monkeypatch):
    walked: list[str] = []
    real = scanner.walk_listings

    def counting(base, *args, **kwargs):
        walked.append(base)
        return real(base, *args, **kwargs)

    monkeypatch.setattr(scanner, "walk_listings", counting)
    walk = scanner.list_root("models")
    folders = [*models.values(), *(models["ckpt_a"].parent / n for n in ("not-there", "datasets", "mixed"))]
    assert sorted(walked) == sorted(os.path.abspath(p) for p in folders)
    # ckpt_a, ckpt_a/sub, ckpt_b, shared, classifiers, datasets, mixed; hidden and missing ones aren't
    assert walk.dirs_listed == 7
    for path in walk.files:
        assert walk.stats[path].st_size == os.stat(path).st_size


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
    walk = file_utils.walk_listings(str(tmp_path), stat_files=True)
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
    other = _write(tmp_path / "other.bin")
    os.utime(other, ns=(0, path.stat().st_mtime_ns - 60_000_000_000))
    stale = other.stat()  # a real stat, with an mtime the catalogue doesn't have
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
    walk = file_utils.walk_listings(str(tmp_path), stat_files=True)
    counter = _Counter()

    specs, _tags, _skipped = build_asset_specs(
        paths, set(), enable_metadata_extraction=False, progress=counter, listed_stats=walk.stats
    )

    assert sorted(s["abs_path"] for s in specs) == sorted(paths)
    assert counter.files_statted == 3  # the second stats only


def test_a_listing_frozen_while_a_writer_holds_the_file_sends_it_to_the_watch_list(
    tmp_path, monkeypatch
):
    """The listing kept the size and mtime from before the writer opened the file, while
    the file itself has grown: the second stat disagrees, so it isn't admitted."""
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    path = _write(tmp_path / "resumed.safetensors", b"x" * 10)
    frozen = os.stat(path)
    _write(path, b"x" * 20)

    specs, _tags, _skipped = build_asset_specs(
        [str(path)], set(), enable_metadata_extraction=False, listed_stats={str(path): frozen}
    )

    assert specs == []
    assert [entry.path for entry in _WATCH_LIST] == [str(path)]


def test_a_name_differing_only_in_case_is_shadowed_where_names_ignore_case(tmp_path, monkeypatch):
    """On a case-insensitive filesystem (Windows), the loader resolves case.safetensors to
    the first folder's Case.safetensors, so the second folder's file is not catalogued."""
    first = _write(tmp_path / "a" / "Case.safetensors")
    second = _write(tmp_path / "b" / "case.safetensors")
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "checkpoints": ([str(first.parent), str(second.parent)], {".safetensors"}),
    })
    monkeypatch.setattr(scanner.os.path, "normcase", str.lower)
    assert scanner.list_root("models").files == [str(first)]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_folder_reached_twice_inside_one_model_folder_is_catalogued_once(models):
    """Unlike the loader's listing, the walk lists a directory once per model folder, so a
    second symlink to it adds no second path for the same file."""
    os.symlink(models["ckpt_a"] / "sub", models["ckpt_a"] / "alias")
    files = scanner.list_root("models").files
    deep = [p for p in files if p.endswith("deep.safetensors")]
    assert len(deep) == 1
    assert len([p for p in _master_collect_models_files() if p.endswith("deep.safetensors")]) == 2


def test_the_models_walk_takes_the_scans_pause_and_cancel(models, monkeypatch):
    gates = (lambda: False, lambda: False, lambda: False)
    passed: list[tuple] = []
    real = scanner.walk_listings

    def recording(base, *args, **kwargs):
        passed.append(args)
        return real(base, *args, **kwargs)

    monkeypatch.setattr(scanner, "walk_listings", recording)
    scanner.list_root("models", *gates)
    assert passed and all(args == gates for args in passed)


def test_only_a_scan_that_checks_every_row_stats_while_listing(tmp_path, monkeypatch):
    _write(tmp_path / "out.png")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    assert scanner.list_root("output").stats == {}
    assert set(scanner.list_root("output", stat_files=True).stats) == {str(tmp_path / "out.png")}


def test_admission_stats_a_file_the_walk_has_no_stat_for(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    paths = [str(_write(tmp_path / f"f{i}.png")) for i in range(3)]
    stats = file_utils.walk_listings(str(tmp_path), stat_files=True).stats
    del stats[paths[0]]
    counter = _Counter()

    specs, _tags, _skipped = build_asset_specs(
        paths, set(), enable_metadata_extraction=False, progress=counter, listed_stats=stats
    )

    assert len(specs) == 3
    assert counter.files_statted == 4  # one first stat, three second


def test_a_file_listed_empty_and_filled_since_is_admitted(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    path = _write(tmp_path / "late.png", b"")
    listed = os.stat(path)
    _write(path, b"filled")

    specs, _tags, _skipped = build_asset_specs(
        [str(path)], set(), enable_metadata_extraction=False, listed_stats={str(path): listed}
    )

    assert [s["size_bytes"] for s in specs] == [6]


def test_a_file_that_cant_be_stated_at_admission_is_skipped_not_fatal(tmp_path, monkeypatch):
    """The first stat may be the walk's, taken long before admission's second; an I/O error
    in between drops that one file, as the discovery stat's errors always did."""
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    good = str(_write(tmp_path / "good.png"))
    flaky = str(_write(tmp_path / "flaky.png"))
    stats = file_utils.walk_listings(str(tmp_path), stat_files=True).stats
    real_stat = os.stat

    def failing_stat(path, *args, **kwargs):
        if str(path) == flaky:
            raise OSError(5, "Input/output error", path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr("app.assets.scanner_admission.os.stat", failing_stat)

    specs, _tags, _skipped = build_asset_specs(
        [good, flaky], set(), enable_metadata_extraction=False, listed_stats=stats
    )

    assert [s["abs_path"] for s in specs] == [good]

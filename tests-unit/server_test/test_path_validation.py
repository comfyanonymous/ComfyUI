from pathlib import Path

from comfy.path_validation import resolve_safe_path


def test_resolve_safe_path_allows_nested_relative_path(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()

    resolved = resolve_safe_path(base, "subdir/image.png")

    assert resolved == base / "subdir" / "image.png"


def test_resolve_safe_path_rejects_parent_traversal(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()

    assert resolve_safe_path(base, "../escape.png") is None


def test_resolve_safe_path_rejects_windows_separator_parent_traversal(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()

    assert resolve_safe_path(base, r"..\escape.png") is None


def test_resolve_safe_path_rejects_posix_absolute_path(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()

    assert resolve_safe_path(base, "/tmp/escape.png") is None


def test_resolve_safe_path_rejects_windows_drive_path(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()

    assert resolve_safe_path(base, r"C:\Users\public\escape.png") is None


def test_resolve_safe_path_rejects_windows_unc_path(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()

    assert resolve_safe_path(base, r"\\server\share\escape.png") is None


def test_resolve_safe_path_rejects_null_bytes(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()

    assert resolve_safe_path(base, "image.png\0.txt") is None


def test_resolve_safe_path_rejects_symlink_escape(tmp_path: Path) -> None:
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    (outside / "secret.png").write_text("secret")
    (base / "linked").symlink_to(outside, target_is_directory=True)

    assert resolve_safe_path(base, "linked/secret.png") is None

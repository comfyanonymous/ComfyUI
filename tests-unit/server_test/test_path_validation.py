from pathlib import Path

from comfy.path_validation import resolve_safe_path


def resolve_upload_image_path(upload_dir: Path, subfolder: str, filename: str) -> Path | None:
    upload_folder = resolve_safe_path(upload_dir, subfolder)
    if upload_folder is None:
        return None
    return resolve_safe_path(upload_folder, filename)


def resolve_view_path(output_dir: Path, subfolder: str, filename: str) -> Path | None:
    return resolve_safe_path(output_dir, subfolder, filename)


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


def test_upload_image_path_allows_safe_subfolder_and_filename_with_dots(tmp_path: Path) -> None:
    upload_dir = tmp_path / "input"
    upload_dir.mkdir()

    resolved = resolve_upload_image_path(upload_dir, "safe/subfolder", "foo..bar.png")

    assert resolved == upload_dir / "safe" / "subfolder" / "foo..bar.png"


def test_upload_image_path_rejects_bad_subfolder_or_filename(tmp_path: Path) -> None:
    upload_dir = tmp_path / "input"
    upload_dir.mkdir()

    assert resolve_upload_image_path(upload_dir, "../escape", "image.png") is None
    assert resolve_upload_image_path(upload_dir, "safe", "../escape.png") is None
    assert resolve_upload_image_path(upload_dir, "safe", r"\\server\share\escape.png") is None
    assert resolve_upload_image_path(upload_dir, "safe", "image.png\0.txt") is None


def test_view_path_allows_annotated_base_dir_with_subfolder(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    resolved = resolve_view_path(output_dir, "renders", "image.png")

    assert resolved == output_dir / "renders" / "image.png"


def test_view_path_rejects_traversal_absolute_unc_and_null_byte(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    assert resolve_view_path(output_dir, "../escape", "image.png") is None
    assert resolve_view_path(output_dir, "safe", "/tmp/escape.png") is None
    assert resolve_view_path(output_dir, "safe", r"C:\Users\public\escape.png") is None
    assert resolve_view_path(output_dir, "safe", r"\\server\share\escape.png") is None
    assert resolve_view_path(output_dir, "safe", "image.png\0.txt") is None

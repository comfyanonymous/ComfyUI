import os
from pathlib import Path

import folder_paths


def _setup_media_dirs(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir()
    output_dir.mkdir()
    temp_dir.mkdir()
    folder_paths.set_input_directory(str(input_dir))
    folder_paths.set_output_directory(str(output_dir))
    folder_paths.set_temp_directory(str(temp_dir))
    return input_dir, output_dir, temp_dir


def test_symlink_from_input_to_temp_is_allowed(tmp_path: Path):
    input_dir, _output_dir, temp_dir = _setup_media_dirs(tmp_path)
    target = temp_dir / "image.png"
    target.write_bytes(b"png")
    link = input_dir / "link.png"
    link.symlink_to(target)

    assert folder_paths.exists_annotated_filepath("link.png") is True
    assert folder_paths.get_annotated_filepath("link.png") == os.path.abspath(str(link))


def test_symlink_from_input_to_output_is_allowed(tmp_path: Path):
    input_dir, output_dir, _temp_dir = _setup_media_dirs(tmp_path)
    target = output_dir / "image.png"
    target.write_bytes(b"png")
    link = input_dir / "link.png"
    link.symlink_to(target)

    assert folder_paths.exists_annotated_filepath("link.png") is True


def test_symlink_escaping_media_roots_is_rejected(tmp_path: Path):
    input_dir, _output_dir, _temp_dir = _setup_media_dirs(tmp_path)
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"nope")
    link = input_dir / "evil.png"
    link.symlink_to(outside)

    assert folder_paths.exists_annotated_filepath("evil.png") is False
    try:
        folder_paths.get_annotated_filepath("evil.png")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_path_traversal_is_still_rejected(tmp_path: Path):
    _setup_media_dirs(tmp_path)
    assert folder_paths.exists_annotated_filepath("../secret.png") is False

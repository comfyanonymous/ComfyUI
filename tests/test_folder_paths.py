import os

import folder_paths


def test_resolve_subfolder_accepts_directory_inside_base(tmp_path):
    base = tmp_path / "output"
    nested = base / "images"
    nested.mkdir(parents=True)

    resolved = folder_paths.resolve_subfolder(str(base), "images")

    assert resolved == os.path.join(base, "images")


def test_resolve_subfolder_rejects_escape_from_base(tmp_path):
    base = tmp_path / "output"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()

    assert folder_paths.resolve_subfolder(str(base), os.path.join("..", "outside")) is None


def test_resolve_subfolder_handles_cross_drive_paths(monkeypatch):
    def cross_drive_commonpath(paths):
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(os.path, "commonpath", cross_drive_commonpath)

    assert folder_paths.resolve_subfolder(r"M:\output", r"E:\images") is None

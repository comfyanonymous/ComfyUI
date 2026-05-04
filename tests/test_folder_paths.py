import folder_paths


def test_set_input_directory_creates_missing_directory(tmp_path):
    original_input_directory = folder_paths.get_input_directory()
    custom_input_directory = tmp_path / "custom-input"

    try:
        folder_paths.set_input_directory(str(custom_input_directory))

        assert custom_input_directory.is_dir()
    finally:
        folder_paths.set_input_directory(original_input_directory)

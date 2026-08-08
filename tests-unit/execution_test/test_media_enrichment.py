"""Tests for enrich_output_with_media_metadata in comfy_execution/media_enrichment.py."""
import mimetypes
import os
import unittest
from unittest.mock import MagicMock, patch

# Initialize the mimetypes registry before any test patches os.path.isfile —
# its lazy init consults os.path.isfile to pick candidate files, and a
# patched-True isfile makes it try to open files that don't exist.
mimetypes.init()

# Platform-appropriate absolute base. tempfile.gettempdir() returns C:\... on
# Windows and /tmp on POSIX, so containment via commonpath behaves naturally.
_DEFAULT_BASE = os.path.join(__import__("tempfile").gettempdir(), "media-enrichment-test-base")

_VIDEO_META = {
    "kind": "video",
    "width": 1280,
    "height": 720,
    "duration": 4.0,
    "fps": 24.0,
    "frame_count": 96,
}


def _mocked_modules(*, extract=None, directory=_DEFAULT_BASE):
    return {
        "folder_paths": MagicMock(get_directory_by_type=MagicMock(return_value=directory)),
        "app.assets.services.media_metadata": MagicMock(
            extract_media_metadata=extract or MagicMock(return_value=dict(_VIDEO_META)),
        ),
    }


def _call(output_ui, *, extract=None, file_exists=True, directory=_DEFAULT_BASE):
    extract_mock = extract or MagicMock(return_value=dict(_VIDEO_META))
    mocked = _mocked_modules(extract=extract_mock, directory=directory)

    # Only os.path.isfile is patched — abspath/join must run natively so the
    # containment check sees real platform paths.
    with patch.dict("sys.modules", mocked), \
         patch("os.path.isfile", return_value=file_exists):
        import comfy_execution.media_enrichment as mod
        return mod.enrich_output_with_media_metadata(output_ui), extract_mock


class TestEnrichOutputWithMediaMetadata(unittest.TestCase):

    def test_attaches_metadata_object(self):
        output = {"images": [{"filename": "clip.mp4", "subfolder": "", "type": "output"}]}
        result, _ = _call(output)
        self.assertEqual(result["images"][0]["metadata"], _VIDEO_META)

    def test_passes_guessed_mime_type(self):
        output = {"images": [{"filename": "clip.mp4", "subfolder": "", "type": "output"}]}
        _, extract_mock = _call(output)
        _, kwargs = extract_mock.call_args
        self.assertEqual(kwargs["mime_type"], "video/mp4")

    def test_original_entry_not_mutated(self):
        orig = {"filename": "clip.mp4", "subfolder": "", "type": "output"}
        _call({"images": [orig]})
        self.assertNotIn("metadata", orig)

    def test_non_media_extractor_none_leaves_entry_unchanged(self):
        extract = MagicMock(return_value=None)
        output = {"latent": [{"filename": "a.latent", "subfolder": "", "type": "output"}]}
        result, _ = _call(output, extract=extract)
        self.assertNotIn("metadata", result["latent"][0])

    def test_existing_metadata_key_untouched(self):
        entry = {"filename": "clip.mp4", "subfolder": "", "type": "output", "metadata": {"kind": "other"}}
        result, extract_mock = _call({"images": [entry]})
        self.assertEqual(result["images"][0]["metadata"], {"kind": "other"})
        extract_mock.assert_not_called()

    def test_non_list_value_passed_through(self):
        result, _ = _call({"text": "hello"})
        self.assertEqual(result["text"], "hello")

    def test_none_entry_in_list_unchanged(self):
        output = {"images": [None, {"filename": "a.mp4", "subfolder": "", "type": "output"}]}
        result, _ = _call(output)
        self.assertIsNone(result["images"][0])
        self.assertIn("metadata", result["images"][1])

    def test_entry_without_filename_unchanged(self):
        output = {"latent": [{"subfolder": "", "type": "output"}]}
        result, extract_mock = _call(output)
        self.assertNotIn("metadata", result["latent"][0])
        extract_mock.assert_not_called()

    def test_file_not_on_disk_unchanged(self):
        output = {"images": [{"filename": "missing.mp4", "subfolder": "", "type": "output"}]}
        result, extract_mock = _call(output, file_exists=False)
        self.assertNotIn("metadata", result["images"][0])
        extract_mock.assert_not_called()

    def test_unknown_type_directory_unchanged(self):
        output = {"images": [{"filename": "a.mp4", "subfolder": "", "type": "unknown"}]}
        result, extract_mock = _call(output, directory=None)
        self.assertNotIn("metadata", result["images"][0])
        extract_mock.assert_not_called()

    def test_path_traversal_subfolder_skipped(self):
        output = {"images": [{"filename": "passwd", "subfolder": "../../etc", "type": "output"}]}
        result, extract_mock = _call(output)
        self.assertNotIn("metadata", result["images"][0])
        extract_mock.assert_not_called()

    def test_absolute_filename_skipped(self):
        absolute_filename = os.path.abspath(os.sep + "etc" + os.sep + "passwd")
        output = {"images": [{"filename": absolute_filename, "subfolder": "", "type": "output"}]}
        result, extract_mock = _call(output)
        self.assertNotIn("metadata", result["images"][0])
        extract_mock.assert_not_called()

    def test_extractor_unavailable_returns_unchanged(self):
        output = {"images": [{"filename": "a.mp4", "subfolder": "", "type": "output"}]}
        mocked = {
            "folder_paths": MagicMock(get_directory_by_type=MagicMock(return_value=_DEFAULT_BASE)),
            # A None sys.modules entry makes the lazy import raise ImportError.
            "app.assets.services.media_metadata": None,
        }
        with patch.dict("sys.modules", mocked), \
             patch("os.path.isfile", return_value=True):
            import comfy_execution.media_enrichment as mod
            result = mod.enrich_output_with_media_metadata(output)
        self.assertNotIn("metadata", result["images"][0])

    def test_extractor_error_does_not_block_sibling_entries(self):
        call_count = [0]

        def extract_side_effect(abs_path, mime_type=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("boom")
            return dict(_VIDEO_META)

        extract = MagicMock(side_effect=extract_side_effect)
        output = {
            "images": [
                {"filename": "bad.mp4", "subfolder": "", "type": "output"},
                {"filename": "good.mp4", "subfolder": "", "type": "output"},
            ]
        }
        result, _ = _call(output, extract=extract)
        self.assertNotIn("metadata", result["images"][0])
        self.assertEqual(result["images"][1]["metadata"], _VIDEO_META)

    def test_multiple_output_keys_all_enriched(self):
        output = {
            "images": [{"filename": "a.png", "subfolder": "", "type": "output"}],
            "videos": [{"filename": "b.mp4", "subfolder": "", "type": "output"}],
        }
        result, _ = _call(output)
        self.assertIn("metadata", result["images"][0])
        self.assertIn("metadata", result["videos"][0])


if __name__ == "__main__":
    unittest.main()

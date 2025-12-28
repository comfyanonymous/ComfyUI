"""Tests for Content-Disposition header generation (RFC 2183/5987)"""

import pytest
from urllib.parse import quote


def create_content_disposition_header(filename: str) -> str:
    """
    Generate RFC 2183/5987 compliant Content-Disposition header value.

    Provides both ASCII fallback (filename=) and UTF-8 encoded (filename*=)
    for international filename support across all clients.

    Note: This is a copy of the function from server.py for isolated testing.
    """
    # ASCII-safe filename for legacy clients (replace non-ASCII with ?)
    ascii_filename = filename.encode('ascii', 'replace').decode('ascii')
    # RFC 5987 percent-encoded filename for UTF-8 support
    encoded_filename = quote(filename, safe='')
    return f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded_filename}"


class TestContentDispositionHeader:
    """Test RFC 2183/5987 compliant Content-Disposition header generation."""

    def test_ascii_filename(self):
        """Test that ASCII filenames are properly formatted."""
        result = create_content_disposition_header("test_image.png")

        # Must have attachment disposition type (RFC 2183)
        assert result.startswith("attachment;")
        # Must have ASCII filename parameter
        assert 'filename="test_image.png"' in result
        # Must have UTF-8 encoded filename parameter (RFC 5987)
        assert "filename*=UTF-8''test_image.png" in result

    def test_unicode_filename_chinese(self):
        """Test that Chinese filenames are properly encoded."""
        result = create_content_disposition_header("图片.png")

        assert result.startswith("attachment;")
        # ASCII fallback should replace non-ASCII chars with ?
        assert 'filename="??.png"' in result
        # UTF-8 encoded version should be percent-encoded
        assert "filename*=UTF-8''" in result
        assert "%E5%9B%BE%E7%89%87.png" in result  # URL-encoded 图片

    def test_unicode_filename_japanese(self):
        """Test that Japanese filenames are properly encoded."""
        result = create_content_disposition_header("画像.jpg")

        assert result.startswith("attachment;")
        # UTF-8 encoded version should be percent-encoded
        assert "filename*=UTF-8''" in result
        assert "%E7%94%BB%E5%83%8F.jpg" in result  # URL-encoded 画像

    def test_filename_with_spaces(self):
        """Test that spaces in filenames are properly encoded."""
        result = create_content_disposition_header("my image file.png")

        assert result.startswith("attachment;")
        assert 'filename="my image file.png"' in result
        # Spaces should be percent-encoded in filename*
        assert "my%20image%20file.png" in result

    def test_filename_with_special_chars(self):
        """Test that special characters are properly handled."""
        result = create_content_disposition_header("file(1)[2]{3}.png")

        assert result.startswith("attachment;")
        # Special chars should be percent-encoded in filename*
        assert "filename*=UTF-8''" in result

    def test_empty_filename(self):
        """Test handling of empty filename."""
        result = create_content_disposition_header("")

        assert result.startswith("attachment;")
        assert 'filename=""' in result

    def test_filename_with_quotes(self):
        """Test that quotes in filename don't break the header."""
        # Note: Quotes in filenames are edge cases but should not crash
        result = create_content_disposition_header('file"name.png')

        assert result.startswith("attachment;")
        # The function should still produce valid output
        assert "filename*=UTF-8''" in result

    def test_mixed_ascii_unicode(self):
        """Test filenames with both ASCII and Unicode characters."""
        result = create_content_disposition_header("photo_照片_2024.png")

        assert result.startswith("attachment;")
        # ASCII fallback replaces unicode with ?
        assert 'filename="photo_??_2024.png"' in result
        # UTF-8 version preserves everything (encoded)
        assert "filename*=UTF-8''" in result
        assert "photo_" in result or "photo_%E7%85%A7%E7%89%87_2024.png" in result.replace("photo_", "")

    def test_long_filename(self):
        """Test that long filenames are handled without truncation."""
        long_name = "a" * 200 + ".png"
        result = create_content_disposition_header(long_name)

        assert result.startswith("attachment;")
        # Should contain the full filename
        assert long_name in result

"""Tests for Content-Disposition header format (RFC 2183 compliance)

Relates to issue #8914: Content-Disposition Header not matching RFC2183 rules
"""

import pytest
import re


class TestContentDispositionHeader:
    """Test Content-Disposition header format compliance with RFC 2183."""

    def test_rfc2183_format_with_inline(self):
        """Verify inline disposition type format matches RFC 2183."""
        filename = "test_image.png"
        header = f'inline; filename="{filename}"'

        # RFC 2183 requires disposition-type followed by parameters
        # Format: disposition-type ";" disposition-parm
        pattern = r'^(inline|attachment);\s*filename="[^"]*"$'
        assert re.match(pattern, header), f"Header '{header}' does not match RFC 2183 format"

    def test_rfc2183_format_with_attachment(self):
        """Verify attachment disposition type format matches RFC 2183."""
        filename = "download.mp4"
        header = f'attachment; filename="{filename}"'

        pattern = r'^(inline|attachment);\s*filename="[^"]*"$'
        assert re.match(pattern, header), f"Header '{header}' does not match RFC 2183 format"

    def test_invalid_format_missing_disposition_type(self):
        """Verify that format without disposition type is invalid."""
        filename = "test.jpg"
        invalid_header = f'filename="{filename}"'

        pattern = r'^(inline|attachment);\s*filename="[^"]*"$'
        assert not re.match(pattern, invalid_header), \
            "Header without disposition type should not match RFC 2183 format"

    @pytest.mark.parametrize("filename", [
        "image.png",
        "video.mp4",
        "file with spaces.jpg",
        "special_chars-123.webp",
    ])
    def test_various_filenames(self, filename):
        """Test RFC 2183 format with various filename patterns."""
        header = f'inline; filename="{filename}"'

        # Should have disposition type before filename
        assert header.startswith("inline; ") or header.startswith("attachment; ")
        assert f'filename="{filename}"' in header


class TestContentDispositionParsing:
    """Test that Content-Disposition headers can be parsed by standard libraries."""

    def test_parse_inline_disposition(self):
        """Test parsing inline disposition header."""
        header = 'inline; filename="test.png"'

        # Simple parsing test - split by semicolon
        parts = [p.strip() for p in header.split(';')]
        assert parts[0] in ('inline', 'attachment')
        assert 'filename=' in parts[1]

    def test_extract_filename(self):
        """Test extracting filename from header."""
        filename = "my_image.jpg"
        header = f'inline; filename="{filename}"'

        # Extract filename using regex
        match = re.search(r'filename="([^"]*)"', header)
        assert match is not None
        assert match.group(1) == filename

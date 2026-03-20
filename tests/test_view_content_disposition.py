"""
Tests for the /view endpoint Content-Disposition header fix.

Verifies that the Content-Disposition header complies with RFC 2183
by using `attachment; filename="..."` format instead of just `filename="..."`.

See: https://github.com/Comfy-Org/ComfyUI/issues/8914
"""

import pytest
import subprocess
import time
import os
import tempfile
import struct
import pytest

# Simple 1x1 PNG image (minimal valid PNG)
MINIMAL_PNG = bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk length + type
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 dimensions
    0x08, 0x02, 0x00, 0x00, 0x00,                     # bit depth, color type, etc
    0x90, 0x77, 0x53, 0xDE,                           # IHDR CRC
    0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54,  # IDAT chunk
    0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0xFF, 0x00,  # image data
    0x05, 0xFE, 0x02, 0xFE,                           # IDAT CRC
    0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44,  # IEND chunk
    0xAE, 0x42, 0x60, 0x82,                           # IEND CRC
])


@pytest.mark.execution
class TestViewContentDisposition:
    """Test suite for Content-Disposition header in /view endpoint."""

    @pytest.fixture(scope="class", autouse=True)
    def output_dir(self, args_pytest):
        """Use the test output directory."""
        return args_pytest["output_dir"]

    @pytest.fixture(scope="class", autouse=True)
    def _server(self, args_pytest):
        """Start ComfyUI server for testing."""
        pargs = [
            'python', 'main.py',
            '--output-directory', args_pytest["output_dir"],
            '--listen', args_pytest["listen"],
            '--port', str(args_pytest["port"]),
            '--cpu',
        ]
        p = subprocess.Popen(pargs, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        yield
        p.kill()
        p.wait()

    @pytest.fixture(scope="class")
    def server_url(self, args_pytest):
        """Get the server base URL."""
        return f"http://{args_pytest['listen']}:{args_pytest['port']}"

    @pytest.fixture(scope="class")
    def client(self, server_url, _server):
        """Create HTTP client with retry."""
        import requests
        session = requests.Session()
        n_tries = 10
        for i in range(n_tries):
            time.sleep(2)
            try:
                resp = session.get(f"{server_url}/system_stats", timeout=5)
                if resp.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i == n_tries - 1:
                    raise
        return session

    @pytest.fixture
    def test_image_file(self, output_dir):
        """Create a temporary test image file and clean it up after the test."""
        filename = "test_content_disposition.png"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(MINIMAL_PNG)
        yield filename, filepath
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)

    def test_view_content_disposition_is_attachment_format(self, client, server_url, test_image_file):
        """Test that /view endpoint returns Content-Disposition with RFC 2183 compliant format.

        The header should be: Content-Disposition: attachment; filename="test_content_disposition.png"
        NOT: Content-Disposition: filename="test_content_disposition.png"

        This test validates the fix for: https://github.com/Comfy-Org/ComfyUI/issues/8914
        """
        filename, _ = test_image_file
        response = client.get(
            f"{server_url}/view",
            params={"filename": filename},
            timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        content_disposition = response.headers.get("Content-Disposition", "")
        assert content_disposition, "Content-Disposition header should be present"

        # RFC 2183 compliant format: "attachment; filename=\"name.ext\""
        expected_prefix = f"attachment; filename=\"{filename}\""
        assert content_disposition == expected_prefix, (
            f"Content-Disposition header should be RFC 2183 compliant.\n"
            f"Expected: {expected_prefix}\n"
            f"Got: {content_disposition}"
        )

    def test_view_with_subfolder_content_disposition(self, client, server_url, output_dir):
        """Test Content-Disposition with subfolder parameter."""
        import requests

        # Create a subfolder and test image
        subfolder = "test_subfolder"
        subfolder_path = os.path.join(output_dir, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

        filename = "test_subfolder.png"
        filepath = os.path.join(subfolder_path, filename)
        with open(filepath, 'wb') as f:
            f.write(MINIMAL_PNG)

        try:
            response = client.get(
                f"{server_url}/view",
                params={"filename": filename, "subfolder": subfolder},
                timeout=10
            )

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"

            content_disposition = response.headers.get("Content-Disposition", "")
            assert content_disposition, "Content-Disposition header should be present"

            expected_prefix = f"attachment; filename=\"{filename}\""
            assert content_disposition == expected_prefix, (
                f"Content-Disposition header should be RFC 2183 compliant.\n"
                f"Expected: {expected_prefix}\n"
                f"Got: {content_disposition}"
            )
        finally:
            # Cleanup
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(subfolder_path):
                os.rmdir(subfolder_path)

"""Tests for request_logger credential masking.

The API request logger writes request/response details to persistent plaintext
files in the temp directory. Without masking, the Authorization header (which
carries the user's Comfy API bearer token) is written verbatim to every log
file. These tests verify that sensitive headers are masked before logging.
"""

import os


# Import the logger module directly to avoid pulling heavy deps
# (torch, comfy_aimdo) through comfy_api_nodes.util.__init__
_logger_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "comfy_api_nodes",
    "util",
    "request_logger.py",
)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "request_logger", os.path.abspath(_logger_path)
)
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)

_mask_sensitive_headers = rl._mask_sensitive_headers
_format_data_for_logging = rl._format_data_for_logging
log_request_response = rl.log_request_response


class TestMaskSensitiveHeaders:
    def test_authorization_header_masked(self):
        """The Authorization header must be replaced with '***'."""
        headers = {
            "Authorization": "Bearer sk-secret-12345",
            "Accept": "application/json",
        }
        masked = _mask_sensitive_headers(headers)
        assert masked["Authorization"] == "***"
        assert masked["Accept"] == "application/json"

    def test_case_insensitive_masking(self):
        """Header name matching must be case-insensitive (HTTP headers are case-insensitive)."""
        headers = {"authorization": "Bearer token", "AUTHORIZATION": "Bearer token"}
        masked = _mask_sensitive_headers(headers)
        for key in masked:
            assert masked[key] == "***"

    def test_x_api_key_masked(self):
        headers = {"X-API-Key": "key-abc123", "Content-Type": "application/json"}
        masked = _mask_sensitive_headers(headers)
        assert masked["X-API-Key"] == "***"
        assert masked["Content-Type"] == "application/json"

    def test_cookie_masked(self):
        headers = {"Cookie": "session=abc123", "Set-Cookie": "token=xyz"}
        masked = _mask_sensitive_headers(headers)
        assert masked["Cookie"] == "***"
        assert masked["Set-Cookie"] == "***"

    def test_non_sensitive_headers_unchanged(self):
        """Headers that are NOT in the sensitive set must pass through unchanged."""
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI",
        }
        masked = _mask_sensitive_headers(headers)
        assert masked == headers

    def test_none_input(self):
        assert _mask_sensitive_headers(None) is None

    def test_empty_dict(self):
        assert _mask_sensitive_headers({}) == {}

    def test_original_dict_not_mutated(self):
        """The original headers dict must not be modified (returns a copy)."""
        headers = {"Authorization": "Bearer secret"}
        _mask_sensitive_headers(headers)
        assert headers["Authorization"] == "Bearer secret"


class TestLogRequestResponseMasking:
    def test_auth_token_not_in_log_file(self, tmp_path, monkeypatch):
        """When log_request_response writes to disk, the auth token must NOT appear."""
        log_dir = str(tmp_path / "api_logs")
        os.makedirs(log_dir, exist_ok=True)
        monkeypatch.setattr(rl, "get_log_directory", lambda: log_dir)

        headers_with_token = {
            "Authorization": "Bearer sk-test-token-99999",
            "Accept": "application/json",
        }

        rl.log_request_response(
            operation_id="POST_test",
            request_method="POST",
            request_url="https://api.comfy.org/v1/test",
            request_headers=headers_with_token,
            request_data={"prompt": "test"},
            response_status_code=200,
            response_content={"status": "ok"},
        )

        # Read the log file
        log_files = os.listdir(log_dir)
        assert len(log_files) == 1
        content = open(os.path.join(log_dir, log_files[0])).read()

        # The token must NOT appear in the log
        assert "sk-test-token-99999" not in content
        assert "Bearer" not in content
        # But the masked placeholder should be present
        assert "Authorization" in content
        assert "***" in content

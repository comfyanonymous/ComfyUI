"""Tests for WebDirectoryProxy — allow-list, traversal prevention, content serving."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from comfy.isolation.proxies.web_directory_proxy import WebDirectoryProxy


@pytest.fixture()
def web_dir_with_mixed_files(tmp_path: Path) -> Path:
    """Create a temp web directory with allowed and disallowed file types."""
    web = tmp_path / "web"
    js_dir = web / "js"
    js_dir.mkdir(parents=True)

    # Allowed types
    (js_dir / "app.js").write_text("console.log('hello');")
    (web / "index.html").write_text("<html></html>")
    (web / "style.css").write_text("body { margin: 0; }")

    # Disallowed types
    (web / "backdoor.py").write_text("import os; os.system('rm -rf /')")
    (web / "malware.exe").write_bytes(b"\x00" * 16)
    (web / "exploit.sh").write_text("#!/bin/bash\nrm -rf /")

    return web


@pytest.fixture()
def proxy_with_web_dir(web_dir_with_mixed_files: Path) -> WebDirectoryProxy:
    """Create a WebDirectoryProxy with a registered test web directory."""
    proxy = WebDirectoryProxy()
    # Clear class-level state to avoid cross-test pollution
    WebDirectoryProxy._web_dirs = {}
    WebDirectoryProxy.register_web_dir("test-extension", str(web_dir_with_mixed_files))
    return proxy


class TestAllowList:
    """AC-2: list_web_files returns only allowed file types."""

    def test_allowlist_only_safe_types(
        self, proxy_with_web_dir: WebDirectoryProxy
    ) -> None:
        files = proxy_with_web_dir.list_web_files("test-extension")
        extensions = {Path(f["relative_path"]).suffix for f in files}

        # Only .js, .html, .css should appear
        assert extensions == {".js", ".html", ".css"}

    def test_allowlist_excludes_dangerous_types(
        self, proxy_with_web_dir: WebDirectoryProxy
    ) -> None:
        files = proxy_with_web_dir.list_web_files("test-extension")
        paths = [f["relative_path"] for f in files]

        assert not any(p.endswith(".py") for p in paths)
        assert not any(p.endswith(".exe") for p in paths)
        assert not any(p.endswith(".sh") for p in paths)

    def test_allowlist_correct_count(
        self, proxy_with_web_dir: WebDirectoryProxy
    ) -> None:
        files = proxy_with_web_dir.list_web_files("test-extension")
        # 3 allowed files: app.js, index.html, style.css
        assert len(files) == 3

    def test_allowlist_unknown_extension_returns_empty(
        self, proxy_with_web_dir: WebDirectoryProxy
    ) -> None:
        files = proxy_with_web_dir.list_web_files("nonexistent-extension")
        assert files == []


class TestTraversal:
    """AC-3: get_web_file rejects directory traversal attempts."""

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "../../../etc/passwd",
            "/etc/passwd",
            "../../__init__.py",
        ],
    )
    def test_traversal_rejected(
        self, proxy_with_web_dir: WebDirectoryProxy, malicious_path: str
    ) -> None:
        with pytest.raises(ValueError):
            proxy_with_web_dir.get_web_file("test-extension", malicious_path)


class TestContent:
    """AC-4: get_web_file returns base64 content with correct MIME types."""

    def test_content_js_mime_type(
        self, proxy_with_web_dir: WebDirectoryProxy
    ) -> None:
        result = proxy_with_web_dir.get_web_file("test-extension", "js/app.js")
        assert result["content_type"] == "application/javascript"

    def test_content_html_mime_type(
        self, proxy_with_web_dir: WebDirectoryProxy
    ) -> None:
        result = proxy_with_web_dir.get_web_file("test-extension", "index.html")
        assert result["content_type"] == "text/html"

    def test_content_css_mime_type(
        self, proxy_with_web_dir: WebDirectoryProxy
    ) -> None:
        result = proxy_with_web_dir.get_web_file("test-extension", "style.css")
        assert result["content_type"] == "text/css"

    def test_content_base64_roundtrip(
        self, proxy_with_web_dir: WebDirectoryProxy, web_dir_with_mixed_files: Path
    ) -> None:
        result = proxy_with_web_dir.get_web_file("test-extension", "js/app.js")
        decoded = base64.b64decode(result["content"])
        source = (web_dir_with_mixed_files / "js" / "app.js").read_bytes()
        assert decoded == source

    def test_content_disallowed_type_rejected(
        self, proxy_with_web_dir: WebDirectoryProxy
    ) -> None:
        with pytest.raises(ValueError, match="Disallowed file type"):
            proxy_with_web_dir.get_web_file("test-extension", "backdoor.py")

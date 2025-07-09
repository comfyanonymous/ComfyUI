"""
Unit tests for the mobile interface functionality.
"""
import pytest
import os


class TestMobileInterfaceFiles:
    """Test cases for mobile interface file structure and content."""

    def test_mobile_html_structure(self):
        """Test that mobile HTML has required structure."""
        mobile_html = os.path.join(
            os.path.dirname(__file__), "..", "web_mobile", "index.html"
        )

        if not os.path.exists(mobile_html):
            pytest.skip("Mobile HTML file not found")

        with open(mobile_html, 'r') as f:
            content = f.read()

        # Check essential HTML structure
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "<title>ComfyUI Mobile</title>" in content

        # Check required elements
        assert 'id="app"' in content
        assert 'id="workflowPipeline"' in content
        assert 'id="queuePrompt"' in content
        assert 'id="nodeDetailModal"' in content
        assert 'class="action-bar"' in content

        # Check CSS and JS includes
        assert "/mobile_static/styles.css" in content
        assert "/mobile_static/utils.js" in content
        assert "/mobile_static/app.js" in content
        assert "/mobile_static/mobile-interface.js" in content

    def test_mobile_js_files_exist(self):
        """Test that all required JavaScript files exist."""
        mobile_dir = os.path.join(os.path.dirname(__file__), "..", "web_mobile")

        if not os.path.exists(mobile_dir):
            pytest.skip("Mobile directory not found")

        required_js_files = [
            "utils.js",
            "graph-linearization.js",
            "api-client.js",
            "mobile-interface.js",
            "app.js"
        ]

        for js_file in required_js_files:
            file_path = os.path.join(mobile_dir, js_file)
            assert os.path.exists(file_path), f"Missing required JS file: {js_file}"

            # Check file is not empty
            with open(file_path, 'r') as f:
                content = f.read().strip()
                assert len(content) > 0, f"JS file is empty: {js_file}"

    def test_mobile_css_file_exists(self):
        """Test that the CSS file exists and has content."""
        css_file = os.path.join(
            os.path.dirname(__file__), "..", "web_mobile", "styles.css"
        )

        if not os.path.exists(css_file):
            pytest.skip("CSS file not found")

        with open(css_file, 'r') as f:
            content = f.read().strip()

        assert len(content) > 0, "CSS file is empty"

        # Check for essential CSS classes
        assert ".header" in content
        assert ".workflow-pipeline" in content
        assert ".node-card" in content
        assert ".action-bar" in content
        assert ".modal" in content

        # Check for mobile-specific CSS
        assert "@media" in content, "No responsive CSS found"
        assert "touch" in content.lower(), "No touch-specific CSS found"


class TestMobileInterfaceJavaScript:
    """Test cases for mobile interface JavaScript functionality."""

    def test_javascript_classes_defined(self):
        """Test that required JavaScript classes are defined."""
        js_files = {
            "utils.js": ["Utils"],
            "graph-linearization.js": ["GraphLinearization"],
            "api-client.js": ["ComfyUIAPIClient"],
            "mobile-interface.js": ["MobileInterface"],
            "app.js": ["MobileApp"]
        }

        mobile_dir = os.path.join(os.path.dirname(__file__), "..", "web_mobile")

        for js_file, expected_classes in js_files.items():
            file_path = os.path.join(mobile_dir, js_file)

            if not os.path.exists(file_path):
                pytest.skip(f"JS file not found: {js_file}")

            with open(file_path, 'r') as f:
                content = f.read()

            for class_name in expected_classes:
                assert f"class {class_name}" in content, \
                    f"Class {class_name} not found in {js_file}"

    def test_javascript_exports(self):
        """Test that JavaScript files export classes to window."""
        js_files = {
            "utils.js": "Utils",
            "graph-linearization.js": "GraphLinearization",
            "api-client.js": "ComfyUIAPIClient",
            "mobile-interface.js": "MobileInterface"
        }

        mobile_dir = os.path.join(os.path.dirname(__file__), "..", "web_mobile")

        for js_file, class_name in js_files.items():
            file_path = os.path.join(mobile_dir, js_file)

            if not os.path.exists(file_path):
                pytest.skip(f"JS file not found: {js_file}")

            with open(file_path, 'r') as f:
                content = f.read()

            assert f"window.{class_name}" in content, \
                f"Class {class_name} not exported to window in {js_file}"

    def test_mobile_interface_initialization(self):
        """Test that mobile interface has proper initialization."""
        app_js = os.path.join(
            os.path.dirname(__file__), "..", "web_mobile", "app.js"
        )

        if not os.path.exists(app_js):
            pytest.skip("app.js not found")

        with open(app_js, 'r') as f:
            content = f.read()

        # Check for DOM ready event listener
        assert "DOMContentLoaded" in content

        # Check for MobileApp initialization
        assert "new MobileApp()" in content

        # Check for error handling
        assert "try" in content and "catch" in content

        # Check for proper cleanup
        assert "beforeunload" in content
        assert "cleanup" in content


if __name__ == "__main__":
    pytest.main([__file__])

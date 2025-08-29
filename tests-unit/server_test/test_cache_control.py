"""Tests for server cache control middleware"""

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from middleware.cache_middleware import cache_control, ONE_HOUR, ONE_DAY, IMG_EXTENSIONS

pytestmark = pytest.mark.asyncio  # Apply asyncio mark to all tests


class TestCacheControl:
    """Test cache control middleware functionality"""

    @pytest.fixture
    def mock_handler(self):
        """Create a mock handler that returns a response with 200 status"""

        async def handler(request):
            return web.Response(status=200)

        return handler

    @pytest.fixture
    def mock_handler_404(self):
        """Create a mock handler that returns a 404 response"""

        async def handler(request):
            return web.Response(status=404)

        return handler

    async def test_image_extensions_200_status(self, mock_handler):
        """Test that images with 200 status get 24-hour cache"""
        for ext in IMG_EXTENSIONS:
            request = make_mocked_request("GET", f"/test{ext}")
            response = await cache_control(request, mock_handler)

            assert response.status == 200
            assert "Cache-Control" in response.headers
            assert response.headers["Cache-Control"] == f"public, max-age={ONE_DAY}"

    async def test_image_extensions_404_status(self, mock_handler_404):
        """Test that images with 404 status get 1-hour cache"""
        request = make_mocked_request("GET", "/missing.jpg")
        response = await cache_control(request, mock_handler_404)

        assert response.status == 404
        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == f"public, max-age={ONE_HOUR}"

    async def test_case_insensitive_image_extension(self, mock_handler):
        """Test that image extensions are matched case-insensitively"""
        test_paths = ["/image.JPG", "/photo.PNG", "/pic.JpEg"]

        for path in test_paths:
            request = make_mocked_request("GET", path)
            response = await cache_control(request, mock_handler)

            assert "Cache-Control" in response.headers
            assert response.headers["Cache-Control"] == f"public, max-age={ONE_DAY}"

    async def test_js_files_no_cache(self, mock_handler):
        """Test that .js files get no-cache header"""
        request = make_mocked_request("GET", "/script.js")
        response = await cache_control(request, mock_handler)

        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == "no-cache"

    async def test_css_files_no_cache(self, mock_handler):
        """Test that .css files get no-cache header"""
        request = make_mocked_request("GET", "/styles.css")
        response = await cache_control(request, mock_handler)

        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == "no-cache"

    async def test_index_json_no_cache(self, mock_handler):
        """Test that index.json gets no-cache header"""
        request = make_mocked_request("GET", "/api/index.json")
        response = await cache_control(request, mock_handler)

        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == "no-cache"

    async def test_js_css_preserves_existing_headers(self):
        """Test that .js/.css files preserve existing Cache-Control headers"""

        async def handler_with_cache(request):
            return web.Response(status=200, headers={"Cache-Control": "max-age=3600"})

        request = make_mocked_request("GET", "/script.js")
        response = await cache_control(request, handler_with_cache)

        # setdefault should preserve existing header
        assert response.headers["Cache-Control"] == "max-age=3600"

    async def test_image_preserves_existing_headers(self):
        """Test that image cache headers preserve existing Cache-Control"""

        async def handler_with_cache(request):
            return web.Response(
                status=200, headers={"Cache-Control": "private, no-cache"}
            )

        request = make_mocked_request("GET", "/image.jpg")
        response = await cache_control(request, handler_with_cache)

        # setdefault should preserve existing header
        assert response.headers["Cache-Control"] == "private, no-cache"

    async def test_non_matching_files_unchanged(self, mock_handler):
        """Test that non-matching files don't get cache headers"""
        test_paths = ["/index.html", "/data.txt", "/api/endpoint", "/file.pdf"]

        for path in test_paths:
            request = make_mocked_request("GET", path)
            response = await cache_control(request, mock_handler)

            assert "Cache-Control" not in response.headers

    async def test_query_strings_ignored(self, mock_handler):
        """Test that query strings don't affect image detection"""
        request = make_mocked_request("GET", "/image.jpg?v=123&size=large")
        response = await cache_control(request, mock_handler)

        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == f"public, max-age={ONE_DAY}"

    async def test_multiple_dots_in_path(self, mock_handler):
        """Test files with multiple dots still match correctly"""
        request = make_mocked_request("GET", "/image.min.jpg")
        response = await cache_control(request, mock_handler)

        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == f"public, max-age={ONE_DAY}"

    async def test_2xx_success_statuses_get_long_cache(self):
        """Test that all 2xx success statuses get 24-hour cache for images"""
        success_statuses = [200, 201, 202, 204, 206]

        for status in success_statuses:

            async def handler_success(request):
                return web.Response(status=status)

            request = make_mocked_request("GET", "/success.jpg")
            response = await cache_control(request, handler_success)

            assert response.status == status
            assert "Cache-Control" in response.headers
            assert response.headers["Cache-Control"] == f"public, max-age={ONE_DAY}"

    async def test_permanent_redirects_get_long_cache(self):
        """Test that permanent redirects (301, 308) get 24-hour cache for images"""
        permanent_redirects = [301, 308]

        for status in permanent_redirects:

            async def handler_redirect(request):
                return web.Response(status=status)

            request = make_mocked_request("GET", "/permanent.jpg")
            response = await cache_control(request, handler_redirect)

            assert response.status == status
            assert "Cache-Control" in response.headers
            assert response.headers["Cache-Control"] == f"public, max-age={ONE_DAY}"

    async def test_temporary_redirects_get_no_cache(self):
        """Test that temporary redirects (302, 303, 307) get no-cache for images"""
        temporary_redirects = [302, 303, 307]

        for status in temporary_redirects:

            async def handler_redirect(request):
                return web.Response(status=status)

            request = make_mocked_request("GET", "/temporary.png")
            response = await cache_control(request, handler_redirect)

            assert response.status == status
            assert "Cache-Control" in response.headers
            assert response.headers["Cache-Control"] == "no-cache"

    async def test_304_not_modified_inherits_cache(self):
        """Test that 304 Not Modified doesn't set cache headers for images"""

        async def handler_304(request):
            return web.Response(status=304, headers={"Cache-Control": "max-age=7200"})

        request = make_mocked_request("GET", "/not-modified.jpg")
        response = await cache_control(request, handler_304)

        assert response.status == 304
        # Should preserve existing cache header, not override
        assert response.headers["Cache-Control"] == "max-age=7200"

    async def test_all_image_extensions(self, mock_handler):
        """Test all defined image extensions are handled"""
        expected_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
        )
        assert IMG_EXTENSIONS == expected_extensions

        for ext in IMG_EXTENSIONS:
            request = make_mocked_request("GET", f"/image{ext}")
            response = await cache_control(request, mock_handler)

            assert "Cache-Control" in response.headers
            assert response.headers["Cache-Control"] == f"public, max-age={ONE_DAY}"

    async def test_nested_paths_with_images(self, mock_handler):
        """Test that images in nested paths are handled correctly"""
        test_paths = [
            "/static/images/photo.jpg",
            "/assets/img/banner.png",
            "/uploads/2024/12/image.webp",
        ]

        for path in test_paths:
            request = make_mocked_request("GET", path)
            response = await cache_control(request, mock_handler)

            assert "Cache-Control" in response.headers
            assert response.headers["Cache-Control"] == f"public, max-age={ONE_DAY}"

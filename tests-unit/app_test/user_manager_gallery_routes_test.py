import pytest
import os
import shutil
import json
import time
from unittest import IsolatedAsyncioTestCase # Using IsolatedAsyncioTestCase for async test methods
from unittest.mock import patch, MagicMock

from aiohttp import web

# Assuming UserManager is in app.user_manager
# Adjust the import path if your project structure is different
from app.user_manager import UserManager, GALLERY_SUFFIX

# Mock comfy.cli_args and folder_paths before they are imported by UserManager
# This is a common pattern if these modules are read at import time by the tested code.
mock_args = MagicMock()
mock_args.multi_user = False # Default to single-user mode for simplicity in most tests

mock_folder_paths = MagicMock()

# We'll set get_user_directory dynamically in test setup using tmp_path

# Apply patches at the module level if they need to be active before UserManager is imported
# or within specific test classes/methods if more fine-grained control is needed.
# For now, let's assume UserManager can be instantiated after these are patched.

@pytest.fixture
def app_client_factory(event_loop): # event_loop is a pytest-asyncio fixture
    """Factory to create aiohttp test clients."""
    async def _create_client(routes_def_func, *args_for_func):
        app = web.Application(loop=event_loop)
        routes = web.RouteTableDef()
        routes_def_func(routes, *args_for_func) # Call the function that defines routes
        app.add_routes(routes)
        return await event_loop.create_task(pytest.aiohttp.plugin.make_aiohttp_client(app))
    return _create_client


class TestUserManagerGalleryRoutes(IsolatedAsyncioTestCase):

    def setUp(self):
        # Create a temporary directory for user data
        self.test_user_dir_root = "temp_test_user_data"
        os.makedirs(self.test_user_dir_root, exist_ok=True)
        self.default_user_path = os.path.join(self.test_user_dir_root, "default")
        os.makedirs(self.default_user_path, exist_ok=True)

        # Patch folder_paths.get_user_directory and args
        self.patch_folder_paths = patch('app.user_manager.folder_paths', mock_folder_paths)
        self.patch_args = patch('app.user_manager.args', mock_args)

        self.mock_folder_paths = self.patch_folder_paths.start()
        self.mock_args = self.patch_args.start()

        self.mock_folder_paths.get_user_directory.return_value = self.test_user_dir_root
        self.mock_args.multi_user = False # Explicitly set for each test run

        self.user_manager = UserManager()

        # Setup routes for the user_manager
        self.app = web.Application()
        self.user_manager.add_routes(self.app.router)


    async def asyncSetUp(self):
        # Create a test client for making requests
        self.client = await pytest.aiohttp.plugin.make_aiohttp_client(self.app)


    async def asyncTearDown(self):
        await self.client.close() # Close the client
        self.patch_folder_paths.stop()
        self.patch_args.stop()
        if os.path.exists(self.test_user_dir_root):
            shutil.rmtree(self.test_user_dir_root)

    # --- Helper Methods ---
    def _create_file(self, filename, content="test", user="default", subdir=None):
        user_specific_path = os.path.join(self.test_user_dir_root, user)
        if subdir:
            user_specific_path = os.path.join(user_specific_path, subdir)
            os.makedirs(user_specific_path, exist_ok=True)

        filepath = os.path.join(user_specific_path, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    def _get_user_data_path(self, filename, user="default", subdir=None):
        user_specific_path = os.path.join(self.test_user_dir_root, user)
        if subdir:
            user_specific_path = os.path.join(user_specific_path, subdir)
        return os.path.join(user_specific_path, filename)

    # --- Test Cases for /gallery (GET) ---
    async def test_list_gallery_empty(self):
        resp = await self.client.get("/gallery")
        assert resp.status == 200
        data = await resp.json()
        assert data == []

    async def test_list_one_gallery_item(self):
        filename_orig = "image.png"
        filename_gallery = f"image{GALLERY_SUFFIX}.png"
        self._create_file(filename_gallery) # Create the .gallery.png file

        resp = await self.client.get("/gallery")
        assert resp.status == 200
        data = await resp.json()

        assert len(data) == 1
        item = data[0]
        assert item["filename"] == "image.png" # Original name without .gallery
        assert item["path"] == filename_gallery # Path includes .gallery
        assert item["size"] == 4 # "test"
        assert "modified" in item

    async def test_list_multiple_gallery_items_and_subdirs(self):
        self._create_file(f"img1{GALLERY_SUFFIX}.jpg")
        self._create_file(f"img2{GALLERY_SUFFIX}.jpeg", subdir="photos")
        self._create_file(f"document{GALLERY_SUFFIX}.pdf", subdir="docs/work")
        self._create_file("not_gallery.txt") # Should not be listed
        self._create_file(f"also_not_gallery{GALLERY_SUFFIX}") # No extension, should not match *.gallery.*
        self._create_file(f"another.gallery#fake.png") # Invalid char, but testing suffix rule

        resp = await self.client.get("/gallery")
        assert resp.status == 200
        data = await resp.json()

        assert len(data) == 3 # Only 3 valid gallery items
        filenames_found = sorted([item["filename"] for item in data])
        expected_filenames = sorted(["img1.jpg", "img2.jpeg", "document.pdf"])
        assert filenames_found == expected_filenames

        paths_found = sorted([item["path"] for item in data])
        expected_paths = sorted([
            f"img1{GALLERY_SUFFIX}.jpg",
            f"photos/img2{GALLERY_SUFFIX}.jpeg",
            f"docs/work/document{GALLERY_SUFFIX}.pdf"
        ])
        assert paths_found == expected_paths

    async def test_list_non_gallery_items_not_listed(self):
        self._create_file("textfile.txt")
        self._create_file(f"image_not_gallery.png") # No .gallery suffix in name
        self._create_file(f"image_with_gallery_suffix_only{GALLERY_SUFFIX}") # No further extension

        resp = await self.client.get("/gallery")
        assert resp.status == 200
        data = await resp.json()
        assert data == []

    # --- Test Cases for /userdata/{file}/gallery (POST) ---
    async def test_toggle_gallery_add(self):
        filename = "add_me.png"
        created_path = self._create_file(filename)

        resp = await self.client.post(f"/userdata/{filename}/gallery")
        assert resp.status == 200
        data = await resp.json()

        expected_new_filename = f"add_me{GALLERY_SUFFIX}.png"
        assert data["filename"] == expected_new_filename

        assert not os.path.exists(created_path)
        assert os.path.exists(self._get_user_data_path(expected_new_filename))

    async def test_toggle_gallery_add_with_subdir(self):
        filename = "add_me_subdir.jpg"
        subdir = "level1/level2"
        created_path = self._create_file(filename, subdir=subdir)

        # Path in URL needs to be URL encoded if it has slashes
        url_path = f"{subdir}/{filename}"

        resp = await self.client.post(f"/userdata/{url_path}/gallery")
        assert resp.status == 200
        data = await resp.json()

        expected_new_filename = f"add_me_subdir{GALLERY_SUFFIX}.jpg"
        assert data["filename"] == expected_new_filename # Response is basename

        assert not os.path.exists(created_path)
        assert os.path.exists(self._get_user_data_path(expected_new_filename, subdir=subdir))


    async def test_toggle_gallery_remove(self):
        original_filename_part = "remove_me"
        ext = ".jpeg"
        gallery_filename = f"{original_filename_part}{GALLERY_SUFFIX}{ext}"
        created_gallery_path = self._create_file(gallery_filename)

        resp = await self.client.post(f"/userdata/{gallery_filename}/gallery")
        assert resp.status == 200
        data = await resp.json()

        expected_new_filename = f"{original_filename_part}{ext}"
        assert data["filename"] == expected_new_filename

        assert not os.path.exists(created_gallery_path)
        assert os.path.exists(self._get_user_data_path(expected_new_filename))

    async def test_toggle_gallery_file_not_found(self):
        resp = await self.client.post("/userdata/nonexistentfile.png/gallery")
        assert resp.status == 404 # Or 400 if file not specified, but here it is specified
        data = await resp.json() # Assuming error responses are JSON
        assert "File not found" in data.get("error", "") or "File not found" in await resp.text()


    async def test_toggle_gallery_invalid_file_param(self):
        # Test with an empty file parameter or one that might be problematic
        # The route itself might catch this before user_manager logic if path is malformed
        # Depending on aiohttp's routing, this might result in a 404 for the route itself
        # or a 400 if the handler's file extraction fails.
        # UserManager's toggle_gallery_status expects `file` from `request.match_info`.
        # If `file` is empty, it returns 400 "File not specified".

        # This test is more about how aiohttp handles empty path parameters
        # For instance, a route like /userdata//gallery might not match or might pass an empty string.
        # Let's assume it passes an empty string if the route matches /userdata/{file}/gallery
        # For this, we'd need to register a route that can produce an empty 'file' match_info.
        # The current route definition /userdata/{file}/gallery will likely not match /userdata//gallery.
        # So, let's test the handler directly with a mock request if we want to ensure "File not specified".

        mock_request = MagicMock(spec=web.Request)
        mock_request.match_info = {} # No 'file'
        mock_request.headers = {} # For get_request_user_id

        # Mock get_request_user_id if it's called before file check
        # self.user_manager.get_request_user_id = MagicMock(return_value="default")

        response = await self.user_manager.toggle_gallery_status(mock_request)
        assert response.status == 400
        # text_response = await response.text() # Not needed if using response.text
        assert "File not specified" in response.text


if __name__ == "__main__":
    pytest.main()

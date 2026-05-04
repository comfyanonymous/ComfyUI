"""E2E Tests for System User Protection HTTP Endpoints

Tests cover:
- HTTP endpoint blocking: System Users cannot access /userdata (GET, POST, DELETE, move)
- User creation blocking: System User names cannot be created via POST /users
- Backward compatibility: Public Users work as before
- Custom node scenario: Internal API works while HTTP is blocked
- Structural security: get_public_user_directory() provides automatic protection
"""

import pytest
import os
from aiohttp import web
from app.user_manager import UserManager
from unittest.mock import patch
import folder_paths


@pytest.fixture
def mock_user_directory(tmp_path):
    """Create a temporary user directory."""
    original_dir = folder_paths.get_user_directory()
    folder_paths.set_user_directory(str(tmp_path))
    yield tmp_path
    folder_paths.set_user_directory(original_dir)


@pytest.fixture
def user_manager_multi_user(mock_user_directory):
    """Create UserManager in multi-user mode."""
    with patch('app.user_manager.args') as mock_args:
        mock_args.multi_user = True
        um = UserManager()
        # Add test users
        um.users = {"default": "default", "test_user_123": "Test User"}
        yield um


@pytest.fixture
def app_multi_user(user_manager_multi_user):
    """Create app with multi-user mode enabled."""
    app = web.Application()
    routes = web.RouteTableDef()
    user_manager_multi_user.add_routes(routes)
    app.add_routes(routes)
    return app


class TestSystemUserEndpointBlocking:
    """E2E tests for System User blocking on all HTTP endpoints.

    Verifies:
    - GET /userdata blocked for System Users
    - POST /userdata blocked for System Users
    - DELETE /userdata blocked for System Users
    - POST /userdata/.../move/... blocked for System Users
    """

    @pytest.mark.asyncio
    async def test_userdata_get_blocks_system_user(
        self, aiohttp_client, app_multi_user, mock_user_directory
    ):
        """
        GET /userdata with System User header should be blocked.
        """
        # Create test directory for System User (simulating internal creation)
        system_user_dir = mock_user_directory / "__system"
        system_user_dir.mkdir()
        (system_user_dir / "secret.txt").write_text("sensitive data")

        client = await aiohttp_client(app_multi_user)

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            # Attempt to access System User's data via HTTP
            resp = await client.get(
                "/userdata?dir=.",
                headers={"comfy-user": "__system"}
            )

        # Should be blocked (403 Forbidden or similar error)
        assert resp.status in [400, 403, 500], \
            f"System User access should be blocked, got {resp.status}"

    @pytest.mark.asyncio
    async def test_userdata_post_blocks_system_user(
        self, aiohttp_client, app_multi_user, mock_user_directory
    ):
        """
        POST /userdata with System User header should be blocked.
        """
        client = await aiohttp_client(app_multi_user)

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            resp = await client.post(
                "/userdata/test.txt",
                headers={"comfy-user": "__system"},
                data=b"malicious content"
            )

        assert resp.status in [400, 403, 500], \
            f"System User write should be blocked, got {resp.status}"

        # Verify no file was created
        assert not (mock_user_directory / "__system" / "test.txt").exists()

    @pytest.mark.asyncio
    async def test_userdata_delete_blocks_system_user(
        self, aiohttp_client, app_multi_user, mock_user_directory
    ):
        """
        DELETE /userdata with System User header should be blocked.
        """
        # Create a file in System User directory
        system_user_dir = mock_user_directory / "__system"
        system_user_dir.mkdir()
        secret_file = system_user_dir / "secret.txt"
        secret_file.write_text("do not delete")

        client = await aiohttp_client(app_multi_user)

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            resp = await client.delete(
                "/userdata/secret.txt",
                headers={"comfy-user": "__system"}
            )

        assert resp.status in [400, 403, 500], \
            f"System User delete should be blocked, got {resp.status}"

        # Verify file still exists
        assert secret_file.exists()

    @pytest.mark.asyncio
    async def test_v2_userdata_blocks_system_user(
        self, aiohttp_client, app_multi_user, mock_user_directory
    ):
        """
        GET /v2/userdata with System User header should be blocked.
        """
        client = await aiohttp_client(app_multi_user)

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            resp = await client.get(
                "/v2/userdata",
                headers={"comfy-user": "__system"}
            )

        assert resp.status in [400, 403, 500], \
            f"System User v2 access should be blocked, got {resp.status}"

    @pytest.mark.asyncio
    async def test_move_userdata_blocks_system_user(
        self, aiohttp_client, app_multi_user, mock_user_directory
    ):
        """
        POST /userdata/{file}/move/{dest} with System User header should be blocked.
        """
        system_user_dir = mock_user_directory / "__system"
        system_user_dir.mkdir()
        (system_user_dir / "source.txt").write_text("sensitive data")

        client = await aiohttp_client(app_multi_user)

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            resp = await client.post(
                "/userdata/source.txt/move/dest.txt",
                headers={"comfy-user": "__system"}
            )

        assert resp.status in [400, 403, 500], \
            f"System User move should be blocked, got {resp.status}"

        # Verify source file still exists (move was blocked)
        assert (system_user_dir / "source.txt").exists()


class TestSystemUserCreationBlocking:
    """E2E tests for blocking System User name creation via POST /users.

    Verifies:
    - POST /users returns 400 for System User name (not 500)
    """

    @pytest.mark.asyncio
    async def test_post_users_blocks_system_user_name(
        self, aiohttp_client, app_multi_user
    ):
        """POST /users with System User name should return 400 Bad Request."""
        client = await aiohttp_client(app_multi_user)

        resp = await client.post(
            "/users",
            json={"username": "__system"}
        )

        assert resp.status == 400, \
            f"System User creation should return 400, got {resp.status}"

    @pytest.mark.asyncio
    async def test_post_users_blocks_system_user_prefix_variations(
        self, aiohttp_client, app_multi_user
    ):
        """POST /users with any System User prefix variation should return 400 Bad Request."""
        client = await aiohttp_client(app_multi_user)

        system_user_names = ["__system", "__cache", "__config", "__anything"]

        for name in system_user_names:
            resp = await client.post("/users", json={"username": name})
            assert resp.status == 400, \
                f"System User name '{name}' should return 400, got {resp.status}"


class TestPublicUserStillWorks:
    """E2E tests for backward compatibility - Public Users should work as before.

    Verifies:
    - Public Users can access their data via HTTP
    - Public Users can create files via HTTP
    """

    @pytest.mark.asyncio
    async def test_public_user_can_access_userdata(
        self, aiohttp_client, app_multi_user, mock_user_directory
    ):
        """
        Public Users should still be able to access their data.
        """
        # Create test directory for Public User
        user_dir = mock_user_directory / "default"
        user_dir.mkdir()
        test_dir = user_dir / "workflows"
        test_dir.mkdir()
        (test_dir / "test.json").write_text('{"test": true}')

        client = await aiohttp_client(app_multi_user)

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            resp = await client.get(
                "/userdata?dir=workflows",
                headers={"comfy-user": "default"}
            )

        assert resp.status == 200
        data = await resp.json()
        assert "test.json" in data

    @pytest.mark.asyncio
    async def test_public_user_can_create_files(
        self, aiohttp_client, app_multi_user, mock_user_directory
    ):
        """
        Public Users should still be able to create files.
        """
        # Create user directory
        user_dir = mock_user_directory / "default"
        user_dir.mkdir()

        client = await aiohttp_client(app_multi_user)

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            resp = await client.post(
                "/userdata/newfile.txt",
                headers={"comfy-user": "default"},
                data=b"user content"
            )

        assert resp.status == 200
        assert (user_dir / "newfile.txt").exists()


class TestCustomNodeScenario:
    """Tests for custom node use case: internal API access vs HTTP blocking.

    Verifies:
    - Internal API (get_system_user_directory) works for custom nodes
    - HTTP endpoint cannot access data created via internal API
    """

    def test_internal_api_can_access_system_user(self, mock_user_directory):
        """
        Internal API (get_system_user_directory) should work for custom nodes.
        """
        # Custom node uses internal API
        system_path = folder_paths.get_system_user_directory("mynode_config")

        assert system_path is not None
        assert "__mynode_config" in system_path

        # Can create and write to System User directory
        os.makedirs(system_path, exist_ok=True)
        config_file = os.path.join(system_path, "settings.json")
        with open(config_file, "w") as f:
            f.write('{"api_key": "secret"}')

        assert os.path.exists(config_file)

    @pytest.mark.asyncio
    async def test_http_cannot_access_internal_data(
        self, aiohttp_client, app_multi_user, mock_user_directory
    ):
        """
        HTTP endpoint cannot access data created via internal API.
        """
        # Custom node creates data via internal API
        system_path = folder_paths.get_system_user_directory("mynode_config")
        os.makedirs(system_path, exist_ok=True)
        with open(os.path.join(system_path, "secret.json"), "w") as f:
            f.write('{"api_key": "secret"}')

        client = await aiohttp_client(app_multi_user)

        # Attacker tries to access via HTTP
        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            resp = await client.get(
                "/userdata/secret.json",
                headers={"comfy-user": "__mynode_config"}
            )

        # Should be blocked
        assert resp.status in [400, 403, 500]


class TestStructuralSecurity:
    """Tests for structural security pattern.

    Verifies:
    - get_public_user_directory() automatically blocks System Users
    - New endpoints using this function are automatically protected
    """

    def test_get_public_user_directory_blocks_system_user(self):
        """
        Any code using get_public_user_directory() is automatically protected.
        """
        # This is the structural security - any new endpoint using this function
        # will automatically block System Users
        assert folder_paths.get_public_user_directory("__system") is None
        assert folder_paths.get_public_user_directory("__cache") is None
        assert folder_paths.get_public_user_directory("__anything") is None

        # Public Users work
        assert folder_paths.get_public_user_directory("default") is not None
        assert folder_paths.get_public_user_directory("user123") is not None

    def test_structural_security_pattern(self, mock_user_directory):
        """
        Demonstrate the structural security pattern for new endpoints.

        Any new endpoint should follow this pattern:
        1. Get user from request
        2. Use get_public_user_directory() - automatically blocks System Users
        3. If None, return error
        """
        def new_endpoint_handler(user_id: str) -> str | None:
            """Example of how new endpoints should be implemented."""
            user_path = folder_paths.get_public_user_directory(user_id)
            if user_path is None:
                return None  # Blocked
            return user_path

        # System Users are automatically blocked
        assert new_endpoint_handler("__system") is None
        assert new_endpoint_handler("__secret") is None

        # Public Users work
        assert new_endpoint_handler("default") is not None

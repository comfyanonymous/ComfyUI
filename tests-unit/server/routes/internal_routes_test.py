import pytest
from aiohttp import web
from unittest.mock import MagicMock, patch
from api_server.routes.internal.internal_routes import InternalRoutes
from api_server.services.file_service import FileService
from folder_paths import models_dir, user_directory, output_directory


@pytest.fixture
def internal_routes():
    return InternalRoutes()

@pytest.fixture
def aiohttp_client_factory(aiohttp_client, internal_routes):
    async def _get_client():
        app = internal_routes.get_app()
        return await aiohttp_client(app)
    return _get_client

@pytest.mark.asyncio
async def test_list_files_valid_directory(aiohttp_client_factory, internal_routes):
    mock_file_list = [
        {"name": "file1.txt", "path": "file1.txt", "type": "file", "size": 100},
        {"name": "dir1", "path": "dir1", "type": "directory"}
    ]
    internal_routes.file_service.list_files = MagicMock(return_value=mock_file_list)
    client = await aiohttp_client_factory()
    resp = await client.get('/files?directory=models')
    assert resp.status == 200
    data = await resp.json()
    assert 'files' in data
    assert len(data['files']) == 2
    assert data['files'] == mock_file_list

    # Check other valid directories
    resp = await client.get('/files?directory=user')
    assert resp.status == 200
    resp = await client.get('/files?directory=output')
    assert resp.status == 200

@pytest.mark.asyncio
async def test_list_files_invalid_directory(aiohttp_client_factory, internal_routes):
    internal_routes.file_service.list_files = MagicMock(side_effect=ValueError("Invalid directory key"))
    client = await aiohttp_client_factory()
    resp = await client.get('/files?directory=invalid')
    assert resp.status == 400
    data = await resp.json()
    assert 'error' in data
    assert data['error'] == "Invalid directory key"

@pytest.mark.asyncio
async def test_list_files_exception(aiohttp_client_factory, internal_routes):
    internal_routes.file_service.list_files = MagicMock(side_effect=Exception("Unexpected error"))
    client = await aiohttp_client_factory()
    resp = await client.get('/files?directory=models')
    assert resp.status == 500
    data = await resp.json()
    assert 'error' in data
    assert data['error'] == "Unexpected error"

@pytest.mark.asyncio
async def test_list_files_no_directory_param(aiohttp_client_factory, internal_routes):
    mock_file_list = []
    internal_routes.file_service.list_files = MagicMock(return_value=mock_file_list)
    client = await aiohttp_client_factory()
    resp = await client.get('/files')
    assert resp.status == 200
    data = await resp.json()
    assert 'files' in data
    assert len(data['files']) == 0

def test_setup_routes(internal_routes):
    internal_routes.setup_routes()
    routes = internal_routes.routes
    assert any(route.method == 'GET' and str(route.path) == '/files' for route in routes)

def test_get_app(internal_routes):
    app = internal_routes.get_app()
    assert isinstance(app, web.Application)
    assert internal_routes._app is not None

def test_get_app_reuse(internal_routes):
    app1 = internal_routes.get_app()
    app2 = internal_routes.get_app()
    assert app1 is app2

@pytest.mark.asyncio
async def test_routes_added_to_app(aiohttp_client_factory, internal_routes):
    client = await aiohttp_client_factory()
    try:
        resp = await client.get('/files')
        print(f"Response received: status {resp.status}")
    except Exception as e:
        print(f"Exception occurred during GET request: {e}")
        raise

    assert resp.status != 404, "Route /files does not exist"

@pytest.mark.asyncio
async def test_file_service_initialization():
    with patch('api_server.routes.internal.internal_routes.FileService') as MockFileService:
        # Create a mock instance
        mock_file_service_instance = MagicMock(spec=FileService)
        MockFileService.return_value = mock_file_service_instance
        internal_routes = InternalRoutes()

        # Check if FileService was initialized with the correct parameters
        MockFileService.assert_called_once_with({
            "models": models_dir,
            "user": user_directory,
            "output": output_directory
        })

        # Verify that the file_service attribute of InternalRoutes is set
        assert internal_routes.file_service == mock_file_service_instance
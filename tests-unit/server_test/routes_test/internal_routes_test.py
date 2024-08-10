import pytest
from aiohttp import web
from unittest.mock import MagicMock, patch
from api_server.routes.internal.internal_routes import InternalRoutes
from folder_paths import models_dir, user_directory, output_directory


@pytest.fixture
def internal_routes():
    return InternalRoutes()

@pytest.fixture
async def client(aiohttp_client, internal_routes):
    app = internal_routes.get_app()
    return await aiohttp_client(app)

@pytest.mark.asyncio
async def test_list_files_valid_directory(client, internal_routes):
    mock_file_list = [
        {"name": "file1.txt", "path": "file1.txt", "type": "file", "size": 100},
        {"name": "dir1", "path": "dir1", "type": "directory"}
    ]
    internal_routes.file_service.list_files = MagicMock(return_value=mock_file_list)

    resp = await client.get('/files?directory=models')
    assert resp.status == 200
    data = await resp.json()
    assert 'files' in data
    assert len(data['files']) == 2
    assert data['files'] == mock_file_list

@pytest.mark.asyncio
async def test_list_files_invalid_directory(client, internal_routes):
    internal_routes.file_service.list_files = MagicMock(side_effect=ValueError("Invalid directory key"))

    resp = await client.get('/files?directory=invalid')
    assert resp.status == 400
    data = await resp.json()
    assert 'error' in data
    assert data['error'] == "Invalid directory key"

@pytest.mark.asyncio
async def test_list_files_exception(client, internal_routes):
    internal_routes.file_service.list_files = MagicMock(side_effect=Exception("Unexpected error"))

    resp = await client.get('/files?directory=models')
    assert resp.status == 500
    data = await resp.json()
    assert 'error' in data
    assert data['error'] == "Unexpected error"

@pytest.mark.asyncio
async def test_list_files_no_directory_param(client, internal_routes):
    mock_file_list = []
    internal_routes.file_service.list_files = MagicMock(return_value=mock_file_list)

    resp = await client.get('/files')
    assert resp.status == 200
    data = await resp.json()
    assert 'files' in data
    assert len(data['files']) == 0

@patch('server.routes.internal_routes.FileService')
def test_file_service_initialization(mock_file_service):
    InternalRoutes()
    mock_file_service.assert_called_once_with({
        "models": models_dir,
        "user": user_directory,
        "output": output_directory
    })

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

# Additional test to check if routes are added to the application
@pytest.mark.asyncio
async def test_routes_added_to_app(aiohttp_client, internal_routes):
    app = internal_routes.get_app()
    client = await aiohttp_client(app)
    
    # This will raise an exception if the route doesn't exist
    await client.get('/files')
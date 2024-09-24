import pytest
from unittest.mock import MagicMock
from api_server.services.file_service import FileService

@pytest.fixture
def mock_file_system_ops():
    return MagicMock()

@pytest.fixture
def file_service(mock_file_system_ops):
    allowed_directories = {
        "models": "/path/to/models",
        "user": "/path/to/user",
        "output": "/path/to/output"
    }
    return FileService(allowed_directories, file_system_ops=mock_file_system_ops)

def test_list_files_valid_directory(file_service, mock_file_system_ops):
    mock_file_system_ops.walk_directory.return_value = [
        {"name": "file1.txt", "path": "file1.txt", "type": "file", "size": 100},
        {"name": "dir1", "path": "dir1", "type": "directory"}
    ]
    
    result = file_service.list_files("models")
    
    assert len(result) == 2
    assert result[0]["name"] == "file1.txt"
    assert result[1]["name"] == "dir1"
    mock_file_system_ops.walk_directory.assert_called_once_with("/path/to/models")

def test_list_files_invalid_directory(file_service):
    # Does not support walking directories outside of the allowed directories
    with pytest.raises(ValueError, match="Invalid directory key"):
        file_service.list_files("invalid_key")

def test_list_files_empty_directory(file_service, mock_file_system_ops):
    mock_file_system_ops.walk_directory.return_value = []
    
    result = file_service.list_files("models")
    
    assert len(result) == 0
    mock_file_system_ops.walk_directory.assert_called_once_with("/path/to/models")

@pytest.mark.parametrize("directory_key", ["models", "user", "output"])
def test_list_files_all_allowed_directories(file_service, mock_file_system_ops, directory_key):
    mock_file_system_ops.walk_directory.return_value = [
        {"name": f"file_{directory_key}.txt", "path": f"file_{directory_key}.txt", "type": "file", "size": 100}
    ]
    
    result = file_service.list_files(directory_key)
    
    assert len(result) == 1
    assert result[0]["name"] == f"file_{directory_key}.txt"
    mock_file_system_ops.walk_directory.assert_called_once_with(f"/path/to/{directory_key}")
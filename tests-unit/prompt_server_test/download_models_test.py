import pytest
import aiohttp
import uuid
from unittest.mock import AsyncMock, MagicMock
from model_filemanager import download_model, DownloadStatus, DownloadStatusType


async def async_iterator(chunks):
    for chunk in chunks:
        yield chunk

@pytest.mark.asyncio
async def test_download_model_success():
    # Create a temporary directory for testing
    model_directory = str(uuid.uuid4())
    
    # Create a mock session
    session = AsyncMock(spec=aiohttp.ClientSession)
    
    # Mock the response
    mock_response = MagicMock(spec=aiohttp.ClientResponse)
    mock_response.status = 200
    mock_response.headers = {'Content-Length': '100'}
    mock_response.content.iter_chunked.return_value = async_iterator([b'chunk1', b'chunk2'])
    
    session.get.return_value.__aenter__.return_value = mock_response
    
    # Create a mock progress callback
    progress_callback = AsyncMock()
    
    # Call the function
    result = await download_model(session, 'model.safetensors', 'http://example.com/model.safetensors', model_directory, progress_callback)
    
    # Assert the expected behavior
    assert result['status'] == DownloadStatusType.COMPLETED
    assert result['message'] == 'Successfully downloaded model.safetensors'
    assert result['already_existed'] is False
    relative_path = '/'.join([model_directory, 'model.safetensors'])
    progress_callback.assert_awaited_with(relative_path, DownloadStatus(status=DownloadStatusType.COMPLETED, progress_percentage=100, message='Successfully downloaded model.safetensors'))
    

@pytest.mark.asyncio
async def test_download_model_failure():
    # Create a temporary directory for testing
    model_directory = str(uuid.uuid4())
    
    # Create a mock session
    session = AsyncMock(spec=aiohttp.ClientSession)
    
    # Mock the response with an error status code
    mock_response = MagicMock(spec=aiohttp.ClientResponse)
    mock_response.status = 500
    session.get.return_value.__aenter__.return_value = mock_response
    
    # Create a mock progress callback
    progress_callback = AsyncMock()
    
    # Call the function
    result = await download_model(session, 'model.safetensors', 'http://example.com/model.safetensors', model_directory, progress_callback)
    print(result)
    
    # Assert the expected behavior
    assert result['status'] == DownloadStatusType.ERROR
    assert result['message'].strip() == 'Failed to download model.safetensors. Status code: 500'
    assert result['already_existed'] is False
    
    relative_path = '/'.join([model_directory, 'model.safetensors'])
    progress_callback.assert_awaited_with(relative_path, DownloadStatus(status=DownloadStatusType.ERROR, progress_percentage=0, message='Failed to download model.safetensors. Status code: 500'))
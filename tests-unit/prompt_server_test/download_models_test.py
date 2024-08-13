import pytest
import aiohttp
from aiohttp import ClientResponse
import itertools
import os 
from unittest.mock import AsyncMock, patch, MagicMock
from model_filemanager import download_model, validate_model_subdirectory, track_download_progress, create_model_path, check_file_exists, DownloadStatusType, DownloadModelStatus, validate_filename

class AsyncIteratorMock:
    """
    A mock class that simulates an asynchronous iterator.
    This is used to mimic the behavior of aiohttp's content iterator.
    """
    def __init__(self, seq):
        # Convert the input sequence into an iterator
        self.iter = iter(seq)

    def __aiter__(self):
        # This method is called when 'async for' is used
        return self

    async def __anext__(self):
        # This method is called for each iteration in an 'async for' loop
        try:
            return next(self.iter)
        except StopIteration:
            # This is the asynchronous equivalent of StopIteration
            raise StopAsyncIteration

class ContentMock:
    """
    A mock class that simulates the content attribute of an aiohttp ClientResponse.
    This class provides the iter_chunked method which returns an async iterator of chunks.
    """
    def __init__(self, chunks):
        # Store the chunks that will be returned by the iterator
        self.chunks = chunks

    def iter_chunked(self, chunk_size):
        # This method mimics aiohttp's content.iter_chunked()
        # For simplicity in testing, we ignore chunk_size and just return our predefined chunks
        return AsyncIteratorMock(self.chunks)

@pytest.mark.asyncio
async def test_download_model_success():
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.status = 200
    mock_response.headers = {'Content-Length': '1000'}
    # Create a mock for content that returns an async iterator directly
    chunks = [b'a' * 500, b'b' * 300, b'c' * 200]
    mock_response.content = ContentMock(chunks)

    mock_make_request = AsyncMock(return_value=mock_response)
    mock_progress_callback = AsyncMock()

    # Mock file operations
    mock_open = MagicMock()
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    time_values = itertools.count(0, 0.1)

    with patch('model_filemanager.create_model_path', return_value=('models/checkpoints/model.sft', 'checkpoints/model.sft')), \
         patch('model_filemanager.check_file_exists', return_value=None), \
         patch('builtins.open', mock_open), \
         patch('time.time', side_effect=time_values):  # Simulate time passing

        result = await download_model(
            mock_make_request,
            'model.sft',
            'http://example.com/model.sft',
            'checkpoints',
            mock_progress_callback
        )

    # Assert the result
    assert isinstance(result, DownloadModelStatus)
    assert result.message == 'Successfully downloaded model.sft'
    assert result.status == 'completed'
    assert result.already_existed is False

    # Check progress callback calls
    assert mock_progress_callback.call_count >= 3  # At least start, one progress update, and completion
    
    # Check initial call
    mock_progress_callback.assert_any_call(
        'checkpoints/model.sft',
        DownloadModelStatus(DownloadStatusType.PENDING, 0, "Starting download of model.sft", False)
    )

    # Check final call
    mock_progress_callback.assert_any_call(
        'checkpoints/model.sft',
        DownloadModelStatus(DownloadStatusType.COMPLETED, 100, "Successfully downloaded model.sft", False)
    )

    # Verify file writing
    mock_file.write.assert_any_call(b'a' * 500)
    mock_file.write.assert_any_call(b'b' * 300)
    mock_file.write.assert_any_call(b'c' * 200)

    # Verify request was made
    mock_make_request.assert_called_once_with('http://example.com/model.sft')

@pytest.mark.asyncio
async def test_download_model_url_request_failure():
    # Mock dependencies
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 404  # Simulate a "Not Found" error
    mock_get = AsyncMock(return_value=mock_response)
    mock_progress_callback = AsyncMock()

    # Mock the create_model_path function
    with patch('model_filemanager.create_model_path', return_value=('/mock/path/model.safetensors', 'mock/path/model.safetensors')):
        # Mock the check_file_exists function to return None (file doesn't exist)
        with patch('model_filemanager.check_file_exists', return_value=None):
            # Call the function
            result = await download_model(
                mock_get,
                'model.safetensors',
                'http://example.com/model.safetensors',
                'mock_directory',
                mock_progress_callback
            )

    # Assert the expected behavior
    assert isinstance(result, DownloadModelStatus)
    assert result.status == 'error'
    assert result.message == 'Failed to download model.safetensors. Status code: 404'
    assert result.already_existed is False

    # Check that progress_callback was called with the correct arguments
    mock_progress_callback.assert_any_call(
        'mock_directory/model.safetensors',
        DownloadModelStatus(
            status=DownloadStatusType.PENDING,
            progress_percentage=0,
            message='Starting download of model.safetensors',
            already_existed=False
        )
    )
    mock_progress_callback.assert_called_with(
        'mock_directory/model.safetensors',
        DownloadModelStatus(
            status=DownloadStatusType.ERROR,
            progress_percentage=0,
            message='Failed to download model.safetensors. Status code: 404',
            already_existed=False
        )
    )

    # Verify that the get method was called with the correct URL
    mock_get.assert_called_once_with('http://example.com/model.safetensors')

@pytest.mark.asyncio
async def test_download_model_invalid_model_subdirectory():
    
    mock_make_request = AsyncMock()
    mock_progress_callback = AsyncMock()

    
    result = await download_model(
        mock_make_request,
        'model.sft',
        'http://example.com/model.sft',
        '../bad_path',
        mock_progress_callback
    )

    # Assert the result
    assert isinstance(result, DownloadModelStatus)
    assert result.message == 'Invalid model subdirectory'
    assert result.status == 'error'
    assert result.already_existed is False


# For create_model_path function
def test_create_model_path(tmp_path, monkeypatch):
    mock_models_dir = tmp_path / "models"
    monkeypatch.setattr('folder_paths.models_dir', str(mock_models_dir))
    
    model_name = "test_model.sft"
    model_directory = "test_dir"
    
    file_path, relative_path = create_model_path(model_name, model_directory, mock_models_dir)
    
    assert file_path == str(mock_models_dir / model_directory / model_name)
    assert relative_path == f"{model_directory}/{model_name}"
    assert os.path.exists(os.path.dirname(file_path))


@pytest.mark.asyncio
async def test_check_file_exists_when_file_exists(tmp_path):
    file_path = tmp_path / "existing_model.sft"
    file_path.touch()  # Create an empty file
    
    mock_callback = AsyncMock()
    
    result = await check_file_exists(str(file_path), "existing_model.sft", mock_callback, "test/existing_model.sft")
    
    assert result is not None
    assert result.status == "completed"
    assert result.message == "existing_model.sft already exists"
    assert result.already_existed is True
    
    mock_callback.assert_called_once_with(
        "test/existing_model.sft",
        DownloadModelStatus(DownloadStatusType.COMPLETED, 100, "existing_model.sft already exists", already_existed=True)
    )

@pytest.mark.asyncio
async def test_check_file_exists_when_file_does_not_exist(tmp_path):
    file_path = tmp_path / "non_existing_model.sft"
    
    mock_callback = AsyncMock()
    
    result = await check_file_exists(str(file_path), "non_existing_model.sft", mock_callback, "test/non_existing_model.sft")
    
    assert result is None
    mock_callback.assert_not_called()

@pytest.mark.asyncio
async def test_track_download_progress_no_content_length():
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.headers = {}  # No Content-Length header
    mock_response.content.iter_chunked.return_value = AsyncIteratorMock([b'a' * 500, b'b' * 500])

    mock_callback = AsyncMock()
    mock_open = MagicMock(return_value=MagicMock())

    with patch('builtins.open', mock_open):
        result = await track_download_progress(
            mock_response, '/mock/path/model.sft', 'model.sft',
            mock_callback, 'models/model.sft', interval=0.1
        )

    assert result.status == "completed"
    # Check that progress was reported even without knowing the total size
    mock_callback.assert_any_call(
        'models/model.sft',
        DownloadModelStatus(DownloadStatusType.IN_PROGRESS, 0, "Downloading model.sft", already_existed=False)
    )

@pytest.mark.asyncio
async def test_track_download_progress_interval():
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.headers = {'Content-Length': '1000'}
    mock_response.content.iter_chunked.return_value = AsyncIteratorMock([b'a' * 100] * 10)

    mock_callback = AsyncMock()
    mock_open = MagicMock(return_value=MagicMock())

    # Create a mock time function that returns incremental float values
    mock_time = MagicMock()
    mock_time.side_effect = [i * 0.5 for i in range(30)]  # This should be enough for 10 chunks

    with patch('builtins.open', mock_open), \
         patch('time.time', mock_time):
        await track_download_progress(
            mock_response, '/mock/path/model.sft', 'model.sft',
            mock_callback, 'models/model.sft', interval=1.0
        )

    # Print out the actual call count and the arguments of each call for debugging
    print(f"mock_callback was called {mock_callback.call_count} times")
    for i, call in enumerate(mock_callback.call_args_list):
        args, kwargs = call
        print(f"Call {i + 1}: {args[1].status}, Progress: {args[1].progress_percentage:.2f}%")

    # Assert that progress was updated at least 3 times (start, at least one interval, and end)
    assert mock_callback.call_count >= 3, f"Expected at least 3 calls, but got {mock_callback.call_count}"

    # Verify the first and last calls
    first_call = mock_callback.call_args_list[0]
    assert first_call[0][1].status == "in_progress"
    # Allow for some initial progress, but it should be less than 50%
    assert 0 <= first_call[0][1].progress_percentage < 50, f"First call progress was {first_call[0][1].progress_percentage}%"

    last_call = mock_callback.call_args_list[-1]
    assert last_call[0][1].status == "completed"
    assert last_call[0][1].progress_percentage == 100

def test_valid_subdirectory():
    assert validate_model_subdirectory("valid-model123") is True

def test_subdirectory_too_long():
    assert validate_model_subdirectory("a" * 51) is False

def test_subdirectory_with_double_dots():
    assert validate_model_subdirectory("model/../unsafe") is False

def test_subdirectory_with_slash():
    assert validate_model_subdirectory("model/unsafe") is False

def test_subdirectory_with_special_characters():
    assert validate_model_subdirectory("model@unsafe") is False

def test_subdirectory_with_underscore_and_dash():
    assert validate_model_subdirectory("valid_model-name") is True

def test_empty_subdirectory():
    assert validate_model_subdirectory("") is False

@pytest.mark.parametrize("filename, expected", [
    ("valid_model.safetensors", True),
    ("valid_model.sft", True),
    ("valid model.safetensors", True), # Test with space
    ("UPPERCASE_MODEL.SAFETENSORS", True),
    ("model_with.multiple.dots.pt", False),
    ("", False),  # Empty string
    ("../../../etc/passwd", False),  # Path traversal attempt
    ("/etc/passwd", False),  # Absolute path
    ("\\windows\\system32\\config\\sam", False),  # Windows path
    (".hidden_file.pt", False),  # Hidden file
    ("invalid<char>.ckpt", False),  # Invalid character
    ("invalid?.ckpt", False),  # Another invalid character
    ("very" * 100 + ".safetensors", False),  # Too long filename
    ("\nmodel_with_newline.pt", False),  # Newline character
    ("model_with_emoji😊.pt", False),  # Emoji in filename
])
def test_validate_filename(filename, expected):
    assert validate_filename(filename) == expected

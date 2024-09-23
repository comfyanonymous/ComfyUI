import pytest
import tempfile
import aiohttp
from aiohttp import ClientResponse
import itertools
import os
from unittest.mock import AsyncMock, patch, MagicMock
from model_filemanager import download_model, track_download_progress, create_model_path, check_file_exists, DownloadStatusType, DownloadModelStatus, validate_filename
import folder_paths

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

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
async def test_download_model_success(temp_dir):
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.status = 200
    mock_response.headers = {'Content-Length': '1000'}
    # Create a mock for content that returns an async iterator directly
    chunks = [b'a' * 500, b'b' * 300, b'c' * 200]
    mock_response.content = ContentMock(chunks)

    mock_make_request = AsyncMock(return_value=mock_response)
    mock_progress_callback = AsyncMock()

    time_values = itertools.count(0, 0.1)

    fake_paths = {'checkpoints': ([temp_dir], folder_paths.supported_pt_extensions)}

    with patch('model_filemanager.create_model_path', return_value=('models/checkpoints/model.sft', 'model.sft')), \
         patch('model_filemanager.check_file_exists', return_value=None), \
         patch('folder_paths.folder_names_and_paths', fake_paths), \
         patch('time.time', side_effect=time_values):  # Simulate time passing

        result = await download_model(
            mock_make_request,
            'model.sft',
            'http://example.com/model.sft',
            'checkpoints',
            temp_dir,
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
        'model.sft',
        DownloadModelStatus(DownloadStatusType.PENDING, 0, "Starting download of model.sft", False)
    )

    # Check final call
    mock_progress_callback.assert_any_call(
        'model.sft',
        DownloadModelStatus(DownloadStatusType.COMPLETED, 100, "Successfully downloaded model.sft", False)
    )

    mock_file_path = os.path.join(temp_dir, 'model.sft')
    assert os.path.exists(mock_file_path)
    with open(mock_file_path, 'rb') as mock_file:
        assert mock_file.read() == b''.join(chunks)
    os.remove(mock_file_path)

    # Verify request was made
    mock_make_request.assert_called_once_with('http://example.com/model.sft')

@pytest.mark.asyncio
async def test_download_model_url_request_failure(temp_dir):
    # Mock dependencies
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 404  # Simulate a "Not Found" error
    mock_get = AsyncMock(return_value=mock_response)
    mock_progress_callback = AsyncMock()
    
    fake_paths = {'checkpoints': ([temp_dir], folder_paths.supported_pt_extensions)}

    # Mock the create_model_path function
    with patch('model_filemanager.create_model_path', return_value='/mock/path/model.safetensors'), \
         patch('model_filemanager.check_file_exists', return_value=None), \
         patch('folder_paths.folder_names_and_paths', fake_paths):
        # Call the function
        result = await download_model(
            mock_get,
            'model.safetensors',
            'http://example.com/model.safetensors',
            'checkpoints',
            temp_dir,
            mock_progress_callback
        )

    # Assert the expected behavior
    assert isinstance(result, DownloadModelStatus)
    assert result.status == 'error'
    assert result.message == 'Failed to download model.safetensors. Status code: 404'
    assert result.already_existed is False

    # Check that progress_callback was called with the correct arguments
    mock_progress_callback.assert_any_call(
        'model.safetensors',
        DownloadModelStatus(
            status=DownloadStatusType.PENDING,
            progress_percentage=0,
            message='Starting download of model.safetensors',
            already_existed=False
        )
    )
    mock_progress_callback.assert_called_with(
        'model.safetensors',
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
        '../bad_path',
        mock_progress_callback
    )

    # Assert the result
    assert isinstance(result, DownloadModelStatus)
    assert result.message.startswith('Invalid or unrecognized model directory')
    assert result.status == 'error'
    assert result.already_existed is False

@pytest.mark.asyncio
async def test_download_model_invalid_folder_path():
    mock_make_request = AsyncMock()
    mock_progress_callback = AsyncMock()

    result = await download_model(
        mock_make_request,
        'model.sft',
        'http://example.com/model.sft',
        'checkpoints',
        'invalid_path',
        mock_progress_callback
    )

    # Assert the result
    assert isinstance(result, DownloadModelStatus)
    assert result.message.startswith("Invalid folder path")
    assert result.status == 'error'
    assert result.already_existed is False

def test_create_model_path(tmp_path, monkeypatch):
    model_name = "model.safetensors"
    folder_path = os.path.join(tmp_path, "mock_dir")

    file_path = create_model_path(model_name, folder_path)

    assert file_path == os.path.join(folder_path, "model.safetensors")
    assert os.path.exists(os.path.dirname(file_path))

    with pytest.raises(Exception, match="Invalid model directory"):
        create_model_path("../path_traversal.safetensors", folder_path)

    with pytest.raises(Exception, match="Invalid model directory"):
        create_model_path("/etc/some_root_path", folder_path)


@pytest.mark.asyncio
async def test_check_file_exists_when_file_exists(tmp_path):
    file_path = tmp_path / "existing_model.sft"
    file_path.touch()  # Create an empty file

    mock_callback = AsyncMock()

    result = await check_file_exists(str(file_path), "existing_model.sft", mock_callback)

    assert result is not None
    assert result.status == "completed"
    assert result.message == "existing_model.sft already exists"
    assert result.already_existed is True

    mock_callback.assert_called_once_with(
        "existing_model.sft",
        DownloadModelStatus(DownloadStatusType.COMPLETED, 100, "existing_model.sft already exists", already_existed=True)
    )

@pytest.mark.asyncio
async def test_check_file_exists_when_file_does_not_exist(tmp_path):
    file_path = tmp_path / "non_existing_model.sft"

    mock_callback = AsyncMock()

    result = await check_file_exists(str(file_path), "non_existing_model.sft", mock_callback)

    assert result is None
    mock_callback.assert_not_called()

@pytest.mark.asyncio
async def test_track_download_progress_no_content_length(temp_dir):
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.headers = {}  # No Content-Length header
    chunks = [b'a' * 500, b'b' * 500]
    mock_response.content.iter_chunked.return_value = AsyncIteratorMock(chunks)

    mock_callback = AsyncMock()

    full_path = os.path.join(temp_dir, 'model.sft')

    result = await track_download_progress(
        mock_response, full_path, 'model.sft',
        mock_callback, interval=0.1
    )

    assert result.status == "completed"

    assert os.path.exists(full_path)
    with open(full_path, 'rb') as f:
        assert f.read() == b''.join(chunks)
    os.remove(full_path)

    # Check that progress was reported even without knowing the total size
    mock_callback.assert_any_call(
        'model.sft',
        DownloadModelStatus(DownloadStatusType.IN_PROGRESS, 0, "Downloading model.sft", already_existed=False)
    )

@pytest.mark.asyncio
async def test_track_download_progress_interval(temp_dir):
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.headers = {'Content-Length': '1000'}
    chunks = [b'a' * 100] * 10
    mock_response.content.iter_chunked.return_value = AsyncIteratorMock(chunks)

    mock_callback = AsyncMock()
    mock_open = MagicMock(return_value=MagicMock())

    # Create a mock time function that returns incremental float values
    mock_time = MagicMock()
    mock_time.side_effect = [i * 0.5 for i in range(30)]  # This should be enough for 10 chunks

    full_path = os.path.join(temp_dir, 'model.sft')

    with patch('time.time', mock_time):
        await track_download_progress(
            mock_response, full_path, 'model.sft',
            mock_callback, interval=1.0
        )
    
    assert os.path.exists(full_path)
    with open(full_path, 'rb') as f:
        assert f.read() == b''.join(chunks)
    os.remove(full_path)

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

#NOTE: This was an experiment and WILL BE REMOVED
from __future__ import annotations
import aiohttp
import os
import traceback
import logging
from folder_paths import folder_names_and_paths, get_folder_paths
import re
from typing import Callable, Any, Optional, Awaitable, Dict
from enum import Enum
import time
from dataclasses import dataclass


class DownloadStatusType(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class DownloadModelStatus():
    status: str
    progress_percentage: float
    message: str
    already_existed: bool = False

    def __init__(self, status: DownloadStatusType, progress_percentage: float, message: str, already_existed: bool):
        self.status = status.value  # Store the string value of the Enum
        self.progress_percentage = progress_percentage
        self.message = message
        self.already_existed = already_existed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "progress_percentage": self.progress_percentage,
            "message": self.message,
            "already_existed": self.already_existed
        }


async def download_model(model_download_request: Callable[[str], Awaitable[aiohttp.ClientResponse]],
                         model_name: str,
                         model_url: str,
                         model_directory: str,
                         folder_path: str,
                         progress_callback: Callable[[str, DownloadModelStatus], Awaitable[Any]],
                         progress_interval: float = 1.0) -> DownloadModelStatus:
    """
    Download a model file from a given URL into the models directory.

    Args:
        model_download_request (Callable[[str], Awaitable[aiohttp.ClientResponse]]):
            A function that makes an HTTP request. This makes it easier to mock in unit tests.
        model_name (str):
            The name of the model file to be downloaded. This will be the filename on disk.
        model_url (str):
            The URL from which to download the model.
        model_directory (str):
            The subdirectory within the main models directory where the model
            should be saved (e.g., 'checkpoints', 'loras', etc.).
        progress_callback (Callable[[str, DownloadModelStatus], Awaitable[Any]]):
            An asynchronous function to call with progress updates.
        folder_path (str);
            Path to which model folder should be used as the root.

    Returns:
        DownloadModelStatus: The result of the download operation.
    """
    if not validate_filename(model_name):
        return DownloadModelStatus(
            DownloadStatusType.ERROR,
            0,
            "Invalid model name",
            False
        )

    if not model_directory in folder_names_and_paths:
        return DownloadModelStatus(
            DownloadStatusType.ERROR,
            0,
            "Invalid or unrecognized model directory. model_directory must be a known model type (eg 'checkpoints'). If you are seeing this error for a custom model type, ensure the relevant custom nodes are installed and working.",
            False
        )

    if not folder_path in get_folder_paths(model_directory):
        return DownloadModelStatus(
            DownloadStatusType.ERROR,
            0,
            f"Invalid folder path '{folder_path}', does not match the list of known directories ({get_folder_paths(model_directory)}). If you're seeing this in the downloader UI, you may need to refresh the page.",
            False
        )

    file_path = create_model_path(model_name, folder_path)
    existing_file = await check_file_exists(file_path, model_name, progress_callback)
    if existing_file:
        return existing_file

    try:
        logging.info(f"Downloading {model_name} from {model_url}")
        status = DownloadModelStatus(DownloadStatusType.PENDING, 0, f"Starting download of {model_name}", False)
        await progress_callback(model_name, status)

        response = await model_download_request(model_url)
        if response.status != 200:
            error_message = f"Failed to download {model_name}. Status code: {response.status}"
            logging.error(error_message)
            status = DownloadModelStatus(DownloadStatusType.ERROR, 0, error_message, False)
            await progress_callback(model_name, status)
            return DownloadModelStatus(DownloadStatusType.ERROR, 0, error_message, False)

        return await track_download_progress(response, file_path, model_name, progress_callback, progress_interval)

    except Exception as e:
        logging.error(f"Error in downloading model: {e}")
        return await handle_download_error(e, model_name, progress_callback)


def create_model_path(model_name: str, folder_path: str) -> tuple[str, str]:
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, model_name)

    # Ensure the resulting path is still within the base directory
    abs_file_path = os.path.abspath(file_path)
    abs_base_dir = os.path.abspath(folder_path)
    if os.path.commonprefix([abs_file_path, abs_base_dir]) != abs_base_dir:
        raise Exception(f"Invalid model directory: {folder_path}/{model_name}")

    return file_path


async def check_file_exists(file_path: str,
                            model_name: str,
                            progress_callback: Callable[[str, DownloadModelStatus], Awaitable[Any]]
                            ) -> Optional[DownloadModelStatus]:
    if os.path.exists(file_path):
        status = DownloadModelStatus(DownloadStatusType.COMPLETED, 100, f"{model_name} already exists", True)
        await progress_callback(model_name, status)
        return status
    return None


async def track_download_progress(response: aiohttp.ClientResponse,
                                  file_path: str,
                                  model_name: str,
                                  progress_callback: Callable[[str, DownloadModelStatus], Awaitable[Any]],
                                  interval: float = 1.0) -> DownloadModelStatus:
    try:
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        last_update_time = time.time()

        async def update_progress():
            nonlocal last_update_time
            progress = (downloaded / total_size) * 100 if total_size > 0 else 0
            status = DownloadModelStatus(DownloadStatusType.IN_PROGRESS, progress, f"Downloading {model_name}", False)
            await progress_callback(model_name, status)
            last_update_time = time.time()

        temp_file_path = file_path + '.tmp'
        with open(temp_file_path, 'wb') as f:
            chunk_iterator = response.content.iter_chunked(8192)
            while True:
                try:
                    chunk = await chunk_iterator.__anext__()
                except StopAsyncIteration:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if time.time() - last_update_time >= interval:
                    await update_progress()

        os.rename(temp_file_path, file_path)

        await update_progress()

        logging.info(f"Successfully downloaded {model_name}. Total downloaded: {downloaded}")
        status = DownloadModelStatus(DownloadStatusType.COMPLETED, 100, f"Successfully downloaded {model_name}", False)
        await progress_callback(model_name, status)

        return status
    except Exception as e:
        logging.error(f"Error in track_download_progress: {e}")
        logging.error(traceback.format_exc())
        return await handle_download_error(e, model_name, progress_callback)


async def handle_download_error(e: Exception,
                                model_name: str,
                                progress_callback: Callable[[str, DownloadModelStatus], Any]
                                ) -> DownloadModelStatus:
    error_message = f"Error downloading {model_name}: {str(e)}"
    status = DownloadModelStatus(DownloadStatusType.ERROR, 0, error_message, False)
    await progress_callback(model_name, status)
    return status


def validate_filename(filename: str)-> bool:
    """
    Validate a filename to ensure it's safe and doesn't contain any path traversal attempts.

    Args:
    filename (str): The filename to validate

    Returns:
    bool: True if the filename is valid, False otherwise
    """
    if not filename.lower().endswith(('.sft', '.safetensors')):
        return False

    # Check if the filename is empty, None, or just whitespace
    if not filename or not filename.strip():
        return False

    # Check for any directory traversal attempts or invalid characters
    if any(char in filename for char in ['..', '/', '\\', '\n', '\r', '\t', '\0']):
        return False

    # Check if the filename starts with a dot (hidden file)
    if filename.startswith('.'):
        return False

    # Use a whitelist of allowed characters
    if not re.match(r'^[a-zA-Z0-9_\-. ]+$', filename):
        return False

    # Ensure the filename isn't too long
    if len(filename) > 255:
        return False

    return True

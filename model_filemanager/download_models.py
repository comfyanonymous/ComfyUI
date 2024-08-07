import aiohttp
import os
import traceback
import logging
from folder_paths import models_dir
from typing import Callable, Any, Optional, Awaitable
from enum import Enum
import time
from dataclasses import dataclass

class DownloadStatusType(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class DownloadStatus():
    status: str
    progress_percentage: float
    message: str

    def __init__(self, status: DownloadStatusType, progress_percentage: float, message: str):
        self.status = status.value  # Store the string value of the Enum
        self.progress_percentage = progress_percentage
        self.message = message

@dataclass
class DownloadModelResult():
    status: str
    message: str
    already_existed: bool

    def __init__(self, status: DownloadStatusType, message: str, already_existed: bool):
        self.status = status.value  # Store the string value of the Enum
        self.message = message
        self.already_existed = already_existed

async def download_model(make_request: Callable[[str], Awaitable[aiohttp.ClientResponse]],
                         model_name: str,  
                         model_url: str, 
                         model_directory: str, 
                         progress_callback: Callable[[str, DownloadStatus], Awaitable[Any]]) -> DownloadModelResult:

    file_path, relative_path = create_model_path(model_name, model_directory, models_dir)
    existing_file = await check_file_exists(file_path, model_name, progress_callback, relative_path)
    if existing_file:
        return existing_file

    try:
        status = DownloadStatus(DownloadStatusType.PENDING, 0, f"Starting download of {model_name}")
        await progress_callback(relative_path, status)

        response = await make_request(model_url)
        if response.status != 200:
            error_message = f"Failed to download {model_name}. Status code: {response.status}"
            status = DownloadStatus(DownloadStatusType.ERROR, 0, error_message)
            await progress_callback(relative_path, status)
            return DownloadModelResult(DownloadStatusType.ERROR, error_message, False)

        return await track_download_progress(response, file_path, model_name, progress_callback, relative_path)

    except Exception as e:
        logging.error(f"Error in track_download_progress: {e}")
        return await handle_download_error(e, model_name, progress_callback, relative_path)
    


async def make_http_request(session: aiohttp.ClientSession, url: str) -> aiohttp.ClientResponse:
    return await session.get(url)

def create_model_path(model_name: str, model_directory: str, models_base_dir: str) -> tuple[str, str]:
    full_model_dir = os.path.join(models_base_dir, model_directory)
    os.makedirs(full_model_dir, exist_ok=True)
    file_path = os.path.join(full_model_dir, model_name)
    relative_path = '/'.join([model_directory, model_name])
    return file_path, relative_path

async def check_file_exists(file_path: str, model_name: str, progress_callback: Callable[[str, DownloadStatus], Awaitable[Any]], relative_path: str) -> Optional[DownloadModelResult]:
    if os.path.exists(file_path):
        status = DownloadStatus(DownloadStatusType.COMPLETED, 100, f"{model_name} already exists")
        await progress_callback(relative_path, status)
        return DownloadModelResult(DownloadStatusType.COMPLETED, f"{model_name} already exists", True)
    return None


async def track_download_progress(response: aiohttp.ClientResponse, file_path: str, model_name: str, progress_callback: Callable[[str, DownloadStatus], Awaitable[Any]], relative_path: str, interval: float = 1.0) -> DownloadModelResult:
    try:
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        last_update_time = time.time()

        async def update_progress():
            nonlocal last_update_time
            progress = (downloaded / total_size) * 100 if total_size > 0 else 0
            status = DownloadStatus(DownloadStatusType.IN_PROGRESS, progress, f"Downloading {model_name}")
            await progress_callback(relative_path, status)
            last_update_time = time.time()

        with open(file_path, 'wb') as f:
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

        await update_progress()
        
        logging.info(f"Successfully downloaded {model_name}. Total downloaded: {downloaded}")
        status = DownloadStatus(DownloadStatusType.COMPLETED, 100, f"Successfully downloaded {model_name}")
        await progress_callback(relative_path, status)

        return DownloadModelResult(DownloadStatusType.COMPLETED, f"Successfully downloaded {model_name}", False)
    except Exception as e:
        logging.error(f"Error in track_download_progress: {e}")
        logging.error(traceback.format_exc())
        return await handle_download_error(e, model_name, progress_callback, relative_path)

async def handle_download_error(e: Exception, model_name: str, progress_callback: Callable[[str, DownloadStatus], Any], relative_path: str) -> DownloadModelResult:
    error_message = f"Error downloading {model_name}: {str(e)}"
    status = DownloadStatus(DownloadStatusType.ERROR, 0, error_message)
    await progress_callback(relative_path, status)
    return DownloadModelResult(DownloadStatusType.ERROR, error_message, False)
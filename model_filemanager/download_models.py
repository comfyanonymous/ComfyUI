import aiohttp
import os
from folder_paths import models_dir
from typing import Callable, Any, Optional
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

async def download_model(session: aiohttp.ClientSession, 
                         model_name: str,  
                         model_url: str, 
                         model_directory: str, 
                         progress_callback: Callable[[str, DownloadStatus], Any]) -> DownloadModelResult:
    file_path, relative_path = create_model_path(model_name, model_directory)
    
    existing_file = await check_file_exists(file_path, model_name, progress_callback, relative_path)
    if existing_file:
        return existing_file

    try:
        status = DownloadStatus(DownloadStatusType.PENDING, 0, f"Starting download of {model_name}")
        await progress_callback(relative_path, status)

        response = await session.get(model_url)
        
        if response.status != 200:
            error_message = f"Failed to download {model_name}. Status code: {response.status}"
            status = DownloadStatus(DownloadStatusType.ERROR, 0, error_message)
            await progress_callback(relative_path, status)
            return DownloadModelResult(DownloadStatusType.ERROR, error_message, False)

        return await track_download_progress(response, file_path, model_name, progress_callback, relative_path)

    except Exception as e:
        return await handle_download_error(e, model_name, progress_callback, relative_path)
    

def create_model_path(model_name: str, model_directory: str) -> tuple[str, str]:
    full_model_dir = os.path.join(models_dir, model_directory)
    os.makedirs(full_model_dir, exist_ok=True)
    file_path = os.path.join(full_model_dir, model_name)
    relative_path = '/'.join([model_directory, model_name])
    return file_path, relative_path

async def check_file_exists(file_path: str, model_name: str, progress_callback: Callable[[str, DownloadStatus], Any], relative_path: str) -> Optional[DownloadModelResult]:
    if os.path.exists(file_path):
        status = DownloadStatus(DownloadStatusType.COMPLETED, 100, f"{model_name} already exists")
        await progress_callback(relative_path, status)
        return DownloadModelResult(DownloadStatusType.COMPLETED, f"{model_name} already exists", True)
    return None


async def track_download_progress(response: aiohttp.ClientResponse, file_path: str, model_name: str, progress_callback: Callable[[str, DownloadStatus], Any], relative_path: str, interval: float = 1.0) -> DownloadModelResult:
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
        async for chunk in response.content.iter_chunked(8192):
            f.write(chunk)
            downloaded += len(chunk)
            
            # Check if it's time to update progress
            if time.time() - last_update_time >= interval:
                await update_progress()

    # Ensure we send a final update
    await update_progress()

    status = DownloadStatus(DownloadStatusType.COMPLETED, 100, f"Successfully downloaded {model_name}")
    await progress_callback(relative_path, status)

    return DownloadModelResult(DownloadStatusType.COMPLETED, f"Successfully downloaded {model_name}", False)

async def handle_download_error(e: Exception, model_name: str, progress_callback: Callable[[str, DownloadStatus], Any], relative_path: str) -> DownloadModelResult:
    error_message = f"Error downloading {model_name}: {str(e)}"
    status = DownloadStatus(DownloadStatusType.ERROR, 0, error_message)
    await progress_callback(relative_path, status)
    return DownloadModelResult(DownloadStatusType.ERROR, error_message, False)
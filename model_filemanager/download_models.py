import aiohttp
import os
from folder_paths import models_dir
from typing import Callable, Any
from enum import Enum

from dataclasses import dataclass

class DownloadStatusType(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class DownloadStatus():
    status: DownloadStatusType
    progress_percentage: float
    message: str

@dataclass
class DownloadModelResult():
    status: DownloadStatusType
    message: str
    already_existed: bool

async def download_model(session: aiohttp.ClientSession, 
                         model_name: str,  
                         model_url: str, 
                         model_directory: str, 
                         progress_callback: Callable[[str, DownloadStatus], Any]) -> DownloadModelResult:
    """
    Asynchronously downloads a model file from a given URL to a specified directory.

    If the file already exists, return success.
    Downloads the file in chunks and reports progress as a percentage through the callback function.
    """
    
    full_model_dir = os.path.join(models_dir, model_directory)
    os.makedirs(full_model_dir, exist_ok=True) # Ensure the directory exists.
    file_path = os.path.join(full_model_dir, model_name)
    relative_path = '/'.join([model_directory, model_name])
    if os.path.exists(file_path):
        status = DownloadStatus(DownloadStatusType.COMPLETED, 100, f"{model_name} already exists")
        await progress_callback(relative_path, status)
        return {"status": DownloadStatusType.COMPLETED, "message": f"{model_name} already exists", "already_existed": True}
    try:
        status = DownloadStatus(DownloadStatusType.PENDING, 0, f"Starting download of {model_name}")
        await progress_callback(relative_path, status)

        async with session.get(model_url) as response:
            if response.status != 200:
                error_message = f"Failed to download {model_name}. Status code: {response.status}"
                status = DownloadStatus(DownloadStatusType.ERROR, 0, error_message)
                await progress_callback(relative_path, status)
                return {"status": DownloadStatusType.ERROR, "message": f"Failed to download {model_name}. Status code: {response.status} ", "already_existed": False}

            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0

            with open(file_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
                    downloaded += len(chunk)                                    
                    progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                    status = DownloadStatus(DownloadStatusType.IN_PROGRESS, progress, f"Downloading {model_name}")
                    await progress_callback(relative_path, status)
        
        status = DownloadStatus(DownloadStatusType.COMPLETED, 100, f"Successfully downloaded {model_name}")
        await progress_callback(relative_path, status)

        return {"status": DownloadStatusType.COMPLETED, "message": f"Successfully downloaded {model_name}", "already_existed": False}

    except Exception as e:
        error_message = f"Error downloading {model_name}: {str(e)}"
        status = DownloadStatus(DownloadStatusType.ERROR, 0, error_message)
        await progress_callback(relative_path, status)
        return {"status": DownloadStatusType.ERROR, "message": error_message, "already_existed": False}
from typing import Dict, List, Optional
from api_server.utils.file_operations import FileSystemOperations, FileSystemItem

class FileService:
    def __init__(self, allowed_directories: Dict[str, str], file_system_ops: Optional[FileSystemOperations] = None):
        self.allowed_directories: Dict[str, str] = allowed_directories
        self.file_system_ops: FileSystemOperations = file_system_ops or FileSystemOperations()

    def list_files(self, directory_key: str) -> List[FileSystemItem]:
        if directory_key not in self.allowed_directories:
            raise ValueError("Invalid directory key")
        directory_path: str = self.allowed_directories[directory_key]
        return self.file_system_ops.walk_directory(directory_path)

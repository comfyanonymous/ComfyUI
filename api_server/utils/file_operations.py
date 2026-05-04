import os
from typing import List, Union, TypedDict, Literal
from typing_extensions import TypeGuard
class FileInfo(TypedDict):
    name: str
    path: str
    type: Literal["file"]
    size: int

class DirectoryInfo(TypedDict):
    name: str
    path: str
    type: Literal["directory"]

FileSystemItem = Union[FileInfo, DirectoryInfo]

def is_file_info(item: FileSystemItem) -> TypeGuard[FileInfo]:
    return item["type"] == "file"

class FileSystemOperations:
    @staticmethod
    def walk_directory(directory: str) -> List[FileSystemItem]:
        file_list: List[FileSystemItem] = []
        for root, dirs, files in os.walk(directory):
            for name in files:
                file_path = os.path.join(root, name)
                relative_path = os.path.relpath(file_path, directory)
                file_list.append({
                    "name": name,
                    "path": relative_path,
                    "type": "file",
                    "size": os.path.getsize(file_path)
                })
            for name in dirs:
                dir_path = os.path.join(root, name)
                relative_path = os.path.relpath(dir_path, directory)
                file_list.append({
                    "name": name,
                    "path": relative_path,
                    "type": "directory"
                })
        return file_list

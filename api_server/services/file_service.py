from api_server.utils.file_operations import FileSystemOperations

class FileService:
    def __init__(self, allowed_directories, file_system_ops=None):
        self.allowed_directories = allowed_directories
        self.file_system_ops = file_system_ops or FileSystemOperations()

    def list_files(self, directory_key):
        if directory_key not in self.allowed_directories:
            raise ValueError("Invalid directory key")
        directory_path = self.allowed_directories[directory_key]
        return self.file_system_ops.walk_directory(directory_path)
"""Simple model downloader for ComfyUI."""

import os
import json
import uuid
import threading
import time
import folder_paths
from typing import Dict, Any, Optional
import urllib.request
import urllib.error


class SimpleDownloader:
    """Simple downloader for ComfyUI models."""

    def __init__(self):
        self.downloads = {}
        self.lock = threading.Lock()

    def create_download(self, url: str, model_type: str, filename: str) -> str:
        """Create a new download task."""
        task_id = str(uuid.uuid4())

        # SECURITY: Validate and sanitize inputs to prevent path traversal
        # Sanitize model_type - remove dangerous characters but keep underscores
        import re
        model_type = re.sub(r'[./\\]', '', model_type)
        model_type = model_type.replace('..', '')

        # Sanitize filename - use os.path.basename and remove traversal attempts
        filename = os.path.basename(filename)
        if '..' in filename or filename.startswith('.'):
            raise ValueError("Invalid filename - no hidden files or path traversal")

        # Validate filename has allowed extension
        allowed_extensions = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.sft']
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError(f"Invalid file extension. Allowed extensions: {allowed_extensions}")

        # Whitelist of allowed model types
        allowed_types = ['checkpoints', 'vae', 'loras', 'controlnet', 'clip', 'unet',
                        'upscale_models', 'text_encoders', 'diffusion_models', 'embeddings']

        # Map alternative names
        type_mapping = {
            'text_encoders': 'clip',
            'diffusion_models': 'unet'
        }
        model_type = type_mapping.get(model_type, model_type)

        if model_type not in allowed_types:
            raise ValueError(f"Invalid model type. Allowed types: {allowed_types}")

        # Determine destination folder
        folder_map = {
            'checkpoints': folder_paths.get_folder_paths('checkpoints')[0],
            'vae': folder_paths.get_folder_paths('vae')[0],
            'loras': folder_paths.get_folder_paths('loras')[0],
            'controlnet': folder_paths.get_folder_paths('controlnet')[0],
            'clip': folder_paths.get_folder_paths('clip')[0],
            'unet': folder_paths.get_folder_paths('diffusion_models')[0],
            'upscale_models': folder_paths.get_folder_paths('upscale_models')[0],
            'embeddings': folder_paths.get_folder_paths('embeddings')[0] if folder_paths.get_folder_paths('embeddings') else os.path.join(folder_paths.models_dir, 'embeddings')
        }

        dest_folder = folder_map.get(model_type)
        if not dest_folder:
            # Only allow creating folders for whitelisted types
            if model_type in allowed_types:
                dest_folder = os.path.join(folder_paths.models_dir, model_type)
                os.makedirs(dest_folder, exist_ok=True)
            else:
                raise ValueError(f"Cannot find or create folder for model type: {model_type}")

        # Use safe path joining and verify result
        dest_path = os.path.abspath(os.path.join(dest_folder, filename))

        # SECURITY: Ensure destination path is within the models directory
        models_base = os.path.abspath(folder_paths.models_dir)
        if not dest_path.startswith(models_base):
            raise ValueError("Invalid destination path - outside models directory")

        with self.lock:
            self.downloads[task_id] = {
                'task_id': task_id,
                'url': url,
                'dest_path': dest_path,
                'filename': filename,
                'model_type': model_type,
                'status': 'pending',
                'progress': 0,
                'total_size': 0,
                'downloaded_size': 0,
                'error': None,
                'thread': None
            }

        # Start download in background
        thread = threading.Thread(target=self._download_file, args=(task_id,))
        thread.daemon = True
        thread.start()

        with self.lock:
            self.downloads[task_id]['thread'] = thread
            self.downloads[task_id]['status'] = 'downloading'

        return task_id

    def _download_file(self, task_id: str):
        """Download file in background."""
        with self.lock:
            task = self.downloads.get(task_id)
            if not task:
                return
            url = task['url']
            dest_path = task['dest_path']

        try:
            # SECURITY: Validate URL before downloading
            from urllib.parse import urlparse
            parsed = urlparse(url)

            # Only allow HTTPS for security
            if parsed.scheme != 'https':
                raise ValueError("Only HTTPS URLs are allowed for security")

            # Prevent SSRF attacks - block local/private IPs
            import socket
            try:
                ip = socket.gethostbyname(parsed.hostname)
                # Block private/local IPs
                if ip.startswith(('127.', '10.', '192.168.', '172.')):
                    raise ValueError("Downloads from local/private networks are not allowed")
            except socket.gaierror:
                pass  # Domain name resolution failed, continue

            # Create request with headers
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'ComfyUI/1.0')

            # Open URL
            response = urllib.request.urlopen(req, timeout=30)

            # Get total size
            total_size = int(response.headers.get('Content-Length', 0))

            with self.lock:
                self.downloads[task_id]['total_size'] = total_size

            # Download in chunks
            chunk_size = 8192
            downloaded = 0

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            with open(dest_path, 'wb') as f:
                while True:
                    with self.lock:
                        if self.downloads[task_id]['status'] == 'cancelled':
                            break

                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    # Update progress
                    with self.lock:
                        self.downloads[task_id]['downloaded_size'] = downloaded
                        if total_size > 0:
                            self.downloads[task_id]['progress'] = (downloaded / total_size) * 100

            # Mark as completed
            with self.lock:
                if self.downloads[task_id]['status'] != 'cancelled':
                    self.downloads[task_id]['status'] = 'completed'
                    self.downloads[task_id]['progress'] = 100

        except Exception as e:
            with self.lock:
                self.downloads[task_id]['status'] = 'failed'
                self.downloads[task_id]['error'] = str(e)

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get download status."""
        with self.lock:
            task = self.downloads.get(task_id)
            if task:
                return {
                    'task_id': task['task_id'],
                    'status': task['status'],
                    'progress': task['progress'],
                    'total_size': task['total_size'],
                    'downloaded_size': task['downloaded_size'],
                    'error': task['error'],
                    'filename': task['filename']
                }
        return None

    def cancel_download(self, task_id: str) -> bool:
        """Cancel a download."""
        with self.lock:
            if task_id in self.downloads:
                self.downloads[task_id]['status'] = 'cancelled'
                return True
        return False

    def get_all_downloads(self) -> list:
        """Get all download statuses."""
        with self.lock:
            return [self.get_status(task_id) for task_id in self.downloads.keys()]


# Global instance
simple_downloader = SimpleDownloader()
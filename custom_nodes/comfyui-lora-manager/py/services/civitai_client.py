from datetime import datetime
import aiohttp
import os
import logging
import asyncio
from email.parser import Parser
from typing import Optional, Dict, Tuple, List
from urllib.parse import unquote

logger = logging.getLogger(__name__)

class CivitaiClient:
    _instance = None
    _lock = asyncio.Lock()
    
    @classmethod
    async def get_instance(cls):
        """Get singleton instance of CivitaiClient"""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        # Check if already initialized for singleton pattern
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.base_url = "https://civitai.com/api/v1"
        self.headers = {
            'User-Agent': 'ComfyUI-LoRA-Manager/1.0'
        }
        self._session = None
        self._session_created_at = None
        # Set default buffer size to 1MB for higher throughput
        self.chunk_size = 1024 * 1024
    
    @property
    async def session(self) -> aiohttp.ClientSession:
        """Lazy initialize the session"""
        if self._session is None:
            # Optimize TCP connection parameters
            connector = aiohttp.TCPConnector(
                ssl=True,
                limit=8,  # Increase from 3 to 8 for better parallelism
                ttl_dns_cache=300,  # Enable DNS caching with reasonable timeout
                force_close=False,  # Keep connections for reuse
                enable_cleanup_closed=True
            )
            trust_env = True  # Allow using system environment proxy settings
            # Configure timeout parameters - increase read timeout for large files
            timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=120)
            self._session = aiohttp.ClientSession(
                connector=connector, 
                trust_env=trust_env,
                timeout=timeout
            )
            self._session_created_at = datetime.now()
        return self._session
    
    async def _ensure_fresh_session(self):
        """Refresh session if it's been open too long"""
        if self._session is not None:
            if not hasattr(self, '_session_created_at') or \
               (datetime.now() - self._session_created_at).total_seconds() > 300:  # 5 minutes
                await self.close()
                self._session = None
        
        return await self.session

    def _parse_content_disposition(self, header: str) -> str:
        """Parse filename from content-disposition header"""
        if not header:
            return None
        
        # Handle quoted filenames
        if 'filename="' in header:
            start = header.index('filename="') + 10
            end = header.index('"', start)
            return unquote(header[start:end])
        
        # Fallback to original parsing
        disposition = Parser().parsestr(f'Content-Disposition: {header}')
        filename = disposition.get_param('filename')
        if filename:
            return unquote(filename)
        return None

    def _get_request_headers(self) -> dict:
        """Get request headers with optional API key"""
        headers = {
            'User-Agent': 'ComfyUI-LoRA-Manager/1.0',
            'Content-Type': 'application/json'
        }
        
        from .settings_manager import settings
        api_key = settings.get('civitai_api_key')
        if (api_key):
            headers['Authorization'] = f'Bearer {api_key}'
            
        return headers

    async def _download_file(self, url: str, save_dir: str, default_filename: str, progress_callback=None) -> Tuple[bool, str]:
        """Download file with content-disposition support and progress tracking

        Args:
            url: Download URL
            save_dir: Directory to save the file
            default_filename: Fallback filename if none provided in headers
            progress_callback: Optional async callback function for progress updates (0-100)

        Returns:
            Tuple[bool, str]: (success, save_path or error message)
        """
        logger.debug(f"Resolving DNS for: {url}")
        session = await self._ensure_fresh_session()
        try:
            headers = self._get_request_headers()
            
            # Add Range header to allow resumable downloads
            headers['Accept-Encoding'] = 'identity'  # Disable compression for better chunked downloads
            
            logger.debug(f"Starting download from: {url}")
            async with session.get(url, headers=headers, allow_redirects=True) as response:
                if response.status != 200:
                    # Handle 401 unauthorized responses
                    if response.status == 401:
                        logger.warning(f"Unauthorized access to resource: {url} (Status 401)")
                        
                        return False, "Invalid or missing CivitAI API key, or early access restriction."
                    
                    # Handle other client errors that might be permission-related
                    if response.status == 403:
                        logger.warning(f"Forbidden access to resource: {url} (Status 403)")
                        return False, "Access forbidden: You don't have permission to download this file."
                    
                    # Generic error response for other status codes
                    logger.error(f"Download failed for {url} with status {response.status}")
                    return False, f"Download failed with status {response.status}"

                # Get filename from content-disposition header
                content_disposition = response.headers.get('Content-Disposition')
                filename = self._parse_content_disposition(content_disposition)
                if not filename:
                    filename = default_filename
                
                save_path = os.path.join(save_dir, filename)
                
                # Get total file size for progress calculation
                total_size = int(response.headers.get('content-length', 0))
                current_size = 0
                last_progress_report_time = datetime.now()

                # Stream download to file with progress updates using larger buffer
                with open(save_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            current_size += len(chunk)
                            
                            # Limit progress update frequency to reduce overhead
                            now = datetime.now()
                            time_diff = (now - last_progress_report_time).total_seconds()
                            
                            if progress_callback and total_size and time_diff >= 1.0:
                                progress = (current_size / total_size) * 100
                                await progress_callback(progress)
                                last_progress_report_time = now
                
                # Ensure 100% progress is reported
                if progress_callback:
                    await progress_callback(100)
                        
                return True, save_path
                
        except aiohttp.ClientError as e:
            logger.error(f"Network error during download: {e}")
            return False, f"Network error: {str(e)}"
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False, str(e)

    async def get_model_by_hash(self, model_hash: str) -> Optional[Dict]:
        try:
            session = await self._ensure_fresh_session()
            async with session.get(f"{self.base_url}/model-versions/by-hash/{model_hash}") as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return None

    async def download_preview_image(self, image_url: str, save_path: str):
        try:
            session = await self._ensure_fresh_session()
            async with session.get(image_url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(save_path, 'wb') as f:
                        f.write(content)
                    return True
                return False
        except Exception as e:
            print(f"Download Error: {str(e)}")
            return False
            
    async def get_model_versions(self, model_id: str) -> List[Dict]:
        """Get all versions of a model with local availability info"""
        try:
            session = await self._ensure_fresh_session()  # Use fresh session
            async with session.get(f"{self.base_url}/models/{model_id}") as response:
                if response.status != 200:
                    return None
                data = await response.json()
                # Also return model type along with versions
                return {
                    'modelVersions': data.get('modelVersions', []),
                    'type': data.get('type', '')
                }
        except Exception as e:
            logger.error(f"Error fetching model versions: {e}")
            return None
            
    async def get_model_version(self, model_id: int = None, version_id: int = None) -> Optional[Dict]:
        """Get specific model version with additional metadata
        
        Args:
            model_id: The Civitai model ID (optional if version_id is provided)
            version_id: Optional specific version ID to retrieve
            
        Returns:
            Optional[Dict]: The model version data with additional fields or None if not found
        """
        try:
            session = await self._ensure_fresh_session()
            headers = self._get_request_headers()
            
            # Case 1: Only version_id is provided
            if model_id is None and version_id is not None:
                # First get the version info to extract model_id
                async with session.get(f"{self.base_url}/model-versions/{version_id}", headers=headers) as response:
                    if response.status != 200:
                        return None
                    
                    version = await response.json()
                    model_id = version.get('modelId')
                    
                    if not model_id:
                        logger.error(f"No modelId found in version {version_id}")
                        return None
                
                # Now get the model data for additional metadata
                async with session.get(f"{self.base_url}/models/{model_id}") as response:
                    if response.status != 200:
                        return version  # Return version without additional metadata
                    
                    model_data = await response.json()
                    
                    # Enrich version with model data
                    version['model']['description'] = model_data.get("description")
                    version['model']['tags'] = model_data.get("tags", [])
                    version['creator'] = model_data.get("creator")
                    
                    return version
            
            # Case 2: model_id is provided (with or without version_id)
            elif model_id is not None:
                # Step 1: Get model data to find version_id if not provided and get additional metadata
                async with session.get(f"{self.base_url}/models/{model_id}") as response:
                    if response.status != 200:
                        return None
                        
                    data = await response.json()
                    model_versions = data.get('modelVersions', [])
                    
                    # Step 2: Determine the version_id to use
                    target_version_id = version_id
                    if target_version_id is None:
                        target_version_id = model_versions[0].get('id')
                
                # Step 3: Get detailed version info using the version_id
                async with session.get(f"{self.base_url}/model-versions/{target_version_id}", headers=headers) as response:
                    if response.status != 200:
                        return None
                    
                    version = await response.json()
                    
                    # Step 4: Enrich version_info with model data
                    # Add description and tags from model data
                    version['model']['description'] = data.get("description")
                    version['model']['tags'] = data.get("tags", [])
                    
                    # Add creator from model data
                    version['creator'] = data.get("creator")
                    
                    return version
            
            # Case 3: Neither model_id nor version_id provided
            else:
                logger.error("Either model_id or version_id must be provided")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching model version: {e}")
            return None

    async def get_model_version_info(self, version_id: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Fetch model version metadata from Civitai
        
        Args:
            version_id: The Civitai model version ID
            
        Returns:
            Tuple[Optional[Dict], Optional[str]]: A tuple containing:
                - The model version data or None if not found
                - An error message if there was an error, or None on success
        """
        try:
            session = await self._ensure_fresh_session()
            url = f"{self.base_url}/model-versions/{version_id}"
            headers = self._get_request_headers()
            
            logger.debug(f"Resolving DNS for model version info: {url}")
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    logger.debug(f"Successfully fetched model version info for: {version_id}")
                    return await response.json(), None
                
                # Handle specific error cases
                if response.status == 404:
                    # Try to parse the error message
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get('error', f"Model not found (status 404)")
                        logger.warning(f"Model version not found: {version_id} - {error_msg}")
                        return None, error_msg
                    except:
                        return None, "Model not found (status 404)"
                
                # Other error cases
                logger.error(f"Failed to fetch model info for {version_id} (status {response.status})")
                return None, f"Failed to fetch model info (status {response.status})"
        except Exception as e:
            error_msg = f"Error fetching model version info: {e}"
            logger.error(error_msg)
            return None, error_msg

    async def get_model_metadata(self, model_id: str) -> Tuple[Optional[Dict], int]:
        """Fetch model metadata (description, tags, and creator info) from Civitai API
        
        Args:
            model_id: The Civitai model ID
            
        Returns:
            Tuple[Optional[Dict], int]: A tuple containing:
                - A dictionary with model metadata or None if not found
                - The HTTP status code from the request
        """
        try:
            session = await self._ensure_fresh_session()
            headers = self._get_request_headers()
            url = f"{self.base_url}/models/{model_id}"
            
            async with session.get(url, headers=headers) as response:
                status_code = response.status
                
                if status_code != 200:
                    logger.warning(f"Failed to fetch model metadata: Status {status_code}")
                    return None, status_code
                
                data = await response.json()
                
                # Extract relevant metadata
                metadata = {
                    "description": data.get("description") or "No model description available",
                    "tags": data.get("tags", []),
                    "creator": {
                        "username": data.get("creator", {}).get("username"),
                        "image": data.get("creator", {}).get("image")
                    }
                }
                
                if metadata["description"] or metadata["tags"] or metadata["creator"]["username"]:
                    return metadata, status_code
                else:
                    logger.warning(f"No metadata found for model {model_id}")
                    return None, status_code
                
        except Exception as e:
            logger.error(f"Error fetching model metadata: {e}", exc_info=True)
            return None, 0

    # Keep old method for backward compatibility, delegating to the new one
    async def get_model_description(self, model_id: str) -> Optional[str]:
        """Fetch the model description from Civitai API (Legacy method)"""
        metadata, _ = await self.get_model_metadata(model_id)
        return metadata.get("description") if metadata else None

    async def close(self):
        """Close the session if it exists"""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _get_hash_from_civitai(self, model_version_id: str) -> Optional[str]:
        """Get hash from Civitai API"""
        try:
            session = await self._ensure_fresh_session()
            if not session:
                return None
            
            version_info = await session.get(f"{self.base_url}/model-versions/{model_version_id}")
            
            if not version_info or not version_info.json().get('files'):
                return None
            
            # Get hash from the first file
            for file_info in version_info.json().get('files', []):
                if file_info.get('hashes', {}).get('SHA256'):
                    # Convert hash to lowercase to standardize
                    hash_value = file_info['hashes']['SHA256'].lower()
                    return hash_value
                
            return None
        except Exception as e:
            logger.error(f"Error getting hash from Civitai: {e}")
            return None

    async def get_image_info(self, image_id: str) -> Optional[Dict]:
        """Fetch image information from Civitai API
        
        Args:
            image_id: The Civitai image ID
            
        Returns:
            Optional[Dict]: The image data or None if not found
        """
        try:
            session = await self._ensure_fresh_session()
            headers = self._get_request_headers()
            url = f"{self.base_url}/images?imageId={image_id}&nsfw=X"
            
            logger.debug(f"Fetching image info for ID: {image_id}")
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and "items" in data and len(data["items"]) > 0:
                        logger.debug(f"Successfully fetched image info for ID: {image_id}")
                        return data["items"][0]
                    logger.warning(f"No image found with ID: {image_id}")
                    return None
                
                logger.error(f"Failed to fetch image info for ID: {image_id} (status {response.status})")
                return None
        except Exception as e:
            error_msg = f"Error fetching image info: {e}"
            logger.error(error_msg)
            return None

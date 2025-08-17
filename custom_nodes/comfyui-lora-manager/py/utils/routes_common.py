import os
import json
import logging
from typing import Dict, List, Callable, Awaitable
from aiohttp import web

from .model_utils import determine_base_model
from .constants import PREVIEW_EXTENSIONS, CARD_PREVIEW_WIDTH
from ..config import config
from ..services.civitai_client import CivitaiClient
from ..services.service_registry import ServiceRegistry
from ..utils.exif_utils import ExifUtils
from ..utils.metadata_manager import MetadataManager
from ..services.download_manager import DownloadManager
from ..services.websocket_manager import ws_manager

logger = logging.getLogger(__name__)


class ModelRouteUtils:
    """Shared utilities for model routes (LoRAs, Checkpoints, etc.)"""

    @staticmethod
    async def load_local_metadata(metadata_path: str) -> Dict:
        """Load local metadata file"""
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata from {metadata_path}: {e}")
        return {}

    @staticmethod
    async def handle_not_found_on_civitai(metadata_path: str, local_metadata: Dict) -> None:
        """Handle case when model is not found on CivitAI"""
        local_metadata['from_civitai'] = False
        await MetadataManager.save_metadata(metadata_path, local_metadata)

    @staticmethod
    async def update_model_metadata(metadata_path: str, local_metadata: Dict, 
                                  civitai_metadata: Dict, client: CivitaiClient) -> None:
        """Update local metadata with CivitAI data"""
        # Save existing trainedWords and customImages if they exist
        existing_civitai = local_metadata.get('civitai') or {}  # Use empty dict if None

        # Create a new civitai metadata by updating existing with new
        merged_civitai = existing_civitai.copy()
        merged_civitai.update(civitai_metadata)

        # Special handling for trainedWords - ensure we don't lose any existing trained words
        if 'trainedWords' in existing_civitai:
            existing_trained_words = existing_civitai.get('trainedWords', [])
            new_trained_words = civitai_metadata.get('trainedWords', [])
            # Use a set to combine words without duplicates, then convert back to list
            merged_trained_words = list(set(existing_trained_words + new_trained_words))
            merged_civitai['trainedWords'] = merged_trained_words

        # Update local metadata with merged civitai data
        local_metadata['civitai'] = merged_civitai
        local_metadata['from_civitai'] = True
        
        # Update model name if available
        if 'model' in civitai_metadata:
            if civitai_metadata.get('model', {}).get('name'):
                local_metadata['model_name'] = civitai_metadata['model']['name']
        
            # Extract model metadata directly from civitai_metadata if available
            model_metadata = None
            
            if 'model' in civitai_metadata and civitai_metadata.get('model'):
                # Data is already available in the response from get_model_version
                model_metadata = {
                    'description': civitai_metadata.get('model', {}).get('description', ''),
                    'tags': civitai_metadata.get('model', {}).get('tags', []),
                    'creator': civitai_metadata.get('creator', {})
                }
            
            # If we have modelId and don't have enough metadata, fetch additional data
            if not model_metadata or not model_metadata.get('description'):
                model_id = civitai_metadata.get('modelId')
                if model_id:
                    fetched_metadata, _ = await client.get_model_metadata(str(model_id))
                    if fetched_metadata:
                        model_metadata = fetched_metadata
            
            # Update local metadata with the model information
            if model_metadata:
                local_metadata['modelDescription'] = model_metadata.get('description', '')
                local_metadata['tags'] = model_metadata.get('tags', [])
                if 'creator' in model_metadata and model_metadata['creator']:
                    local_metadata['civitai']['creator'] = model_metadata['creator']
        
        # Update base model
        local_metadata['base_model'] = determine_base_model(civitai_metadata.get('baseModel'))
        
        # Update preview if needed
        if not local_metadata.get('preview_url') or not os.path.exists(local_metadata['preview_url']):
            first_preview = next((img for img in civitai_metadata.get('images', [])), None)
            if (first_preview):
                # Determine if content is video or image
                is_video = first_preview['type'] == 'video'
                
                if is_video:
                    # For videos use .mp4 extension
                    preview_ext = '.mp4'
                else:
                    # For images use .webp extension
                    preview_ext = '.webp'
                
                base_name = os.path.splitext(os.path.splitext(os.path.basename(metadata_path))[0])[0]
                preview_filename = base_name + preview_ext
                preview_path = os.path.join(os.path.dirname(metadata_path), preview_filename)
                
                if is_video:
                    # Download video as is
                    if await client.download_preview_image(first_preview['url'], preview_path):
                        local_metadata['preview_url'] = preview_path.replace(os.sep, '/')
                        local_metadata['preview_nsfw_level'] = first_preview.get('nsfwLevel', 0)
                else:
                    # For images, download and then optimize to WebP
                    temp_path = preview_path + ".temp"
                    if await client.download_preview_image(first_preview['url'], temp_path):
                        try:
                            # Read the downloaded image
                            with open(temp_path, 'rb') as f:
                                image_data = f.read()
                            
                            # Optimize and convert to WebP
                            optimized_data, _ = ExifUtils.optimize_image(
                                image_data=image_data,
                                target_width=CARD_PREVIEW_WIDTH,
                                format='webp',
                                quality=85,
                                preserve_metadata=False
                            )
                            
                            # Save the optimized WebP image
                            with open(preview_path, 'wb') as f:
                                f.write(optimized_data)
                                
                            # Update metadata
                            local_metadata['preview_url'] = preview_path.replace(os.sep, '/')
                            local_metadata['preview_nsfw_level'] = first_preview.get('nsfwLevel', 0)
                            
                            # Remove the temporary file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                                
                        except Exception as e:
                            logger.error(f"Error optimizing preview image: {e}")
                            # If optimization fails, try to use the downloaded image directly
                            if os.path.exists(temp_path):
                                os.rename(temp_path, preview_path)
                                local_metadata['preview_url'] = preview_path.replace(os.sep, '/')
                                local_metadata['preview_nsfw_level'] = first_preview.get('nsfwLevel', 0)

        # Save updated metadata
        await MetadataManager.save_metadata(metadata_path, local_metadata, True)

    @staticmethod
    async def fetch_and_update_model(
        sha256: str, 
        file_path: str, 
        model_data: dict,
        update_cache_func: Callable[[str, str, Dict], Awaitable[bool]]
    ) -> bool:
        """Fetch and update metadata for a single model
        
        Args:
            sha256: SHA256 hash of the model file
            file_path: Path to the model file
            model_data: The model object in cache to update
            update_cache_func: Function to update the cache with new metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        client = CivitaiClient()
        try:
            # Validate input parameters
            if not isinstance(model_data, dict):
                logger.error(f"Invalid model_data type: {type(model_data)}")
                return False

            metadata_path = os.path.splitext(file_path)[0] + '.metadata.json'
            
            # Check if model metadata exists
            local_metadata = await ModelRouteUtils.load_local_metadata(metadata_path)

            # Fetch metadata from Civitai
            civitai_metadata = await client.get_model_by_hash(sha256)
            if not civitai_metadata:
                # Mark as not from CivitAI if not found
                local_metadata['from_civitai'] = False
                model_data['from_civitai'] = False
                await MetadataManager.save_metadata(file_path, local_metadata)
                return False

            # Update metadata
            await ModelRouteUtils.update_model_metadata(
                metadata_path, 
                local_metadata, 
                civitai_metadata, 
                client
            )
            
            # Update cache object directly using safe .get() method
            update_dict = {
                'model_name': local_metadata.get('model_name'),
                'preview_url': local_metadata.get('preview_url'),
                'from_civitai': True,
                'civitai': civitai_metadata
            }
            model_data.update(update_dict)
            
            # Update cache using the provided function
            await update_cache_func(file_path, file_path, local_metadata)
                
            return True

        except KeyError as e:
            logger.error(f"Error fetching CivitAI data - Missing key: {e} in model_data={model_data}")
            return False
        except Exception as e:
            logger.error(f"Error fetching CivitAI data: {str(e)}", exc_info=True)  # Include stack trace
            return False
        finally:
            await client.close()
    
    @staticmethod
    def filter_civitai_data(data: Dict) -> Dict:
        """Filter relevant fields from CivitAI data"""
        if not data:
            return {}
            
        fields = [
            "id", "modelId", "name", "createdAt", "updatedAt", 
            "publishedAt", "trainedWords", "baseModel", "description",
            "model", "images", "customImages", "creator"
        ]
        return {k: data[k] for k in fields if k in data}

    @staticmethod
    async def delete_model_files(target_dir: str, file_name: str) -> List[str]:
        """Delete model and associated files
        
        Args:
            target_dir: Directory containing the model files
            file_name: Base name of the model file without extension
            
        Returns:
            List of deleted file paths
        """
        patterns = [
            f"{file_name}.safetensors",  # Required
            f"{file_name}.metadata.json",
        ]
        
        # Add all preview file extensions
        for ext in PREVIEW_EXTENSIONS:
            patterns.append(f"{file_name}{ext}")
        
        deleted = []
        main_file = patterns[0]
        main_path = os.path.join(target_dir, main_file).replace(os.sep, '/')
        
        if os.path.exists(main_path):     
            # Delete file
            os.remove(main_path)
            deleted.append(main_path)
        else:
            logger.warning(f"Model file not found: {main_file}")
            
        # Delete optional files
        for pattern in patterns[1:]:
            path = os.path.join(target_dir, pattern)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    deleted.append(pattern)
                except Exception as e:
                    logger.warning(f"Failed to delete {pattern}: {e}")
                    
        return deleted
    
    @staticmethod
    def get_multipart_ext(filename):
        """Get extension that may have multiple parts like .metadata.json or .metadata.json.bak"""
        parts = filename.split(".")
        if len(parts) == 3:  # If contains 2-part extension
            return "." + ".".join(parts[-2:])  # Take the last two parts, like ".metadata.json"
        elif len(parts) >= 4:  # If contains 3-part or more extensions
            return "." + ".".join(parts[-3:])  # Take the last three parts, like ".metadata.json.bak"
        return os.path.splitext(filename)[1]  # Otherwise take the regular extension, like ".safetensors"

    # New common endpoint handlers

    @staticmethod
    async def handle_delete_model(request: web.Request, scanner) -> web.Response:
        """Handle model deletion request
        
        Args:
            request: The aiohttp request
            scanner: The model scanner instance with cache management methods
            
        Returns:
            web.Response: The HTTP response
        """
        try:
            data = await request.json()
            file_path = data.get('file_path')
            if not file_path:
                return web.Response(text='Model path is required', status=400)

            target_dir = os.path.dirname(file_path)
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            
            deleted_files = await ModelRouteUtils.delete_model_files(
                target_dir, 
                file_name
            )
            
            # Remove from cache
            cache = await scanner.get_cached_data()
            cache.raw_data = [item for item in cache.raw_data if item['file_path'] != file_path]
            await cache.resort()

            # Update hash index if available
            if hasattr(scanner, '_hash_index') and scanner._hash_index:
                scanner._hash_index.remove_by_path(file_path)
            
            return web.json_response({
                'success': True,
                'deleted_files': deleted_files
            })
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}", exc_info=True)
            return web.Response(text=str(e), status=500)

    @staticmethod
    async def handle_fetch_civitai(request: web.Request, scanner) -> web.Response:
        """Handle CivitAI metadata fetch request
        
        Args:
            request: The aiohttp request
            scanner: The model scanner instance with cache management methods
            
        Returns:
            web.Response: The HTTP response with metadata on success
        """
        try:
            data = await request.json()
            metadata_path = os.path.splitext(data['file_path'])[0] + '.metadata.json'
            
            # Check if model metadata exists
            local_metadata = await ModelRouteUtils.load_local_metadata(metadata_path)
            if not local_metadata or not local_metadata.get('sha256'):
                return web.json_response({"success": False, "error": "No SHA256 hash found"}, status=400)

            # Create a client for fetching from Civitai
            client = CivitaiClient()
            try:
                # Fetch and update metadata
                civitai_metadata = await client.get_model_by_hash(local_metadata["sha256"])
                if not civitai_metadata:
                    await ModelRouteUtils.handle_not_found_on_civitai(metadata_path, local_metadata)
                    return web.json_response({"success": False, "error": "Not found on CivitAI"}, status=404)

                await ModelRouteUtils.update_model_metadata(metadata_path, local_metadata, civitai_metadata, client)
                
                # Update the cache
                await scanner.update_single_model_cache(data['file_path'], data['file_path'], local_metadata)
                
                # Return the updated metadata along with success status
                return web.json_response({"success": True, "metadata": local_metadata})
            finally:
                await client.close()

        except Exception as e:
            logger.error(f"Error fetching from CivitAI: {e}", exc_info=True)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @staticmethod
    async def handle_replace_preview(request: web.Request, scanner) -> web.Response:
        """Handle preview image replacement request"""
        try:
            reader = await request.multipart()
            
            # Read preview file data
            field = await reader.next()
            if field.name != 'preview_file':
                raise ValueError("Expected 'preview_file' field")
            content_type = field.headers.get('Content-Type', 'image/png')
            
            # Try to get original filename if available
            content_disposition = field.headers.get('Content-Disposition', '')
            original_filename = None
            import re
            filename_match = re.search(r'filename="(.*?)"', content_disposition)
            if filename_match:
                original_filename = filename_match.group(1)
                
            preview_data = await field.read()
            
            # Read model path
            field = await reader.next()
            if field.name != 'model_path':
                raise ValueError("Expected 'model_path' field")
            model_path = (await field.read()).decode()
            
            # Read NSFW level
            nsfw_level = 0  # Default to 0 (unknown)
            field = await reader.next()
            if field and field.name == 'nsfw_level':
                try:
                    nsfw_level = int((await field.read()).decode())
                except (ValueError, TypeError):
                    logger.warning("Invalid NSFW level format, using default 0")
            
            # Save preview file
            base_name = os.path.splitext(os.path.basename(model_path))[0]
            folder = os.path.dirname(model_path)
            
            # Determine format based on content type and original filename
            is_gif = False
            if original_filename and original_filename.lower().endswith('.gif'):
                is_gif = True
            elif content_type.lower() == 'image/gif':
                is_gif = True
                
            # Determine if content is video or image and handle specific formats
            if content_type.startswith('video/'):
                # For videos, preserve original format if possible
                if original_filename:
                    extension = os.path.splitext(original_filename)[1].lower()
                    # Default to .mp4 if no extension or unrecognized
                    if not extension or extension not in ['.mp4', '.webm', '.mov', '.avi']:
                        extension = '.mp4'
                else:
                    # Try to determine extension from content type
                    if 'webm' in content_type:
                        extension = '.webm'
                    else:
                        extension = '.mp4'  # Default
                optimized_data = preview_data  # No optimization for videos
            elif is_gif:
                # Preserve GIF format without optimization
                extension = '.gif'
                optimized_data = preview_data
            else:
                # For other images, optimize and convert to WebP
                optimized_data, _ = ExifUtils.optimize_image(
                    image_data=preview_data,
                    target_width=CARD_PREVIEW_WIDTH,
                    format='webp',
                    quality=85,
                    preserve_metadata=False
                )
                extension = '.webp'
            
            # Delete any existing preview files for this model
            for ext in PREVIEW_EXTENSIONS:
                existing_preview = os.path.join(folder, base_name + ext)
                if os.path.exists(existing_preview):
                    try:
                        os.remove(existing_preview)
                        logger.debug(f"Deleted existing preview: {existing_preview}")
                    except Exception as e:
                        logger.warning(f"Failed to delete existing preview {existing_preview}: {e}")
            
            preview_path = os.path.join(folder, base_name + extension).replace(os.sep, '/')
            
            with open(preview_path, 'wb') as f:
                f.write(optimized_data)
            
            # Update preview path and NSFW level in metadata
            metadata_path = os.path.splitext(model_path)[0] + '.metadata.json'
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Update preview_url and preview_nsfw_level in the metadata dict
                    metadata['preview_url'] = preview_path
                    metadata['preview_nsfw_level'] = nsfw_level
                    
                    await MetadataManager.save_metadata(model_path, metadata)
                except Exception as e:
                    logger.error(f"Error updating metadata: {e}")
            
            # Update preview URL in scanner cache
            await scanner.update_preview_in_cache(model_path, preview_path, nsfw_level)
            
            return web.json_response({
                "success": True,
                "preview_url": config.get_preview_static_url(preview_path),
                "preview_nsfw_level": nsfw_level
            })
            
        except Exception as e:
            logger.error(f"Error replacing preview: {e}", exc_info=True)
            return web.Response(text=str(e), status=500)

    @staticmethod
    async def handle_exclude_model(request: web.Request, scanner) -> web.Response:
        """Handle model exclusion request
        
        Args:
            request: The aiohttp request
            scanner: The model scanner instance with cache management methods
            
        Returns:
            web.Response: The HTTP response
        """
        try:
            data = await request.json()
            file_path = data.get('file_path')
            if not file_path:
                return web.Response(text='Model path is required', status=400)

            # Update metadata to mark as excluded
            metadata_path = os.path.splitext(file_path)[0] + '.metadata.json'
            metadata = await ModelRouteUtils.load_local_metadata(metadata_path)
            metadata['exclude'] = True
            
            # Save updated metadata
            await MetadataManager.save_metadata(file_path, metadata)

            # Update cache
            cache = await scanner.get_cached_data()

            # Find and remove model from cache
            model_to_remove = next((item for item in cache.raw_data if item['file_path'] == file_path), None)
            if model_to_remove:
                # Update tags count
                for tag in model_to_remove.get('tags', []):
                    if tag in scanner._tags_count:
                        scanner._tags_count[tag] = max(0, scanner._tags_count[tag] - 1)
                        if scanner._tags_count[tag] == 0:
                            del scanner._tags_count[tag]

                # Remove from hash index if available
                if hasattr(scanner, '_hash_index') and scanner._hash_index:
                    scanner._hash_index.remove_by_path(file_path)

                # Remove from cache data
                cache.raw_data = [item for item in cache.raw_data if item['file_path'] != file_path]
                await cache.resort()
            
            # Add to excluded models list
            scanner._excluded_models.append(file_path)
            
            return web.json_response({
                'success': True,
                'message': f"Model {os.path.basename(file_path)} excluded"
            })
            
        except Exception as e:
            logger.error(f"Error excluding model: {e}", exc_info=True)
            return web.Response(text=str(e), status=500)

    @staticmethod
    async def handle_download_model(request: web.Request) -> web.Response:
        """Handle model download request"""
        try:
            download_manager = await ServiceRegistry.get_download_manager()
            data = await request.json()
            
            # Get or generate a download ID
            download_id = data.get('download_id', ws_manager.generate_download_id())
            
            # Create progress callback with download ID
            async def progress_callback(progress):
                await ws_manager.broadcast_download_progress(download_id, {
                    'status': 'progress',
                    'progress': progress,
                    'download_id': download_id
                })
            
            # Check which identifier is provided and convert to int
            model_id = None
            model_version_id = None
            
            if data.get('model_id'):
                try:
                    model_id = int(data.get('model_id'))
                except (TypeError, ValueError):
                    return web.json_response({
                        'success': False,
                        'error': "Invalid model_id: Must be an integer"
                    }, status=400)

            # Convert model_version_id to int if provided
            if data.get('model_version_id'):
                try:
                    model_version_id = int(data.get('model_version_id'))
                except (TypeError, ValueError):
                    return web.json_response({
                        'success': False,
                        'error': "Invalid model_version_id: Must be an integer"
                    }, status=400)
            
            # At least one identifier is required
            if not model_id and not model_version_id:
                return web.json_response({
                    'success': False,
                    'error': "Missing required parameter: Please provide either 'model_id' or 'model_version_id'"
                }, status=400)
            
            use_default_paths = data.get('use_default_paths', False)
            
            # Pass the download_id to download_from_civitai
            result = await download_manager.download_from_civitai(
                model_id=model_id,
                model_version_id=model_version_id,
                save_dir=data.get('model_root'),
                relative_path=data.get('relative_path', ''),
                use_default_paths=use_default_paths,
                progress_callback=progress_callback,
                download_id=download_id  # Pass download_id explicitly
            )
            
            # Include download_id in the response
            result['download_id'] = download_id
            
            if not result.get('success', False):
                error_message = result.get('error', 'Unknown error')
                
                # Return 401 for early access errors
                if 'early access' in error_message.lower():
                    logger.warning(f"Early access download failed: {error_message}")
                    return web.json_response({
                        'success': False,
                        'error': f"Early Access Restriction: {error_message}",
                        'download_id': download_id
                    }, status=401)
                
                return web.json_response({
                    'success': False,
                    'error': error_message,
                    'download_id': download_id
                }, status=500)
            
            return web.json_response(result)
            
        except Exception as e:
            error_message = str(e)
            
            # Check if this might be an early access error
            if '401' in error_message:
                logger.warning(f"Early access error (401): {error_message}")
                return web.json_response({
                    'success': False,
                    'error': "Early Access Restriction: This model requires purchase. Please buy early access on Civitai.com."
                }, status=401)
            
            logger.error(f"Error downloading model: {error_message}")
            return web.json_response({
                'success': False,
                'error': error_message
            }, status=500)

    @staticmethod
    async def handle_cancel_download(request: web.Request) -> web.Response:
        """Handle cancellation of a download task
        
        Args:
            request: The aiohttp request
            
        Returns:
            web.Response: The HTTP response
        """
        try:
            download_manager = await ServiceRegistry.get_download_manager()
            download_id = request.match_info.get('download_id')
            if not download_id:
                return web.json_response({
                    'success': False,
                    'error': 'Download ID is required'
                }, status=400)
            
            result = await download_manager.cancel_download(download_id)
            
            # Notify clients about cancellation via WebSocket
            await ws_manager.broadcast_download_progress(download_id, {
                'status': 'cancelled',
                'progress': 0,
                'download_id': download_id,
                'message': 'Download cancelled by user'
            })
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Error cancelling download: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            
    @staticmethod
    async def handle_list_downloads(request: web.Request) -> web.Response:
        """Get list of active downloads
        
        Args:
            request: The aiohttp request
            
        Returns:
            web.Response: The HTTP response with list of downloads
        """
        try:
            download_manager = await ServiceRegistry.get_download_manager()
            result = await download_manager.get_active_downloads()
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Error listing downloads: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    @staticmethod
    async def handle_bulk_delete_models(request: web.Request, scanner) -> web.Response:
        """Handle bulk deletion of models
        
        Args:
            request: The aiohttp request
            scanner: The model scanner instance with cache management methods
            
        Returns:
            web.Response: The HTTP response
        """
        try:
            data = await request.json()
            file_paths = data.get('file_paths', [])
            
            if not file_paths:
                return web.json_response({
                    'success': False, 
                    'error': 'No file paths provided for deletion'
                }, status=400)
            
            # Use the scanner's bulk delete method to handle all cache and file operations
            result = await scanner.bulk_delete_models(file_paths)
            
            return web.json_response({
                'success': result.get('success', False),
                'total_deleted': result.get('total_deleted', 0),
                'total_attempted': result.get('total_attempted', len(file_paths)),
                'results': result.get('results', [])
            })
            
        except Exception as e:
            logger.error(f"Error in bulk delete: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    @staticmethod
    async def handle_relink_civitai(request: web.Request, scanner) -> web.Response:
        """Handle CivitAI metadata re-linking request by model ID and/or version ID
        
        Args:
            request: The aiohttp request
            scanner: The model scanner instance with cache management methods
            
        Returns:
            web.Response: The HTTP response
        """
        try:
            data = await request.json()
            file_path = data.get('file_path')
            model_id = int(data.get('model_id'))
            model_version_id = None 
            if data.get('model_version_id'):
                model_version_id = int(data.get('model_version_id'))
            
            if not file_path or not model_id:
                return web.json_response({"success": False, "error": "Both file_path and model_id are required"}, status=400)
            
            metadata_path = os.path.splitext(file_path)[0] + '.metadata.json'
            
            # Check if model metadata exists
            local_metadata = await ModelRouteUtils.load_local_metadata(metadata_path)
            
            # Create a client for fetching from Civitai
            client = await CivitaiClient.get_instance()
            try:
                # Fetch metadata using get_model_version which includes more comprehensive data
                civitai_metadata = await client.get_model_version(model_id, model_version_id)
                if not civitai_metadata:
                    error_msg = f"Model version not found on CivitAI for ID: {model_id}"
                    if model_version_id:
                        error_msg += f" with version: {model_version_id}"
                    return web.json_response({"success": False, "error": error_msg}, status=404)
                
                # Try to find the primary model file to get the SHA256 hash
                primary_model_file = None
                for file in civitai_metadata.get('files', []):
                    if file.get('primary', False) and file.get('type') == 'Model':
                        primary_model_file = file
                        break
                
                # Update the SHA256 hash in local metadata if available
                if primary_model_file and primary_model_file.get('hashes', {}).get('SHA256'):
                    local_metadata['sha256'] = primary_model_file['hashes']['SHA256'].lower()
                
                # Update metadata with CivitAI information
                await ModelRouteUtils.update_model_metadata(metadata_path, local_metadata, civitai_metadata, client)
                
                # Update the cache
                await scanner.update_single_model_cache(file_path, file_path, local_metadata)
                
                return web.json_response({
                    "success": True,
                    "message": f"Model successfully re-linked to Civitai model {model_id}" + 
                               (f" version {model_version_id}" if model_version_id else ""),
                    "hash": local_metadata.get('sha256', '')
                })
                
            finally:
                await client.close()

        except Exception as e:
            logger.error(f"Error re-linking to CivitAI: {e}", exc_info=True)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @staticmethod
    async def handle_verify_duplicates(request: web.Request, scanner) -> web.Response:
        """Handle verification of duplicate model hashes
        
        Args:
            request: The aiohttp request
            scanner: The model scanner instance with cache management methods
            
        Returns:
            web.Response: The HTTP response with verification results
        """
        try:
            data = await request.json()
            file_paths = data.get('file_paths', [])
            
            if not file_paths:
                return web.json_response({
                    'success': False,
                    'error': 'No file paths provided for verification'
                }, status=400)
            
            # Results tracking
            results = {
                'verified_as_duplicates': True,  # Start true, set to false if any mismatch
                'mismatched_files': [],
                'new_hash_map': {}
            }
            
            # Get expected hash from the first file's metadata
            expected_hash = None
            first_metadata_path = os.path.splitext(file_paths[0])[0] + '.metadata.json'
            first_metadata = await ModelRouteUtils.load_local_metadata(first_metadata_path)
            if first_metadata and 'sha256' in first_metadata:
                expected_hash = first_metadata['sha256'].lower()
            
            # Process each file
            for file_path in file_paths:
                # Skip files that don't exist
                if not os.path.exists(file_path):
                    continue
                    
                # Calculate actual hash
                try:
                    from .file_utils import calculate_sha256
                    actual_hash = await calculate_sha256(file_path)
                    
                    # Get metadata
                    metadata_path = os.path.splitext(file_path)[0] + '.metadata.json'
                    metadata = await ModelRouteUtils.load_local_metadata(metadata_path)
                    
                    # Compare hashes
                    stored_hash = metadata.get('sha256', '').lower()
                    
                    # Set expected hash from first file if not yet set
                    if not expected_hash:
                        expected_hash = stored_hash
                    
                    # Check if hash matches expected hash
                    if actual_hash != expected_hash:
                        results['verified_as_duplicates'] = False
                        results['mismatched_files'].append(file_path)
                        results['new_hash_map'][file_path] = actual_hash
                        
                    # Check if stored hash needs updating
                    if actual_hash != stored_hash:
                        # Update metadata with actual hash
                        metadata['sha256'] = actual_hash
                        
                        # Save updated metadata
                        await MetadataManager.save_metadata(file_path, metadata)
                        
                        # Update cache
                        await scanner.update_single_model_cache(file_path, file_path, metadata)
                except Exception as e:
                    logger.error(f"Error verifying hash for {file_path}: {e}")
                    results['mismatched_files'].append(file_path)
                    results['new_hash_map'][file_path] = "error_calculating_hash"
                    results['verified_as_duplicates'] = False
            
            return web.json_response({
                'success': True,
                **results
            })
            
        except Exception as e:
            logger.error(f"Error verifying duplicate models: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    @staticmethod
    async def handle_rename_model(request: web.Request, scanner) -> web.Response:
        """Handle renaming a model file and its associated files
        
        Args:
            request: The aiohttp request
            scanner: The model scanner instance
            
        Returns:
            web.Response: The HTTP response
        """
        try:
            data = await request.json()
            file_path = data.get('file_path')
            new_file_name = data.get('new_file_name')
            
            if not file_path or not new_file_name:
                return web.json_response({
                    'success': False,
                    'error': 'File path and new file name are required'
                }, status=400)
            
            # Validate the new file name (no path separators or invalid characters)
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            if any(char in new_file_name for char in invalid_chars):
                return web.json_response({
                    'success': False,
                    'error': 'Invalid characters in file name'
                }, status=400)
            
            # Get the directory and current file name
            target_dir = os.path.dirname(file_path)
            old_file_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Check if the target file already exists
            new_file_path = os.path.join(target_dir, f"{new_file_name}.safetensors").replace(os.sep, '/')
            if os.path.exists(new_file_path):
                return web.json_response({
                    'success': False,
                    'error': 'A file with this name already exists'
                }, status=400)
            
            # Define the patterns for associated files
            patterns = [
                f"{old_file_name}.safetensors",  # Required
                f"{old_file_name}.metadata.json",
                f"{old_file_name}.metadata.json.bak",
            ]
            
            # Add all preview file extensions
            for ext in PREVIEW_EXTENSIONS:
                patterns.append(f"{old_file_name}{ext}")
            
            # Find all matching files
            existing_files = []
            for pattern in patterns:
                path = os.path.join(target_dir, pattern)
                if os.path.exists(path):
                    existing_files.append((path, pattern))
            
            # Get the hash from the main file to update hash index
            hash_value = None
            metadata = None
            metadata_path = os.path.join(target_dir, f"{old_file_name}.metadata.json")
            
            if os.path.exists(metadata_path):
                metadata = await ModelRouteUtils.load_local_metadata(metadata_path)
                hash_value = metadata.get('sha256')
            
            # Rename all files
            renamed_files = []
            new_metadata_path = None
            
            for old_path, pattern in existing_files:
                # Get the file extension like .safetensors or .metadata.json
                ext = ModelRouteUtils.get_multipart_ext(pattern)

                # Create the new path
                new_path = os.path.join(target_dir, f"{new_file_name}{ext}").replace(os.sep, '/')
                
                # Rename the file
                os.rename(old_path, new_path)
                renamed_files.append(new_path)
                
                # Keep track of metadata path for later update
                if ext == '.metadata.json':
                    new_metadata_path = new_path
            
            # Update the metadata file with new file name and paths
            if new_metadata_path and metadata:
                # Update file_name, file_path and preview_url in metadata
                metadata['file_name'] = new_file_name
                metadata['file_path'] = new_file_path
                
                # Update preview_url if it exists
                if 'preview_url' in metadata and metadata['preview_url']:
                    old_preview = metadata['preview_url']
                    ext = ModelRouteUtils.get_multipart_ext(old_preview)
                    new_preview = os.path.join(target_dir, f"{new_file_name}{ext}").replace(os.sep, '/')
                    metadata['preview_url'] = new_preview
                
                # Save updated metadata
                await MetadataManager.save_metadata(new_file_path, metadata)
            
            # Update the scanner cache
            if metadata:
                await scanner.update_single_model_cache(file_path, new_file_path, metadata)
                
                # Update recipe files and cache if hash is available and recipe_scanner exists
                if hash_value and hasattr(scanner, 'update_lora_filename_by_hash'):
                    recipe_scanner = await ServiceRegistry.get_recipe_scanner()
                    if recipe_scanner:
                        recipes_updated, cache_updated = await recipe_scanner.update_lora_filename_by_hash(hash_value, new_file_name)
                        logger.info(f"Updated {recipes_updated} recipe files and {cache_updated} cache entries for renamed model")
            
            return web.json_response({
                'success': True,
                'new_file_path': new_file_path,
                'new_preview_path': config.get_preview_static_url(new_preview),
                'renamed_files': renamed_files,
                'reload_required': False
            })
            
        except Exception as e:
            logger.error(f"Error renaming model: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    @staticmethod
    async def handle_save_metadata(request: web.Request, scanner) -> web.Response:
        """Handle saving metadata updates
        
        Args:
            request: The aiohttp request
            scanner: The model scanner instance
            
        Returns:
            web.Response: The HTTP response
        """
        try:
            data = await request.json()
            file_path = data.get('file_path')
            if not file_path:
                return web.Response(text='File path is required', status=400)

            # Remove file path from data to avoid saving it
            metadata_updates = {k: v for k, v in data.items() if k != 'file_path'}
            
            # Get metadata file path
            metadata_path = os.path.splitext(file_path)[0] + '.metadata.json'
            
            # Load existing metadata
            metadata = await ModelRouteUtils.load_local_metadata(metadata_path)

            # Handle nested updates (for civitai.trainedWords)
            for key, value in metadata_updates.items():
                if isinstance(value, dict) and key in metadata and isinstance(metadata[key], dict):
                    # Deep update for nested dictionaries
                    for nested_key, nested_value in value.items():
                        metadata[key][nested_key] = nested_value
                else:
                    # Regular update for top-level keys
                    metadata[key] = value

            # Save updated metadata
            await MetadataManager.save_metadata(file_path, metadata)

            # Update cache
            await scanner.update_single_model_cache(file_path, file_path, metadata)

            # If model_name was updated, resort the cache
            if 'model_name' in metadata_updates:
                cache = await scanner.get_cached_data()
                await cache.resort()

            return web.json_response({'success': True})

        except Exception as e:
            logger.error(f"Error saving metadata: {e}", exc_info=True)
            return web.Response(text=str(e), status=500)

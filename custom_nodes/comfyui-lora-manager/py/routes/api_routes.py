import os
import json
import logging
from aiohttp import web
from typing import Dict
from server import PromptServer # type: ignore

from ..utils.routes_common import ModelRouteUtils
from ..utils.utils import get_lora_info

from ..config import config
from ..services.websocket_manager import ws_manager
import asyncio
from .update_routes import UpdateRoutes
from ..utils.constants import PREVIEW_EXTENSIONS, CARD_PREVIEW_WIDTH, VALID_LORA_TYPES
from ..utils.exif_utils import ExifUtils
from ..utils.metadata_manager import MetadataManager
from ..services.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)

class ApiRoutes:
    """API route handlers for LoRA management"""

    def __init__(self):
        self.scanner = None  # Will be initialized in setup_routes
        self.civitai_client = None  # Will be initialized in setup_routes
        self.download_manager = None  # Will be initialized in setup_routes
        self._download_lock = asyncio.Lock()

    async def initialize_services(self):
        """Initialize services from ServiceRegistry"""
        self.scanner = await ServiceRegistry.get_lora_scanner()
        self.civitai_client = await ServiceRegistry.get_civitai_client()
        self.download_manager = await ServiceRegistry.get_download_manager()

    @classmethod
    def setup_routes(cls, app: web.Application):
        """Register API routes"""
        routes = cls()
        
        # Schedule service initialization on app startup
        app.on_startup.append(lambda _: routes.initialize_services())
        
        app.router.add_post('/api/delete_model', routes.delete_model)
        app.router.add_post('/api/loras/exclude', routes.exclude_model)  # Add new exclude endpoint
        app.router.add_post('/api/fetch-civitai', routes.fetch_civitai)
        app.router.add_post('/api/relink-civitai', routes.relink_civitai)  # Add new relink endpoint
        app.router.add_post('/api/replace_preview', routes.replace_preview)
        app.router.add_get('/api/loras', routes.get_loras)
        app.router.add_post('/api/fetch-all-civitai', routes.fetch_all_civitai)
        app.router.add_get('/ws/fetch-progress', ws_manager.handle_connection)
        app.router.add_get('/ws/download-progress', ws_manager.handle_download_connection)  # Add new WebSocket route for download progress
        app.router.add_get('/ws/init-progress', ws_manager.handle_init_connection)  # Add new WebSocket route
        app.router.add_get('/api/lora-roots', routes.get_lora_roots)
        app.router.add_get('/api/folders', routes.get_folders)
        app.router.add_get('/api/civitai/versions/{model_id}', routes.get_civitai_versions)
        app.router.add_get('/api/civitai/model/version/{modelVersionId}', routes.get_civitai_model_by_version)
        app.router.add_get('/api/civitai/model/hash/{hash}', routes.get_civitai_model_by_hash)
        app.router.add_post('/api/download-model', routes.download_model)
        app.router.add_get('/api/download-model-get', routes.download_model_get)  # Add new GET endpoint
        app.router.add_get('/api/download-progress/{download_id}', routes.get_download_progress)  # Add new endpoint for download progress
        app.router.add_post('/api/move_model', routes.move_model)
        app.router.add_get('/api/lora-model-description', routes.get_lora_model_description)  # Add new route
        app.router.add_post('/api/loras/save-metadata', routes.save_metadata)
        app.router.add_get('/api/lora-preview-url', routes.get_lora_preview_url)  # Add new route
        app.router.add_post('/api/move_models_bulk', routes.move_models_bulk)
        app.router.add_get('/api/loras/top-tags', routes.get_top_tags)  # Add new route for top tags
        app.router.add_get('/api/loras/base-models', routes.get_base_models)  # Add new route for base models
        app.router.add_get('/api/lora-civitai-url', routes.get_lora_civitai_url)  # Add new route for Civitai URL
        app.router.add_post('/api/loras/rename', routes.rename_lora)  # Add new route for renaming LoRA files
        app.router.add_get('/api/loras/scan', routes.scan_loras)  # Add new route for scanning LoRA files
        
        # Add the new trigger words route
        app.router.add_post('/loramanager/get_trigger_words', routes.get_trigger_words)

        # Add new endpoint for letter counts
        app.router.add_get('/api/loras/letter-counts', routes.get_letter_counts)
        
        # Add new endpoints for copying lora data
        app.router.add_get('/api/loras/get-notes', routes.get_lora_notes)
        app.router.add_get('/api/loras/get-trigger-words', routes.get_lora_trigger_words)

        # Add update check routes
        UpdateRoutes.setup_routes(app)

        # Add new endpoints for finding duplicates
        app.router.add_get('/api/loras/find-duplicates', routes.find_duplicate_loras)
        app.router.add_get('/api/loras/find-filename-conflicts', routes.find_filename_conflicts)

        # Add new endpoint for bulk deleting loras
        app.router.add_post('/api/loras/bulk-delete', routes.bulk_delete_loras)

        # Add new endpoint for verifying duplicates
        app.router.add_post('/api/loras/verify-duplicates', routes.verify_duplicates)

    async def delete_model(self, request: web.Request) -> web.Response:
        """Handle model deletion request"""
        if self.scanner is None:
            self.scanner = await ServiceRegistry.get_lora_scanner()
        return await ModelRouteUtils.handle_delete_model(request, self.scanner)

    async def exclude_model(self, request: web.Request) -> web.Response:
        """Handle model exclusion request"""
        if self.scanner is None:
            self.scanner = await ServiceRegistry.get_lora_scanner()
        return await ModelRouteUtils.handle_exclude_model(request, self.scanner)

    async def fetch_civitai(self, request: web.Request) -> web.Response:
        """Handle CivitAI metadata fetch request"""
        if self.scanner is None:
            self.scanner = await ServiceRegistry.get_lora_scanner()
        
        response = await ModelRouteUtils.handle_fetch_civitai(request, self.scanner)
        
        # If successful, format the metadata before returning
        if response.status == 200:
            data = json.loads(response.body.decode('utf-8'))
            if data.get("success") and data.get("metadata"):
                formatted_metadata = self._format_lora_response(data["metadata"])
                return web.json_response({
                    "success": True,
                    "metadata": formatted_metadata
                })
        
        # Otherwise, return the original response
        return response

    async def replace_preview(self, request: web.Request) -> web.Response:
        """Handle preview image replacement request"""
        if self.scanner is None:
            self.scanner = await ServiceRegistry.get_lora_scanner()
        return await ModelRouteUtils.handle_replace_preview(request, self.scanner)
    
    async def scan_loras(self, request: web.Request) -> web.Response:
        """Force a rescan of LoRA files"""
        try:
            # Get full_rebuild parameter from query string, default to false
            full_rebuild = request.query.get('full_rebuild', 'false').lower() == 'true'
                
            await self.scanner.get_cached_data(force_refresh=True, rebuild_cache=full_rebuild)
            return web.json_response({"status": "success", "message": "LoRA scan completed"})
        except Exception as e:
            logger.error(f"Error in scan_loras: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def get_loras(self, request: web.Request) -> web.Response:
        """Handle paginated LoRA data request"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            # Parse query parameters
            page = int(request.query.get('page', '1'))
            page_size = int(request.query.get('page_size', '20'))
            sort_by = request.query.get('sort_by', 'name')
            folder = request.query.get('folder', None)
            search = request.query.get('search', None)
            fuzzy_search = request.query.get('fuzzy', 'false').lower() == 'true'
            
            # Parse search options
            search_options = {
                'filename': request.query.get('search_filename', 'true').lower() == 'true',
                'modelname': request.query.get('search_modelname', 'true').lower() == 'true',
                'tags': request.query.get('search_tags', 'false').lower() == 'true',
                'recursive': request.query.get('recursive', 'false').lower() == 'true'
            }
            
            # Get filter parameters
            base_models = request.query.get('base_models', None)
            tags = request.query.get('tags', None)
            favorites_only = request.query.get('favorites_only', 'false').lower() == 'true'  # New parameter
            
            # New parameter for alphabet filtering
            first_letter = request.query.get('first_letter', None)
            
            # New parameters for recipe filtering
            lora_hash = request.query.get('lora_hash', None)
            lora_hashes = request.query.get('lora_hashes', None)
            
            # Parse filter parameters
            filters = {}
            if base_models:
                filters['base_model'] = base_models.split(',')
            if tags:
                filters['tags'] = tags.split(',')
            
            # Add lora hash filtering options
            hash_filters = {}
            if lora_hash:
                hash_filters['single_hash'] = lora_hash.lower()
            elif lora_hashes:
                hash_filters['multiple_hashes'] = [h.lower() for h in lora_hashes.split(',')]
            
            # Get file data
            data = await self.scanner.get_paginated_data(
                page, 
                page_size, 
                sort_by=sort_by, 
                folder=folder,
                search=search,
                fuzzy_search=fuzzy_search,
                base_models=filters.get('base_model', None),
                tags=filters.get('tags', None),
                search_options=search_options,
                hash_filters=hash_filters,
                favorites_only=favorites_only,  # Pass favorites_only parameter
                first_letter=first_letter  # Pass the new first_letter parameter
            )

            # Get all available folders from cache
            cache = await self.scanner.get_cached_data()
            
            # Convert output to match expected format
            result = {
                'items': [self._format_lora_response(lora) for lora in data['items']],
                'folders': cache.folders,
                'total': data['total'],
                'page': data['page'],
                'page_size': data['page_size'],
                'total_pages': data['total_pages']
            }
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Error retrieving loras: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    def _format_lora_response(self, lora: Dict) -> Dict:
        """Format LoRA data for API response"""
        return {
            "model_name": lora["model_name"],
            "file_name": lora["file_name"],
            "preview_url": config.get_preview_static_url(lora["preview_url"]),
            "preview_nsfw_level": lora.get("preview_nsfw_level", 0),
            "base_model": lora["base_model"],
            "folder": lora["folder"],
            "sha256": lora["sha256"],
            "file_path": lora["file_path"].replace(os.sep, "/"),
            "file_size": lora["size"],
            "modified": lora["modified"],
            "tags": lora["tags"],
            "modelDescription": lora["modelDescription"],
            "from_civitai": lora.get("from_civitai", True),
            "usage_tips": lora.get("usage_tips", ""),
            "notes": lora.get("notes", ""),
            "favorite": lora.get("favorite", False),  # Include favorite status in response
            "civitai": ModelRouteUtils.filter_civitai_data(lora.get("civitai", {}))
        }

    async def fetch_all_civitai(self, request: web.Request) -> web.Response:
        """Fetch CivitAI metadata for all loras in the background"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            cache = await self.scanner.get_cached_data()
            total = len(cache.raw_data)
            processed = 0
            success = 0
            needs_resort = False
            
            # Prepare loras to process
            to_process = [
                lora for lora in cache.raw_data 
                if lora.get('sha256') and (not lora.get('civitai') or 'id' not in lora.get('civitai')) and lora.get('from_civitai', True)  # TODO: for lora not from CivitAI but added traineWords
            ]
            total_to_process = len(to_process)
            
            # Send initial progress
            await ws_manager.broadcast({
                'status': 'started',
                'total': total_to_process,
                'processed': 0,
                'success': 0
            })
            
            for lora in to_process:
                try:
                    original_name = lora.get('model_name')
                    if await ModelRouteUtils.fetch_and_update_model(
                        sha256=lora['sha256'],
                        file_path=lora['file_path'],
                        model_data=lora,
                        update_cache_func=self.scanner.update_single_model_cache
                    ):
                        success += 1
                        if original_name != lora.get('model_name'):
                            needs_resort = True
                    
                    processed += 1
                    
                    # Send progress update
                    await ws_manager.broadcast({
                        'status': 'processing',
                        'total': total_to_process,
                        'processed': processed,
                        'success': success,
                        'current_name': lora.get('model_name', 'Unknown')
                    })
                    
                except Exception as e:
                    logger.error(f"Error fetching CivitAI data for {lora['file_path']}: {e}")
            
            if needs_resort:
                await cache.resort(name_only=True)
            
            # Send completion message
            await ws_manager.broadcast({
                'status': 'completed',
                'total': total_to_process,
                'processed': processed,
                'success': success
            })
                    
            return web.json_response({
                "success": True,
                "message": f"Successfully updated {success} of {processed} processed loras (total: {total})"
            })
            
        except Exception as e:
            # Send error message
            await ws_manager.broadcast({
                'status': 'error',
                'error': str(e)
            })
            logger.error(f"Error in fetch_all_civitai: {e}")
            return web.Response(text=str(e), status=500)

    async def get_lora_roots(self, request: web.Request) -> web.Response:
        """Get all configured LoRA root directories"""
        return web.json_response({
            'roots': config.loras_roots
        })
    
    async def get_folders(self, request: web.Request) -> web.Response:
        """Get all folders in the cache"""
        if self.scanner is None:
            self.scanner = await ServiceRegistry.get_lora_scanner()
            
        cache = await self.scanner.get_cached_data()
        return web.json_response({
            'folders': cache.folders
        })

    async def get_civitai_versions(self, request: web.Request) -> web.Response:
        """Get available versions for a Civitai model with local availability info"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            if self.civitai_client is None:
                self.civitai_client = await ServiceRegistry.get_civitai_client()
                
            model_id = request.match_info['model_id']
            response = await self.civitai_client.get_model_versions(model_id)
            if not response or not response.get('modelVersions'):
                return web.Response(status=404, text="Model not found")
            
            versions = response.get('modelVersions', [])
            model_type = response.get('type', '')
            
            # Check model type - should be LORA, LoCon, or DORA
            if model_type.lower() not in VALID_LORA_TYPES:
                return web.json_response({
                    'error': f"Model type mismatch. Expected LORA or LoCon, got {model_type}"
                }, status=400)
            
            # Check local availability for each version
            for version in versions:
                # Find the model file (type="Model") in the files list
                model_file = next((file for file in version.get('files', []) 
                                  if file.get('type') == 'Model'), None)
                
                if model_file:
                    sha256 = model_file.get('hashes', {}).get('SHA256')
                    if sha256:
                        # Set existsLocally and localPath at the version level
                        version['existsLocally'] = self.scanner.has_hash(sha256)
                        if version['existsLocally']:
                            version['localPath'] = self.scanner.get_path_by_hash(sha256)
                        
                        # Also set the model file size at the version level for easier access
                        version['modelSizeKB'] = model_file.get('sizeKB')
                else:
                    # No model file found in this version
                    version['existsLocally'] = False
                    
            return web.json_response(versions)
        except Exception as e:
            logger.error(f"Error fetching model versions: {e}")
            return web.Response(status=500, text=str(e))
        
    async def get_civitai_model_by_version(self, request: web.Request) -> web.Response:
        """Get CivitAI model details by model version ID"""
        try:
            if self.civitai_client is None:
                self.civitai_client = await ServiceRegistry.get_civitai_client()
                
            model_version_id = request.match_info.get('modelVersionId')
            
            # Get model details from Civitai API    
            model, error_msg = await self.civitai_client.get_model_version_info(model_version_id)
            
            if not model:
                # Log warning for failed model retrieval
                logger.warning(f"Failed to fetch model version {model_version_id}: {error_msg}")
                
                # Determine status code based on error message
                status_code = 404 if error_msg and "not found" in error_msg.lower() else 500
                
                return web.json_response({
                    "success": False,
                    "error": error_msg or "Failed to fetch model information"
                }, status=status_code)
                
            return web.json_response(model)
        except Exception as e:
            logger.error(f"Error fetching model details: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def get_civitai_model_by_hash(self, request: web.Request) -> web.Response:
        """Get CivitAI model details by hash"""
        try:
            if self.civitai_client is None:
                self.civitai_client = await ServiceRegistry.get_civitai_client()
                
            hash = request.match_info.get('hash')
            model = await self.civitai_client.get_model_by_hash(hash)
            return web.json_response(model)
        except Exception as e:
            logger.error(f"Error fetching model details by hash: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def download_model(self, request: web.Request) -> web.Response:
        return await ModelRouteUtils.handle_download_model(request, self.download_manager)

    async def download_model_get(self, request: web.Request) -> web.Response:
        """Handle model download request via GET method
        
        Converts GET parameters to POST format and calls the existing download handler
        
        Args:
            request: The aiohttp request with query parameters
            
        Returns:
            web.Response: The HTTP response
        """
        try:
            # Extract query parameters
            model_id = request.query.get('model_id')
            if not model_id:
                return web.Response(
                    status=400, 
                    text="Missing required parameter: Please provide 'model_id'"
                )
            
            # Get optional parameters
            model_version_id = request.query.get('model_version_id')
            download_id = request.query.get('download_id')
            use_default_paths = request.query.get('use_default_paths', 'false').lower() == 'true'
            
            # Create a data dictionary that mimics what would be received from a POST request
            data = {
                'model_id': model_id
            }
            
            # Add optional parameters only if they are provided
            if model_version_id:
                data['model_version_id'] = model_version_id
                
            if download_id:
                data['download_id'] = download_id
                
            data['use_default_paths'] = use_default_paths
            
            # Create a mock request object with the data
            # Fix: Create a proper Future object and set its result
            future = asyncio.get_event_loop().create_future()
            future.set_result(data)
            
            mock_request = type('MockRequest', (), {
                'json': lambda self=None: future
            })()
            
            # Call the existing download handler
            if self.download_manager is None:
                self.download_manager = await ServiceRegistry.get_download_manager()
                
            return await ModelRouteUtils.handle_download_model(mock_request, self.download_manager)
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error downloading model via GET: {error_message}", exc_info=True)
            return web.Response(status=500, text=error_message)

    async def get_download_progress(self, request: web.Request) -> web.Response:
        """Handle request for download progress by download_id"""
        try:
            # Get download_id from URL path
            download_id = request.match_info.get('download_id')
            if not download_id:
                return web.json_response({
                    'success': False,
                    'error': 'Download ID is required'
                }, status=400)
            
            # Get progress information from websocket manager
            progress_data = ws_manager.get_download_progress(download_id)
            
            if progress_data is None:
                return web.json_response({
                    'success': False,
                    'error': 'Download ID not found'
                }, status=404)
            
            return web.json_response({
                'success': True,
                'progress': progress_data.get('progress', 0)
            })
        except Exception as e:
            logger.error(f"Error getting download progress: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def move_model(self, request: web.Request) -> web.Response:
        """Handle model move request"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            data = await request.json()
            file_path = data.get('file_path') # full path of the model file, e.g. /path/to/model.safetensors
            target_path = data.get('target_path') # folder path to move the model to, e.g. /path/to/target_folder
            
            if not file_path or not target_path:
                return web.Response(text='File path and target path are required', status=400)

            # Check if source and destination are the same
            source_dir = os.path.dirname(file_path)
            if os.path.normpath(source_dir) == os.path.normpath(target_path):
                logger.info(f"Source and target directories are the same: {source_dir}")
                return web.json_response({'success': True, 'message': 'Source and target directories are the same'})

            # Check if target file already exists
            file_name = os.path.basename(file_path)
            target_file_path = os.path.join(target_path, file_name).replace(os.sep, '/')
            
            if os.path.exists(target_file_path):
                return web.json_response({
                    'success': False, 
                    'error': f"Target file already exists: {target_file_path}"
                }, status=409)  # 409 Conflict

            # Call scanner to handle the move operation
            success = await self.scanner.move_model(file_path, target_path)
            
            if success:
                return web.json_response({'success': True})
            else:
                return web.Response(text='Failed to move model', status=500)
                
        except Exception as e:
            logger.error(f"Error moving model: {e}", exc_info=True)
            return web.Response(text=str(e), status=500)

    @classmethod
    async def cleanup(cls):
        """Add cleanup method for application shutdown"""
        # Now we don't need to store an instance, as services are managed by ServiceRegistry
        civitai_client = await ServiceRegistry.get_civitai_client()
        if civitai_client:
            await civitai_client.close()

    async def save_metadata(self, request: web.Request) -> web.Response:
        """Handle saving metadata updates"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
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
            await self.scanner.update_single_model_cache(file_path, file_path, metadata)

            # If model_name was updated, resort the cache
            if 'model_name' in metadata_updates:
                cache = await self.scanner.get_cached_data()
                await cache.resort(name_only=True)

            return web.json_response({'success': True})

        except Exception as e:
            logger.error(f"Error saving metadata: {e}", exc_info=True)
            return web.Response(text=str(e), status=500)

    async def get_lora_preview_url(self, request: web.Request) -> web.Response:
        """Get the static preview URL for a LoRA file"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            # Get lora file name from query parameters
            lora_name = request.query.get('name')
            if not lora_name:
                return web.Response(text='Lora file name is required', status=400)

            # Get cache data
            cache = await self.scanner.get_cached_data()
            
            # Search for the lora in cache data
            for lora in cache.raw_data:
                file_name = lora['file_name']
                if file_name == lora_name:
                    if preview_url := lora.get('preview_url'):
                        # Convert preview path to static URL
                        static_url = config.get_preview_static_url(preview_url)
                        if static_url:
                            return web.json_response({
                                'success': True,
                                'preview_url': static_url
                            })
                    break

            # If no preview URL found
            return web.json_response({
                'success': False,
                'error': 'No preview URL found for the specified lora'
            }, status=404)

        except Exception as e:
            logger.error(f"Error getting lora preview URL: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_lora_civitai_url(self, request: web.Request) -> web.Response:
        """Get the Civitai URL for a LoRA file"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            # Get lora file name from query parameters
            lora_name = request.query.get('name')
            if not lora_name:
                return web.Response(text='Lora file name is required', status=400)

            # Get cache data
            cache = await self.scanner.get_cached_data()
            
            # Search for the lora in cache data
            for lora in cache.raw_data:
                file_name = lora['file_name']
                if file_name == lora_name:
                    civitai_data = lora.get('civitai', {})
                    model_id = civitai_data.get('modelId')
                    version_id = civitai_data.get('id')
                    
                    if model_id:
                        civitai_url = f"https://civitai.com/models/{model_id}"
                        if version_id:
                            civitai_url += f"?modelVersionId={version_id}"
                            
                        return web.json_response({
                            'success': True,
                            'civitai_url': civitai_url,
                            'model_id': model_id,
                            'version_id': version_id
                        })
                    break

            # If no Civitai data found
            return web.json_response({
                'success': False,
                'error': 'No Civitai data found for the specified lora'
            }, status=404)

        except Exception as e:
            logger.error(f"Error getting lora Civitai URL: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def move_models_bulk(self, request: web.Request) -> web.Response:
        """Handle bulk model move request"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            data = await request.json()
            file_paths = data.get('file_paths', []) # list of full paths of the model files, e.g. ["/path/to/model1.safetensors", "/path/to/model2.safetensors"]
            target_path = data.get('target_path') # folder path to move the models to, e.g. "/path/to/target_folder"
            
            if not file_paths or not target_path:
                return web.Response(text='File paths and target path are required', status=400)

            results = []
            for file_path in file_paths:
                # Check if source and destination are the same
                source_dir = os.path.dirname(file_path)
                if os.path.normpath(source_dir) == os.path.normpath(target_path):
                    results.append({
                        "path": file_path, 
                        "success": True, 
                        "message": "Source and target directories are the same"
                    })
                    continue
                
                # Check if target file already exists
                file_name = os.path.basename(file_path)
                target_file_path = os.path.join(target_path, file_name).replace(os.sep, '/')
                
                if os.path.exists(target_file_path):
                    results.append({
                        "path": file_path, 
                        "success": False, 
                        "message": f"Target file already exists: {target_file_path}"
                    })
                    continue
                
                # Try to move the model
                success = await self.scanner.move_model(file_path, target_path)
                results.append({
                    "path": file_path, 
                    "success": success,
                    "message": "Success" if success else "Failed to move model"
                })
            
            # Count successes and failures
            success_count = sum(1 for r in results if r["success"])
            failure_count = len(results) - success_count
            
            return web.json_response({
                'success': True,
                'message': f'Moved {success_count} of {len(file_paths)} models',
                'results': results,
                'success_count': success_count,
                'failure_count': failure_count
            })
                
        except Exception as e:
            logger.error(f"Error moving models in bulk: {e}", exc_info=True)
            return web.Response(text=str(e), status=500)

    async def get_lora_model_description(self, request: web.Request) -> web.Response:
        """Get model description for a Lora model"""
        try:
            if self.civitai_client is None:
                self.civitai_client = await ServiceRegistry.get_civitai_client()
                
            # Get parameters
            model_id = request.query.get('model_id')
            file_path = request.query.get('file_path')
            
            if not model_id:
                return web.json_response({
                    'success': False, 
                    'error': 'Model ID is required'
                }, status=400)
            
            # Check if we already have the description stored in metadata
            description = None
            tags = []
            creator = {}
            if file_path:
                metadata_path = os.path.splitext(file_path)[0] + '.metadata.json'
                metadata = await ModelRouteUtils.load_local_metadata(metadata_path)
                description = metadata.get('modelDescription')
                tags = metadata.get('tags', [])
                creator = metadata.get('creator', {})
            
            # If description is not in metadata, fetch from CivitAI
            if not description:
                logger.info(f"Fetching model metadata for model ID: {model_id}")
                model_metadata, _ = await self.civitai_client.get_model_metadata(model_id)
                
                if (model_metadata):
                    description = model_metadata.get('description')
                    tags = model_metadata.get('tags', [])
                    creator = model_metadata.get('creator', {})
                
                    # Save the metadata to file if we have a file path and got metadata
                    if file_path:
                        try:
                            metadata_path = os.path.splitext(file_path)[0] + '.metadata.json'
                            metadata = await ModelRouteUtils.load_local_metadata(metadata_path)
                            
                            metadata['modelDescription'] = description
                            metadata['tags'] = tags
                            # Ensure the civitai dict exists
                            if 'civitai' not in metadata:
                                metadata['civitai'] = {}
                            # Store creator in the civitai nested structure
                            metadata['civitai']['creator'] = creator
                            
                            await MetadataManager.save_metadata(file_path, metadata, True)
                        except Exception as e:
                            logger.error(f"Error saving model metadata: {e}")
            
            return web.json_response({
                'success': True,
                'description': description or "<p>No model description available.</p>",
                'tags': tags,
                'creator': creator
            })
            
        except Exception as e:
            logger.error(f"Error getting model metadata: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_top_tags(self, request: web.Request) -> web.Response:
        """Handle request for top tags sorted by frequency"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            # Parse query parameters
            limit = int(request.query.get('limit', '20'))
            
            # Validate limit
            if limit < 1 or limit > 100:
                limit = 20  # Default to a reasonable limit
                
            # Get top tags
            top_tags = await self.scanner.get_top_tags(limit)
            
            return web.json_response({
                'success': True,
                'tags': top_tags
            })
            
        except Exception as e:
            logger.error(f"Error getting top tags: {str(e)}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': 'Internal server error'
            }, status=500)

    async def get_base_models(self, request: web.Request) -> web.Response:
        """Get base models used in loras"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            # Parse query parameters
            limit = int(request.query.get('limit', '20'))
            
            # Validate limit
            if limit < 1 or limit > 100:
                limit = 20  # Default to a reasonable limit
                
            # Get base models
            base_models = await self.scanner.get_base_models(limit)
            
            return web.json_response({
                'success': True,
                'base_models': base_models
            })
        except Exception as e:
            logger.error(f"Error retrieving base models: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def rename_lora(self, request: web.Request) -> web.Response:
        """Handle renaming a LoRA file and its associated files"""
        if self.scanner is None:
            self.scanner = await ServiceRegistry.get_lora_scanner()
            
        return await ModelRouteUtils.handle_rename_model(request, self.scanner)

    async def get_trigger_words(self, request: web.Request) -> web.Response:
        """Get trigger words for specified LoRA models"""
        try:
            json_data = await request.json()
            lora_names = json_data.get("lora_names", [])
            node_ids = json_data.get("node_ids", [])
            
            all_trigger_words = []
            for lora_name in lora_names:
                _, trigger_words = get_lora_info(lora_name)
                all_trigger_words.extend(trigger_words)
            
            # Format the trigger words
            trigger_words_text = ",, ".join(all_trigger_words) if all_trigger_words else ""
            
            # Send update to all connected trigger word toggle nodes
            for node_id in node_ids:
                PromptServer.instance.send_sync("trigger_word_update", {
                    "id": node_id,
                    "message": trigger_words_text
                })
            
            return web.json_response({"success": True})

        except Exception as e:
            logger.error(f"Error getting trigger words: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def get_letter_counts(self, request: web.Request) -> web.Response:
        """Get count of loras for each letter of the alphabet"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            # Get letter counts
            letter_counts = await self.scanner.get_letter_counts()
            
            return web.json_response({
                'success': True,
                'letter_counts': letter_counts
            })
        except Exception as e:
            logger.error(f"Error getting letter counts: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_lora_notes(self, request: web.Request) -> web.Response:
        """Get notes for a specific LoRA file"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            # Get lora file name from query parameters
            lora_name = request.query.get('name')
            if not lora_name:
                return web.Response(text='Lora file name is required', status=400)

            # Get cache data
            cache = await self.scanner.get_cached_data()
            
            # Search for the lora in cache data
            for lora in cache.raw_data:
                file_name = lora['file_name']
                if file_name == lora_name:
                    notes = lora.get('notes', '')
                    
                    return web.json_response({
                        'success': True,
                        'notes': notes
                    })

            # If lora not found
            return web.json_response({
                'success': False,
                'error': 'LoRA not found in cache'
            }, status=404)

        except Exception as e:
            logger.error(f"Error getting lora notes: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_lora_trigger_words(self, request: web.Request) -> web.Response:
        """Get trigger words for a specific LoRA file"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            # Get lora file name from query parameters
            lora_name = request.query.get('name')
            if not lora_name:
                return web.Response(text='Lora file name is required', status=400)

            # Get cache data
            cache = await self.scanner.get_cached_data()
            
            # Search for the lora in cache data
            for lora in cache.raw_data:
                file_name = lora['file_name']
                if file_name == lora_name:
                    # Get trigger words from civitai data
                    civitai_data = lora.get('civitai', {})
                    trigger_words = civitai_data.get('trainedWords', [])
                    
                    return web.json_response({
                        'success': True,
                        'trigger_words': trigger_words
                    })

            # If lora not found
            return web.json_response({
                'success': False,
                'error': 'LoRA not found in cache'
            }, status=404)

        except Exception as e:
            logger.error(f"Error getting lora trigger words: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def find_duplicate_loras(self, request: web.Request) -> web.Response:
        """Find loras with duplicate SHA256 hashes"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            # Get duplicate hashes from hash index
            duplicates = self.scanner._hash_index.get_duplicate_hashes()
            
            # Format the response
            result = []
            cache = await self.scanner.get_cached_data()
            
            for sha256, paths in duplicates.items():
                group = {
                    "hash": sha256,
                    "models": []
                }
                # Find matching models for each duplicate path
                for path in paths:
                    model = next((m for m in cache.raw_data if m['file_path'] == path), None)
                    if model:
                        group["models"].append(self._format_lora_response(model))
                
                # Add the primary model too
                primary_path = self.scanner._hash_index.get_path(sha256)
                if primary_path and primary_path not in paths:
                    primary_model = next((m for m in cache.raw_data if m['file_path'] == primary_path), None)
                    if primary_model:
                        group["models"].insert(0, self._format_lora_response(primary_model))
                
                if len(group["models"]) > 1:  # Only include if we found multiple models
                    result.append(group)
                
            return web.json_response({
                "success": True,
                "duplicates": result,
                "count": len(result)
            })
        except Exception as e:
            logger.error(f"Error finding duplicate loras: {e}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def find_filename_conflicts(self, request: web.Request) -> web.Response:
        """Find loras with conflicting filenames"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
                
            # Get duplicate filenames from hash index
            duplicates = self.scanner._hash_index.get_duplicate_filenames()
            
            # Format the response
            result = []
            cache = await self.scanner.get_cached_data()
            
            for filename, paths in duplicates.items():
                group = {
                    "filename": filename,
                    "models": []
                }
                # Find matching models for each path
                for path in paths:
                    model = next((m for m in cache.raw_data if m['file_path'] == path), None)
                    if model:
                        group["models"].append(self._format_lora_response(model))
                
                # Find the model from the main index too
                hash_val = self.scanner._hash_index.get_hash_by_filename(filename)
                if hash_val:
                    main_path = self.scanner._hash_index.get_path(hash_val)
                    if main_path and main_path not in paths:
                        main_model = next((m for m in cache.raw_data if m['file_path'] == main_path), None)
                        if main_model:
                            group["models"].insert(0, self._format_lora_response(main_model))
                
                if group["models"]:  # Only include if we found models
                    result.append(group)
                
            return web.json_response({
                "success": True,
                "conflicts": result,
                "count": len(result)
            })
        except Exception as e:
            logger.error(f"Error finding filename conflicts: {e}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def bulk_delete_loras(self, request: web.Request) -> web.Response:
        """Handle bulk deletion of lora models"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_lora_scanner()
            
            return await ModelRouteUtils.handle_bulk_delete_models(request, self.scanner)
                
        except Exception as e:
            logger.error(f"Error in bulk delete loras: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def relink_civitai(self, request: web.Request) -> web.Response:
        """Handle CivitAI metadata re-linking request by model version ID for LoRAs"""
        if self.scanner is None:
            self.scanner = await ServiceRegistry.get_lora_scanner()
        return await ModelRouteUtils.handle_relink_civitai(request, self.scanner)

    async def verify_duplicates(self, request: web.Request) -> web.Response:
        """Handle verification of duplicate lora hashes"""
        if self.scanner is None:
            self.scanner = await ServiceRegistry.get_lora_scanner()
        return await ModelRouteUtils.handle_verify_duplicates(request, self.scanner)

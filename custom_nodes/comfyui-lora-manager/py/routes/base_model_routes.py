from abc import ABC, abstractmethod
import asyncio
import os
import json
import logging
from aiohttp import web
from typing import Dict

import jinja2

from ..utils.routes_common import ModelRouteUtils
from ..services.websocket_manager import ws_manager
from ..services.settings_manager import settings
from ..utils.utils import calculate_relative_path_for_model
from ..utils.constants import AUTO_ORGANIZE_BATCH_SIZE
from ..config import config

logger = logging.getLogger(__name__)

class BaseModelRoutes(ABC):
    """Base route controller for all model types"""
    
    def __init__(self, service):
        """Initialize the route controller
        
        Args:
            service: Model service instance (LoraService, CheckpointService, etc.)
        """
        self.service = service
        self.model_type = service.model_type
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(config.templates_path),
            autoescape=True
        )
    
    def setup_routes(self, app: web.Application, prefix: str):
        """Setup common routes for the model type
        
        Args:
            app: aiohttp application
            prefix: URL prefix (e.g., 'loras', 'checkpoints')
        """
        # Common model management routes
        app.router.add_get(f'/api/{prefix}/list', self.get_models)
        app.router.add_post(f'/api/{prefix}/delete', self.delete_model)
        app.router.add_post(f'/api/{prefix}/exclude', self.exclude_model)
        app.router.add_post(f'/api/{prefix}/fetch-civitai', self.fetch_civitai)
        app.router.add_post(f'/api/{prefix}/relink-civitai', self.relink_civitai)
        app.router.add_post(f'/api/{prefix}/replace-preview', self.replace_preview)
        app.router.add_post(f'/api/{prefix}/save-metadata', self.save_metadata)
        app.router.add_post(f'/api/{prefix}/rename', self.rename_model)
        app.router.add_post(f'/api/{prefix}/bulk-delete', self.bulk_delete_models)
        app.router.add_post(f'/api/{prefix}/verify-duplicates', self.verify_duplicates)
        app.router.add_post(f'/api/{prefix}/move_model', self.move_model)
        app.router.add_post(f'/api/{prefix}/move_models_bulk', self.move_models_bulk)
        app.router.add_get(f'/api/{prefix}/auto-organize', self.auto_organize_models)
        
        # Common query routes
        app.router.add_get(f'/api/{prefix}/top-tags', self.get_top_tags)
        app.router.add_get(f'/api/{prefix}/base-models', self.get_base_models)
        app.router.add_get(f'/api/{prefix}/scan', self.scan_models)
        app.router.add_get(f'/api/{prefix}/roots', self.get_model_roots)
        app.router.add_get(f'/api/{prefix}/folders', self.get_folders)
        app.router.add_get(f'/api/{prefix}/folder-tree', self.get_folder_tree)
        app.router.add_get(f'/api/{prefix}/unified-folder-tree', self.get_unified_folder_tree)
        app.router.add_get(f'/api/{prefix}/find-duplicates', self.find_duplicate_models)
        app.router.add_get(f'/api/{prefix}/find-filename-conflicts', self.find_filename_conflicts)

        # Common Download management
        app.router.add_post(f'/api/download-model', self.download_model)
        app.router.add_get(f'/api/download-model-get', self.download_model_get)
        app.router.add_get(f'/api/cancel-download-get', self.cancel_download_get)
        app.router.add_get(f'/api/download-progress/{{download_id}}', self.get_download_progress)
        
        # CivitAI integration routes
        app.router.add_post(f'/api/{prefix}/fetch-all-civitai', self.fetch_all_civitai)
        # app.router.add_get(f'/api/civitai/versions/{{model_id}}', self.get_civitai_versions)
        
        # Add generic page route
        app.router.add_get(f'/{prefix}', self.handle_models_page)
        
        # Setup model-specific routes
        self.setup_specific_routes(app, prefix)
    
    @abstractmethod
    def setup_specific_routes(self, app: web.Application, prefix: str):
        """Setup model-specific routes - to be implemented by subclasses"""
        pass
    
    async def handle_models_page(self, request: web.Request) -> web.Response:
        """
        Generic handler for model pages (e.g., /loras, /checkpoints).
        Subclasses should set self.template_env and template_name.
        """
        try:
            # Check if the scanner is initializing
            is_initializing = (
                self.service.scanner._cache is None or
                (hasattr(self.service.scanner, 'is_initializing') and callable(self.service.scanner.is_initializing) and self.service.scanner.is_initializing()) or
                (hasattr(self.service.scanner, '_is_initializing') and self.service.scanner._is_initializing)
            )

            template_name = getattr(self, "template_name", None)
            if not self.template_env or not template_name:
                return web.Response(text="Template environment or template name not set", status=500)

            if is_initializing:
                rendered = self.template_env.get_template(template_name).render(
                    folders=[],
                    is_initializing=True,
                    settings=settings,
                    request=request
                )
            else:
                try:
                    cache = await self.service.scanner.get_cached_data(force_refresh=False)
                    rendered = self.template_env.get_template(template_name).render(
                        folders=getattr(cache, "folders", []),
                        is_initializing=False,
                        settings=settings,
                        request=request
                    )
                except Exception as cache_error:
                    logger.error(f"Error loading cache data: {cache_error}")
                    rendered = self.template_env.get_template(template_name).render(
                        folders=[],
                        is_initializing=True,
                        settings=settings,
                        request=request
                    )
            return web.Response(
                text=rendered,
                content_type='text/html'
            )
        except Exception as e:
            logger.error(f"Error handling models page: {e}", exc_info=True)
            return web.Response(
                text="Error loading models page",
                status=500
            )
    
    async def get_models(self, request: web.Request) -> web.Response:
        """Get paginated model data"""
        try:
            # Parse common query parameters
            params = self._parse_common_params(request)
            
            # Get data from service
            result = await self.service.get_paginated_data(**params)
            
            # Format response items
            formatted_result = {
                'items': [await self.service.format_response(item) for item in result['items']],
                'total': result['total'],
                'page': result['page'],
                'page_size': result['page_size'],
                'total_pages': result['total_pages']
            }
            
            return web.json_response(formatted_result)
            
        except Exception as e:
            logger.error(f"Error in get_{self.model_type}s: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
    
    def _parse_common_params(self, request: web.Request) -> Dict:
        """Parse common query parameters"""
        # Parse basic pagination and sorting
        page = int(request.query.get('page', '1'))
        page_size = min(int(request.query.get('page_size', '20')), 100)
        sort_by = request.query.get('sort_by', 'name')
        folder = request.query.get('folder', None)
        search = request.query.get('search', None)
        fuzzy_search = request.query.get('fuzzy_search', 'false').lower() == 'true'
        
        # Parse filter arrays
        base_models = request.query.getall('base_model', [])
        tags = request.query.getall('tag', [])
        favorites_only = request.query.get('favorites_only', 'false').lower() == 'true'
        
        # Parse search options
        search_options = {
            'filename': request.query.get('search_filename', 'true').lower() == 'true',
            'modelname': request.query.get('search_modelname', 'true').lower() == 'true',
            'tags': request.query.get('search_tags', 'false').lower() == 'true',
            'creator': request.query.get('search_creator', 'false').lower() == 'true',
            'recursive': request.query.get('recursive', 'false').lower() == 'true',
        }
        
        # Parse hash filters if provided
        hash_filters = {}
        if 'hash' in request.query:
            hash_filters['single_hash'] = request.query['hash']
        elif 'hashes' in request.query:
            try:
                hash_list = json.loads(request.query['hashes'])
                if isinstance(hash_list, list):
                    hash_filters['multiple_hashes'] = hash_list
            except (json.JSONDecodeError, TypeError):
                pass
        
        return {
            'page': page,
            'page_size': page_size,
            'sort_by': sort_by,
            'folder': folder,
            'search': search,
            'fuzzy_search': fuzzy_search,
            'base_models': base_models,
            'tags': tags,
            'search_options': search_options,
            'hash_filters': hash_filters,
            'favorites_only': favorites_only,
            # Add model-specific parameters
            **self._parse_specific_params(request)
        }
    
    def _parse_specific_params(self, request: web.Request) -> Dict:
        """Parse model-specific parameters - to be overridden by subclasses"""
        return {}
    
    # Common route handlers
    async def delete_model(self, request: web.Request) -> web.Response:
        """Handle model deletion request"""
        return await ModelRouteUtils.handle_delete_model(request, self.service.scanner)
    
    async def exclude_model(self, request: web.Request) -> web.Response:
        """Handle model exclusion request"""
        return await ModelRouteUtils.handle_exclude_model(request, self.service.scanner)
    
    async def fetch_civitai(self, request: web.Request) -> web.Response:
        """Handle CivitAI metadata fetch request"""
        response = await ModelRouteUtils.handle_fetch_civitai(request, self.service.scanner)
        
        # If successful, format the metadata before returning
        if response.status == 200:
            data = json.loads(response.body.decode('utf-8'))
            if data.get("success") and data.get("metadata"):
                formatted_metadata = await self.service.format_response(data["metadata"])
                return web.json_response({
                    "success": True,
                    "metadata": formatted_metadata
                })
        
        return response
    
    async def relink_civitai(self, request: web.Request) -> web.Response:
        """Handle CivitAI metadata re-linking request"""
        return await ModelRouteUtils.handle_relink_civitai(request, self.service.scanner)
    
    async def replace_preview(self, request: web.Request) -> web.Response:
        """Handle preview image replacement"""
        return await ModelRouteUtils.handle_replace_preview(request, self.service.scanner)
    
    async def save_metadata(self, request: web.Request) -> web.Response:
        """Handle saving metadata updates"""
        return await ModelRouteUtils.handle_save_metadata(request, self.service.scanner)
    
    async def rename_model(self, request: web.Request) -> web.Response:
        """Handle renaming a model file and its associated files"""
        return await ModelRouteUtils.handle_rename_model(request, self.service.scanner)
    
    async def bulk_delete_models(self, request: web.Request) -> web.Response:
        """Handle bulk deletion of models"""
        return await ModelRouteUtils.handle_bulk_delete_models(request, self.service.scanner)
    
    async def verify_duplicates(self, request: web.Request) -> web.Response:
        """Handle verification of duplicate model hashes"""
        return await ModelRouteUtils.handle_verify_duplicates(request, self.service.scanner)
    
    async def get_top_tags(self, request: web.Request) -> web.Response:
        """Handle request for top tags sorted by frequency"""
        try:
            limit = int(request.query.get('limit', '20'))
            if limit < 1 or limit > 100:
                limit = 20
                
            top_tags = await self.service.get_top_tags(limit)
            
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
        """Get base models used in models"""
        try:
            limit = int(request.query.get('limit', '20'))
            if limit < 1 or limit > 100:
                limit = 20
                
            base_models = await self.service.get_base_models(limit)
            
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
    
    async def scan_models(self, request: web.Request) -> web.Response:
        """Force a rescan of model files"""
        try:
            full_rebuild = request.query.get('full_rebuild', 'false').lower() == 'true'
            
            await self.service.scan_models(force_refresh=True, rebuild_cache=full_rebuild)
            return web.json_response({
                "status": "success", 
                "message": f"{self.model_type.capitalize()} scan completed"
            })
        except Exception as e:
            logger.error(f"Error in scan_{self.model_type}s: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_model_roots(self, request: web.Request) -> web.Response:
        """Return the model root directories"""
        try:
            roots = self.service.get_model_roots()
            return web.json_response({
                "success": True,
                "roots": roots
            })
        except Exception as e:
            logger.error(f"Error getting {self.model_type} roots: {e}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
        
    async def get_folders(self, request: web.Request) -> web.Response:
        """Get all folders in the cache"""
        try:
            cache = await self.service.scanner.get_cached_data()
            return web.json_response({
                'folders': cache.folders
            })
        except Exception as e:
            logger.error(f"Error getting folders: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def get_folder_tree(self, request: web.Request) -> web.Response:
        """Get hierarchical folder tree structure for download modal"""
        try:
            model_root = request.query.get('model_root')
            if not model_root:
                return web.json_response({
                    'success': False,
                    'error': 'model_root parameter is required'
                }, status=400)
            
            folder_tree = await self.service.get_folder_tree(model_root)
            return web.json_response({
                'success': True,
                'tree': folder_tree
            })
        except Exception as e:
            logger.error(f"Error getting folder tree: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def get_unified_folder_tree(self, request: web.Request) -> web.Response:
        """Get unified folder tree across all model roots"""
        try:
            unified_tree = await self.service.get_unified_folder_tree()
            return web.json_response({
                'success': True,
                'tree': unified_tree
            })
        except Exception as e:
            logger.error(f"Error getting unified folder tree: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def find_duplicate_models(self, request: web.Request) -> web.Response:
        """Find models with duplicate SHA256 hashes"""
        try:
            # Get duplicate hashes from service
            duplicates = self.service.find_duplicate_hashes()
            
            # Format the response
            result = []
            cache = await self.service.scanner.get_cached_data()
            
            for sha256, paths in duplicates.items():
                group = {
                    "hash": sha256,
                    "models": []
                }
                # Find matching models for each path
                for path in paths:
                    model = next((m for m in cache.raw_data if m['file_path'] == path), None)
                    if model:
                        group["models"].append(await self.service.format_response(model))
                
                # Add the primary model too
                primary_path = self.service.get_path_by_hash(sha256)
                if primary_path and primary_path not in paths:
                    primary_model = next((m for m in cache.raw_data if m['file_path'] == primary_path), None)
                    if primary_model:
                        group["models"].insert(0, await self.service.format_response(primary_model))
                
                if len(group["models"]) > 1:  # Only include if we found multiple models
                    result.append(group)
                
            return web.json_response({
                "success": True,
                "duplicates": result,
                "count": len(result)
            })
        except Exception as e:
            logger.error(f"Error finding duplicate {self.model_type}s: {e}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def find_filename_conflicts(self, request: web.Request) -> web.Response:
        """Find models with conflicting filenames"""
        try:
            # Get duplicate filenames from service
            duplicates = self.service.find_duplicate_filenames()
            
            # Format the response
            result = []
            cache = await self.service.scanner.get_cached_data()
            
            for filename, paths in duplicates.items():
                group = {
                    "filename": filename,
                    "models": []
                }
                # Find matching models for each path
                for path in paths:
                    model = next((m for m in cache.raw_data if m['file_path'] == path), None)
                    if model:
                        group["models"].append(await self.service.format_response(model))
                
                # Find the model from the main index too
                hash_val = self.service.scanner.get_hash_by_filename(filename)
                if hash_val:
                    main_path = self.service.get_path_by_hash(hash_val)
                    if main_path and main_path not in paths:
                        main_model = next((m for m in cache.raw_data if m['file_path'] == main_path), None)
                        if main_model:
                            group["models"].insert(0, await self.service.format_response(main_model))
                
                if group["models"]:
                    result.append(group)
                
            return web.json_response({
                "success": True,
                "conflicts": result,
                "count": len(result)
            })
        except Exception as e:
            logger.error(f"Error finding filename conflicts for {self.model_type}s: {e}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
        
    # Download management methods
    async def download_model(self, request: web.Request) -> web.Response:
        """Handle model download request"""
        return await ModelRouteUtils.handle_download_model(request)
    
    async def download_model_get(self, request: web.Request) -> web.Response:
        """Handle model download request via GET method"""
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
            future = asyncio.get_event_loop().create_future()
            future.set_result(data)
            
            mock_request = type('MockRequest', (), {
                'json': lambda self=None: future
            })()
            
            # Call the existing download handler
            return await ModelRouteUtils.handle_download_model(mock_request)
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error downloading model via GET: {error_message}", exc_info=True)
            return web.Response(status=500, text=error_message)
    
    async def cancel_download_get(self, request: web.Request) -> web.Response:
        """Handle GET request for cancelling a download by download_id"""
        try:
            download_id = request.query.get('download_id')
            if not download_id:
                return web.json_response({
                    'success': False,
                    'error': 'Download ID is required'
                }, status=400)
            
            # Create a mock request with match_info for compatibility
            mock_request = type('MockRequest', (), {
                'match_info': {'download_id': download_id}
            })()
            return await ModelRouteUtils.handle_cancel_download(mock_request)
        except Exception as e:
            logger.error(f"Error cancelling download via GET: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
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
    
    async def fetch_all_civitai(self, request: web.Request) -> web.Response:
        """Fetch CivitAI metadata for all models in the background"""
        try:
            cache = await self.service.scanner.get_cached_data()
            total = len(cache.raw_data)
            processed = 0
            success = 0
            needs_resort = False
            
            # Prepare models to process
            to_process = [
                model for model in cache.raw_data 
                if model.get('sha256') and (not model.get('civitai') or 'id' not in model.get('civitai')) and model.get('from_civitai', True)
            ]
            total_to_process = len(to_process)
            
            # Send initial progress
            await ws_manager.broadcast({
                'status': 'started',
                'total': total_to_process,
                'processed': 0,
                'success': 0
            })
            
            # Process each model
            for model in to_process:
                try:
                    original_name = model.get('model_name')
                    if await ModelRouteUtils.fetch_and_update_model(
                        sha256=model['sha256'],
                        file_path=model['file_path'],
                        model_data=model,
                        update_cache_func=self.service.scanner.update_single_model_cache
                    ):
                        success += 1
                        if original_name != model.get('model_name'):
                            needs_resort = True
                    
                    processed += 1
                    
                    # Send progress update
                    await ws_manager.broadcast({
                        'status': 'processing',
                        'total': total_to_process,
                        'processed': processed,
                        'success': success,
                        'current_name': model.get('model_name', 'Unknown')
                    })
                    
                except Exception as e:
                    logger.error(f"Error fetching CivitAI data for {model['file_path']}: {e}")
            
            if needs_resort:
                await cache.resort()
            
            # Send completion message
            await ws_manager.broadcast({
                'status': 'completed',
                'total': total_to_process,
                'processed': processed,
                'success': success
            })
                    
            return web.json_response({
                "success": True,
                "message": f"Successfully updated {success} of {processed} processed {self.model_type}s (total: {total})"
            })
            
        except Exception as e:
            # Send error message
            await ws_manager.broadcast({
                'status': 'error',
                'error': str(e)
            })
            logger.error(f"Error in fetch_all_civitai for {self.model_type}s: {e}")
            return web.Response(text=str(e), status=500)
    
    async def get_civitai_versions(self, request: web.Request) -> web.Response:
        """Get available versions for a Civitai model with local availability info"""
        # This will be implemented by subclasses as they need CivitAI client access
        return web.json_response({
            "error": "Not implemented in base class"
        }, status=501)
    
    # Common model move handlers
    async def move_model(self, request: web.Request) -> web.Response:
        """Handle model move request"""
        try:
            data = await request.json()
            file_path = data.get('file_path')
            target_path = data.get('target_path')
            if not file_path or not target_path:
                return web.Response(text='File path and target path are required', status=400)
            import os
            source_dir = os.path.dirname(file_path)
            if os.path.normpath(source_dir) == os.path.normpath(target_path):
                logger.info(f"Source and target directories are the same: {source_dir}")
                return web.json_response({'success': True, 'message': 'Source and target directories are the same'})
            file_name = os.path.basename(file_path)
            target_file_path = os.path.join(target_path, file_name).replace(os.sep, '/')
            if os.path.exists(target_file_path):
                return web.json_response({
                    'success': False, 
                    'error': f"Target file already exists: {target_file_path}"
                }, status=409)
            success = await self.service.scanner.move_model(file_path, target_path)
            if success:
                return web.json_response({'success': True, 'new_file_path': target_file_path})
            else:
                return web.Response(text='Failed to move model', status=500)
        except Exception as e:
            logger.error(f"Error moving model: {e}", exc_info=True)
            return web.Response(text=str(e), status=500)

    async def move_models_bulk(self, request: web.Request) -> web.Response:
        """Handle bulk model move request"""
        try:
            data = await request.json()
            file_paths = data.get('file_paths', [])
            target_path = data.get('target_path')
            if not file_paths or not target_path:
                return web.Response(text='File paths and target path are required', status=400)
            results = []
            import os
            for file_path in file_paths:
                source_dir = os.path.dirname(file_path)
                if os.path.normpath(source_dir) == os.path.normpath(target_path):
                    results.append({
                        "path": file_path, 
                        "success": True, 
                        "message": "Source and target directories are the same"
                    })
                    continue
                file_name = os.path.basename(file_path)
                target_file_path = os.path.join(target_path, file_name).replace(os.sep, '/')
                if os.path.exists(target_file_path):
                    results.append({
                        "path": file_path, 
                        "success": False, 
                        "message": f"Target file already exists: {target_file_path}"
                    })
                    continue
                success = await self.service.scanner.move_model(file_path, target_path)
                results.append({
                    "path": file_path, 
                    "success": success,
                    "message": "Success" if success else "Failed to move model"
                })
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
    
    async def auto_organize_models(self, request: web.Request) -> web.Response:
        """Auto-organize all models based on current settings"""
        try:
            # Get all models from cache
            cache = await self.service.scanner.get_cached_data()
            all_models = cache.raw_data
            
            # Get model roots for this scanner
            model_roots = self.service.get_model_roots()
            if not model_roots:
                return web.json_response({
                    'success': False,
                    'error': 'No model roots configured'
                }, status=400)
            
            # Check if flat structure is configured for this model type
            path_template = settings.get_download_path_template(self.service.model_type)
            is_flat_structure = not path_template
            
            # Prepare results tracking
            results = []
            total_models = len(all_models)
            processed = 0
            success_count = 0
            failure_count = 0
            skipped_count = 0
            
            # Send initial progress via WebSocket
            await ws_manager.broadcast({
                'type': 'auto_organize_progress',
                'status': 'started',
                'total': total_models,
                'processed': 0,
                'success': 0,
                'failures': 0,
                'skipped': 0
            })
            
            # Process models in batches
            for i in range(0, total_models, AUTO_ORGANIZE_BATCH_SIZE):
                batch = all_models[i:i + AUTO_ORGANIZE_BATCH_SIZE]
                
                for model in batch:
                    try:
                        file_path = model.get('file_path')
                        if not file_path:
                            if len(results) < 100:  # Limit detailed results
                                results.append({
                                    "model": model.get('model_name', 'Unknown'),
                                    "success": False,
                                    "message": "No file path found"
                                })
                            failure_count += 1
                            processed += 1
                            continue
                        
                        # Find which model root this file belongs to
                        current_root = None
                        for root in model_roots:
                            # Normalize paths for comparison
                            normalized_root = os.path.normpath(root).replace(os.sep, '/')
                            normalized_file = os.path.normpath(file_path).replace(os.sep, '/')
                            
                            if normalized_file.startswith(normalized_root):
                                current_root = root
                                break
                        
                        if not current_root:
                            if len(results) < 100:  # Limit detailed results
                                results.append({
                                    "model": model.get('model_name', 'Unknown'),
                                    "success": False,
                                    "message": "Model file not found in any configured root directory"
                                })
                            failure_count += 1
                            processed += 1
                            continue
                        
                        # Handle flat structure case
                        if is_flat_structure:
                            current_dir = os.path.dirname(file_path)
                            # Check if already in root directory
                            if os.path.normpath(current_dir) == os.path.normpath(current_root):
                                skipped_count += 1
                                processed += 1
                                continue
                            
                            # Move to root directory for flat structure
                            target_dir = current_root
                        else:
                            # Calculate new relative path based on settings
                            new_relative_path = calculate_relative_path_for_model(model, self.service.model_type)
                            
                            # If no relative path calculated (insufficient metadata), skip
                            if not new_relative_path:
                                if len(results) < 100:  # Limit detailed results
                                    results.append({
                                        "model": model.get('model_name', 'Unknown'),
                                        "success": False,
                                        "message": "Skipped - insufficient metadata for organization"
                                    })
                                skipped_count += 1
                                processed += 1
                                continue
                            
                            # Calculate target directory
                            target_dir = os.path.join(current_root, new_relative_path).replace(os.sep, '/')
                        
                        current_dir = os.path.dirname(file_path)
                        
                        # Skip if already in correct location
                        if current_dir.replace(os.sep, '/') == target_dir.replace(os.sep, '/'):
                            skipped_count += 1
                            processed += 1
                            continue
                        
                        # Check if target file would conflict
                        file_name = os.path.basename(file_path)
                        target_file_path = os.path.join(target_dir, file_name)
                        
                        if os.path.exists(target_file_path):
                            if len(results) < 100:  # Limit detailed results
                                results.append({
                                    "model": model.get('model_name', 'Unknown'),
                                    "success": False,
                                    "message": f"Target file already exists: {target_file_path}"
                                })
                            failure_count += 1
                            processed += 1
                            continue
                        
                        # Perform the move
                        success = await self.service.scanner.move_model(file_path, target_dir)
                        
                        if success:
                            success_count += 1
                        else:
                            if len(results) < 100:  # Limit detailed results
                                results.append({
                                    "model": model.get('model_name', 'Unknown'),
                                    "success": False,
                                    "message": "Failed to move model"
                                })
                            failure_count += 1
                        
                        processed += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing model {model.get('model_name', 'Unknown')}: {e}", exc_info=True)
                        if len(results) < 100:  # Limit detailed results
                            results.append({
                                "model": model.get('model_name', 'Unknown'),
                                "success": False,
                                "message": f"Error: {str(e)}"
                            })
                        failure_count += 1
                        processed += 1
                
                # Send progress update after each batch
                await ws_manager.broadcast({
                    'type': 'auto_organize_progress',
                    'status': 'processing',
                    'total': total_models,
                    'processed': processed,
                    'success': success_count,
                    'failures': failure_count,
                    'skipped': skipped_count
                })
                
                # Small delay between batches to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            # Send completion message
            await ws_manager.broadcast({
                'type': 'auto_organize_progress',
                'status': 'cleaning',
                'total': total_models,
                'processed': processed,
                'success': success_count,
                'failures': failure_count,
                'skipped': skipped_count,
                'message': 'Cleaning up empty directories...'
            })
            
            # Clean up empty directories after organizing
            from ..utils.utils import remove_empty_dirs
            cleanup_counts = {}
            for root in model_roots:
                removed = remove_empty_dirs(root)
                cleanup_counts[root] = removed
                
            # Send cleanup completed message
            await ws_manager.broadcast({
                'type': 'auto_organize_progress',
                'status': 'completed',
                'total': total_models,
                'processed': processed,
                'success': success_count,
                'failures': failure_count,
                'skipped': skipped_count,
                'cleanup': cleanup_counts
            })
            
            # Prepare response with limited details
            response_data = {
                'success': True,
                'message': f'Auto-organize completed: {success_count} moved, {skipped_count} skipped, {failure_count} failed out of {total_models} total',
                'summary': {
                    'total': total_models,
                    'success': success_count,
                    'skipped': skipped_count,
                    'failures': failure_count,
                    'organization_type': 'flat' if is_flat_structure else 'structured',
                    'cleaned_dirs': cleanup_counts
                }
            }
            
            # Only include detailed results if under limit
            if len(results) <= 100:
                response_data['results'] = results
            else:
                response_data['results_truncated'] = True
                response_data['sample_results'] = results[:50]  # Show first 50 as sample
            
            return web.json_response(response_data)
            
        except Exception as e:
            logger.error(f"Error in auto_organize_models: {e}", exc_info=True)
            
            # Send error message via WebSocket
            await ws_manager.broadcast({
                'type': 'auto_organize_progress',
                'status': 'error',
                'error': str(e)
            })
            
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
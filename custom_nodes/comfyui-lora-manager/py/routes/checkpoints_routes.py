import os
import json
import jinja2
from aiohttp import web
import logging
import asyncio

from ..utils.routes_common import ModelRouteUtils
from ..utils.constants import NSFW_LEVELS
from ..utils.metadata_manager import MetadataManager
from ..services.websocket_manager import ws_manager
from ..services.service_registry import ServiceRegistry
from ..config import config
from ..services.settings_manager import settings
from ..utils.utils import fuzzy_match

logger = logging.getLogger(__name__)

class CheckpointsRoutes:
    """API routes for checkpoint management"""
    
    def __init__(self):
        self.scanner = None  # Will be initialized in setup_routes
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(config.templates_path),
            autoescape=True
        )
        self.download_manager = None  # Will be initialized in setup_routes
        self._download_lock = asyncio.Lock()
        
    async def initialize_services(self):
        """Initialize services from ServiceRegistry"""
        self.scanner = await ServiceRegistry.get_checkpoint_scanner()
        self.download_manager = await ServiceRegistry.get_download_manager()
        
    def setup_routes(self, app):
        """Register routes with the aiohttp app"""
        # Schedule service initialization on app startup
        app.on_startup.append(lambda _: self.initialize_services())
        
        app.router.add_get('/checkpoints', self.handle_checkpoints_page)
        app.router.add_get('/api/checkpoints', self.get_checkpoints)
        app.router.add_post('/api/checkpoints/fetch-all-civitai', self.fetch_all_civitai)
        app.router.add_get('/api/checkpoints/base-models', self.get_base_models)
        app.router.add_get('/api/checkpoints/top-tags', self.get_top_tags)
        app.router.add_get('/api/checkpoints/scan', self.scan_checkpoints)
        app.router.add_get('/api/checkpoints/info/{name}', self.get_checkpoint_info)
        app.router.add_get('/api/checkpoints/roots', self.get_checkpoint_roots)
        app.router.add_get('/api/checkpoints/civitai/versions/{model_id}', self.get_civitai_versions)  # Add new route
        
        # Add new routes for model management similar to LoRA routes
        app.router.add_post('/api/checkpoints/delete', self.delete_model)
        app.router.add_post('/api/checkpoints/exclude', self.exclude_model)  # Add new exclude endpoint
        app.router.add_post('/api/checkpoints/fetch-civitai', self.fetch_civitai)
        app.router.add_post('/api/checkpoints/relink-civitai', self.relink_civitai)  # Add new relink endpoint
        app.router.add_post('/api/checkpoints/replace-preview', self.replace_preview)
        app.router.add_post('/api/checkpoints/save-metadata', self.save_metadata) # Add new route
        app.router.add_post('/api/checkpoints/rename', self.rename_checkpoint)  # Add new rename endpoint

        # Add new routes for finding duplicates and filename conflicts
        app.router.add_get('/api/checkpoints/find-duplicates', self.find_duplicate_checkpoints)
        app.router.add_get('/api/checkpoints/find-filename-conflicts', self.find_filename_conflicts)

        # Add new endpoint for bulk deleting checkpoints
        app.router.add_post('/api/checkpoints/bulk-delete', self.bulk_delete_checkpoints)

        # Add new endpoint for verifying duplicates
        app.router.add_post('/api/checkpoints/verify-duplicates', self.verify_duplicates)

    async def get_checkpoints(self, request):
        """Get paginated checkpoint data"""
        try:
            # Parse query parameters
            page = int(request.query.get('page', '1'))
            page_size = min(int(request.query.get('page_size', '20')), 100)
            sort_by = request.query.get('sort_by', 'name')
            folder = request.query.get('folder', None)
            search = request.query.get('search', None)
            fuzzy_search = request.query.get('fuzzy_search', 'false').lower() == 'true'
            base_models = request.query.getall('base_model', [])
            tags = request.query.getall('tag', [])
            favorites_only = request.query.get('favorites_only', 'false').lower() == 'true'  # Add favorites_only parameter
            
            # Process search options
            search_options = {
                'filename': request.query.get('search_filename', 'true').lower() == 'true',
                'modelname': request.query.get('search_modelname', 'true').lower() == 'true',
                'tags': request.query.get('search_tags', 'false').lower() == 'true',
                'recursive': request.query.get('recursive', 'false').lower() == 'true',
            }
            
            # Process hash filters if provided
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
            
            # Get data from scanner
            result = await self.get_paginated_data(
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                folder=folder,
                search=search,
                fuzzy_search=fuzzy_search,
                base_models=base_models,
                tags=tags,
                search_options=search_options,
                hash_filters=hash_filters,
                favorites_only=favorites_only  # Pass favorites_only parameter
            )
            
            # Format response items
            formatted_result = {
                'items': [self._format_checkpoint_response(cp) for cp in result['items']],
                'total': result['total'],
                'page': result['page'],
                'page_size': result['page_size'],
                'total_pages': result['total_pages']
            }
            
            # Return as JSON
            return web.json_response(formatted_result)
            
        except Exception as e:
            logger.error(f"Error in get_checkpoints: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def get_paginated_data(self, page, page_size, sort_by='name', 
                               folder=None, search=None, fuzzy_search=False,
                               base_models=None, tags=None,
                               search_options=None, hash_filters=None,
                               favorites_only=False):  # Add favorites_only parameter with default False
        """Get paginated and filtered checkpoint data"""
        cache = await self.scanner.get_cached_data()

        # Get default search options if not provided
        if search_options is None:
            search_options = {
                'filename': True,
                'modelname': True,
                'tags': False,
                'recursive': False,
            }

        # Get the base data set
        filtered_data = cache.sorted_by_date if sort_by == 'date' else cache.sorted_by_name
        
        # Apply hash filtering if provided (highest priority)
        if hash_filters:
            single_hash = hash_filters.get('single_hash')
            multiple_hashes = hash_filters.get('multiple_hashes')
            
            if single_hash:
                # Filter by single hash
                single_hash = single_hash.lower()  # Ensure lowercase for matching
                filtered_data = [
                    cp for cp in filtered_data
                    if cp.get('sha256', '').lower() == single_hash
                ]
            elif multiple_hashes:
                # Filter by multiple hashes
                hash_set = set(hash.lower() for hash in multiple_hashes)  # Convert to set for faster lookup
                filtered_data = [
                    cp for cp in filtered_data
                    if cp.get('sha256', '').lower() in hash_set
                ]
            
            # Jump to pagination
            total_items = len(filtered_data)
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_items)
            
            result = {
                'items': filtered_data[start_idx:end_idx],
                'total': total_items,
                'page': page,
                'page_size': page_size,
                'total_pages': (total_items + page_size - 1) // page_size
            }
            
            return result
        
        # Apply SFW filtering if enabled in settings
        if settings.get('show_only_sfw', False):
            filtered_data = [
                cp for cp in filtered_data
                if not cp.get('preview_nsfw_level') or cp.get('preview_nsfw_level') < NSFW_LEVELS['R']
            ]
        
        # Apply favorites filtering if enabled
        if favorites_only:
            filtered_data = [
                cp for cp in filtered_data
                if cp.get('favorite', False) is True
            ]
        
        # Apply folder filtering
        if folder is not None:
            if search_options.get('recursive', False):
                # Recursive folder filtering - include all subfolders
                filtered_data = [
                    cp for cp in filtered_data
                    if cp['folder'].startswith(folder)
                ]
            else:
                # Exact folder filtering
                filtered_data = [
                    cp for cp in filtered_data
                    if cp['folder'] == folder
                ]
        
        # Apply base model filtering
        if base_models and len(base_models) > 0:
            filtered_data = [
                cp for cp in filtered_data
                if cp.get('base_model') in base_models
            ]
        
        # Apply tag filtering
        if tags and len(tags) > 0:
            filtered_data = [
                cp for cp in filtered_data
                if any(tag in cp.get('tags', []) for tag in tags)
            ]
        
        # Apply search filtering
        if search:
            search_results = []
            
            for cp in filtered_data:
                # Search by file name
                if search_options.get('filename', True):
                    if fuzzy_search:
                        if fuzzy_match(cp.get('file_name', ''), search):
                            search_results.append(cp)
                            continue
                    elif search.lower() in cp.get('file_name', '').lower():
                        search_results.append(cp)
                        continue
                
                # Search by model name
                if search_options.get('modelname', True):
                    if fuzzy_search:
                        if fuzzy_match(cp.get('model_name', ''), search):
                            search_results.append(cp)
                            continue
                    elif search.lower() in cp.get('model_name', '').lower():
                        search_results.append(cp)
                        continue
                
                # Search by tags
                if search_options.get('tags', False) and 'tags' in cp:
                    if any((fuzzy_match(tag, search) if fuzzy_search else search.lower() in tag.lower()) for tag in cp['tags']):
                        search_results.append(cp)
                        continue
            
            filtered_data = search_results

        # Calculate pagination
        total_items = len(filtered_data)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_items)
        
        result = {
            'items': filtered_data[start_idx:end_idx],
            'total': total_items,
            'page': page,
            'page_size': page_size,
            'total_pages': (total_items + page_size - 1) // page_size
        }
        
        return result

    def _format_checkpoint_response(self, checkpoint):
        """Format checkpoint data for API response"""
        return {
            "model_name": checkpoint["model_name"],
            "file_name": checkpoint["file_name"],
            "preview_url": config.get_preview_static_url(checkpoint.get("preview_url", "")),
            "preview_nsfw_level": checkpoint.get("preview_nsfw_level", 0),
            "base_model": checkpoint.get("base_model", ""),
            "folder": checkpoint["folder"],
            "sha256": checkpoint.get("sha256", ""),
            "file_path": checkpoint["file_path"].replace(os.sep, "/"),
            "file_size": checkpoint.get("size", 0),
            "modified": checkpoint.get("modified", ""),
            "tags": checkpoint.get("tags", []),
            "modelDescription": checkpoint.get("modelDescription", ""),
            "from_civitai": checkpoint.get("from_civitai", True),
            "notes": checkpoint.get("notes", ""),
            "model_type": checkpoint.get("model_type", "checkpoint"),
            "favorite": checkpoint.get("favorite", False),
            "civitai": ModelRouteUtils.filter_civitai_data(checkpoint.get("civitai", {}))
        }
    
    async def fetch_all_civitai(self, request: web.Request) -> web.Response:
        """Fetch CivitAI metadata for all checkpoints in the background"""
        try:
            cache = await self.scanner.get_cached_data()
            total = len(cache.raw_data)
            processed = 0
            success = 0
            needs_resort = False
            
            # Prepare checkpoints to process
            to_process = [
                cp for cp in cache.raw_data 
                if cp.get('sha256') and (not cp.get('civitai') or 'id' not in cp.get('civitai')) and cp.get('from_civitai', True)
            ]
            total_to_process = len(to_process)
            
            # Send initial progress
            await ws_manager.broadcast({
                'status': 'started',
                'total': total_to_process,
                'processed': 0,
                'success': 0
            })
            
            # Process each checkpoint
            for cp in to_process:
                try:
                    original_name = cp.get('model_name')
                    if await ModelRouteUtils.fetch_and_update_model(
                        sha256=cp['sha256'],
                        file_path=cp['file_path'],
                        model_data=cp,
                        update_cache_func=self.scanner.update_single_model_cache
                    ):
                        success += 1
                        if original_name != cp.get('model_name'):
                            needs_resort = True
                    
                    processed += 1
                    
                    # Send progress update
                    await ws_manager.broadcast({
                        'status': 'processing',
                        'total': total_to_process,
                        'processed': processed,
                        'success': success,
                        'current_name': cp.get('model_name', 'Unknown')
                    })
                    
                except Exception as e:
                    logger.error(f"Error fetching CivitAI data for {cp['file_path']}: {e}")
            
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
                "message": f"Successfully updated {success} of {processed} processed checkpoints (total: {total})"
            })
            
        except Exception as e:
            # Send error message
            await ws_manager.broadcast({
                'status': 'error',
                'error': str(e)
            })
            logger.error(f"Error in fetch_all_civitai for checkpoints: {e}")
            return web.Response(text=str(e), status=500)
        
    async def get_top_tags(self, request: web.Request) -> web.Response:
        """Handle request for top tags sorted by frequency"""
        try:
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

    async def scan_checkpoints(self, request):
        """Force a rescan of checkpoint files"""
        try:
            # Get the full_rebuild parameter and convert to bool, default to False
            full_rebuild = request.query.get('full_rebuild', 'false').lower() == 'true'
            
            await self.scanner.get_cached_data(force_refresh=True, rebuild_cache=full_rebuild)
            return web.json_response({"status": "success", "message": "Checkpoint scan completed"})
        except Exception as e:
            logger.error(f"Error in scan_checkpoints: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def get_checkpoint_info(self, request):
        """Get detailed information for a specific checkpoint by name"""
        try:
            name = request.match_info.get('name', '')
            checkpoint_info = await self.scanner.get_model_info_by_name(name)
            
            if checkpoint_info:
                return web.json_response(checkpoint_info)
            else:
                return web.json_response({"error": "Checkpoint not found"}, status=404)
                
        except Exception as e:
            logger.error(f"Error in get_checkpoint_info: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def handle_checkpoints_page(self, request: web.Request) -> web.Response:
        """Handle GET /checkpoints request"""
        try:
            # Check if the CheckpointScanner is initializing
            # It's initializing if the cache object doesn't exist yet,
            # OR if the scanner explicitly says it's initializing (background task running).
            is_initializing = (
                self.scanner._cache is None or
                (hasattr(self.scanner, '_is_initializing') and self.scanner._is_initializing)
            )

            if is_initializing:
                # If still initializing, return loading page
                template = self.template_env.get_template('checkpoints.html')
                rendered = template.render(
                    folders=[],  # 空文件夹列表
                    is_initializing=True,  # 新增标志
                    settings=settings,  # Pass settings to template
                    request=request  # Pass the request object to the template
                )
                
                logger.info("Checkpoints page is initializing, returning loading page")
            else:
                # 正常流程 - 获取已经初始化好的缓存数据
                try:
                    cache = await self.scanner.get_cached_data(force_refresh=False)
                    template = self.template_env.get_template('checkpoints.html')
                    rendered = template.render(
                        folders=cache.folders,
                        is_initializing=False,
                        settings=settings,  # Pass settings to template
                        request=request  # Pass the request object to the template
                    )
                except Exception as cache_error:
                    logger.error(f"Error loading checkpoints cache data: {cache_error}")
                    # 如果获取缓存失败，也显示初始化页面
                    template = self.template_env.get_template('checkpoints.html')
                    rendered = template.render(
                        folders=[],
                        is_initializing=True,
                        settings=settings,
                        request=request
                    )
                    logger.info("Checkpoints cache error, returning initialization page")
            
            return web.Response(
                text=rendered,
                content_type='text/html'
            )
        except Exception as e:
            logger.error(f"Error handling checkpoints request: {e}", exc_info=True)
            return web.Response(
                text="Error loading checkpoints page",
                status=500
            )

    async def delete_model(self, request: web.Request) -> web.Response:
        """Handle checkpoint model deletion request"""
        return await ModelRouteUtils.handle_delete_model(request, self.scanner)

    async def exclude_model(self, request: web.Request) -> web.Response:
        """Handle checkpoint model exclusion request"""
        return await ModelRouteUtils.handle_exclude_model(request, self.scanner)
    
    async def fetch_civitai(self, request: web.Request) -> web.Response:
        """Handle CivitAI metadata fetch request for checkpoints"""
        response = await ModelRouteUtils.handle_fetch_civitai(request, self.scanner)
        
        # If successful, format the metadata before returning
        if response.status == 200:
            data = json.loads(response.body.decode('utf-8'))
            if data.get("success") and data.get("metadata"):
                formatted_metadata = self._format_checkpoint_response(data["metadata"])
                return web.json_response({
                    "success": True,
                    "metadata": formatted_metadata
                })
        
        # Otherwise, return the original response
        return response
    
    async def replace_preview(self, request: web.Request) -> web.Response:
        """Handle preview image replacement for checkpoints"""
        return await ModelRouteUtils.handle_replace_preview(request, self.scanner)

    async def get_checkpoint_roots(self, request):
        """Return the checkpoint root directories"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_checkpoint_scanner()
                
            roots = self.scanner.get_model_roots()
            return web.json_response({
                "success": True,
                "roots": roots
            })
        except Exception as e:
            logger.error(f"Error getting checkpoint roots: {e}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def save_metadata(self, request: web.Request) -> web.Response:
        """Handle saving metadata updates for checkpoints"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_checkpoint_scanner()
                
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

            # Update metadata
            metadata.update(metadata_updates)

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
            logger.error(f"Error saving checkpoint metadata: {e}", exc_info=True)
            return web.Response(text=str(e), status=500)

    async def get_civitai_versions(self, request: web.Request) -> web.Response:
        """Get available versions for a Civitai checkpoint model with local availability info"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_checkpoint_scanner()
                
            # Get the civitai client from service registry
            civitai_client = await ServiceRegistry.get_civitai_client()
                
            model_id = request.match_info['model_id']
            response = await civitai_client.get_model_versions(model_id)
            if not response or not response.get('modelVersions'):
                return web.Response(status=404, text="Model not found")
            
            versions = response.get('modelVersions', [])
            model_type = response.get('type', '')
            
            # Check model type - should be Checkpoint
            if (model_type.lower() != 'checkpoint'):
                return web.json_response({
                    'error': f"Model type mismatch. Expected Checkpoint, got {model_type}"
                }, status=400)
            
            # Check local availability for each version
            for version in versions:
                # Find the primary model file (type="Model" and primary=true) in the files list
                model_file = next((file for file in version.get('files', []) 
                                  if file.get('type') == 'Model' and file.get('primary') == True), None)
                
                # If no primary file found, try to find any model file
                if not model_file:
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
            logger.error(f"Error fetching checkpoint model versions: {e}")
            return web.Response(status=500, text=str(e))

    async def find_duplicate_checkpoints(self, request: web.Request) -> web.Response:
        """Find checkpoints with duplicate SHA256 hashes"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_checkpoint_scanner()
                
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
                # Find matching models for each path
                for path in paths:
                    model = next((m for m in cache.raw_data if m['file_path'] == path), None)
                    if model:
                        group["models"].append(self._format_checkpoint_response(model))
                
                # Add the primary model too
                primary_path = self.scanner._hash_index.get_path(sha256)
                if primary_path and primary_path not in paths:
                    primary_model = next((m for m in cache.raw_data if m['file_path'] == primary_path), None)
                    if primary_model:
                        group["models"].insert(0, self._format_checkpoint_response(primary_model))
                
                if len(group["models"]) > 1:  # Only include if we found multiple models
                    result.append(group)
                
            return web.json_response({
                "success": True,
                "duplicates": result,
                "count": len(result)
            })
        except Exception as e:
            logger.error(f"Error finding duplicate checkpoints: {e}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def find_filename_conflicts(self, request: web.Request) -> web.Response:
        """Find checkpoints with conflicting filenames"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_checkpoint_scanner()
                
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
                        group["models"].append(self._format_checkpoint_response(model))
                
                # Find the model from the main index too
                hash_val = self.scanner._hash_index.get_hash_by_filename(filename)
                if hash_val:
                    main_path = self.scanner._hash_index.get_path(hash_val)
                    if main_path and main_path not in paths:
                        main_model = next((m for m in cache.raw_data if m['file_path'] == main_path), None)
                        if main_model:
                            group["models"].insert(0, self._format_checkpoint_response(main_model))
                
                if group["models"]:
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

    async def bulk_delete_checkpoints(self, request: web.Request) -> web.Response:
        """Handle bulk deletion of checkpoint models"""
        try:
            if self.scanner is None:
                self.scanner = await ServiceRegistry.get_checkpoint_scanner()
            
            return await ModelRouteUtils.handle_bulk_delete_models(request, self.scanner)
                
        except Exception as e:
            logger.error(f"Error in bulk delete checkpoints: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def relink_civitai(self, request: web.Request) -> web.Response:
        """Handle CivitAI metadata re-linking request by model version ID for checkpoints"""
        return await ModelRouteUtils.handle_relink_civitai(request, self.scanner)

    async def verify_duplicates(self, request: web.Request) -> web.Response:
        """Handle verification of duplicate checkpoint hashes"""
        return await ModelRouteUtils.handle_verify_duplicates(request, self.scanner)

    async def rename_checkpoint(self, request: web.Request) -> web.Response:
        """Handle renaming a checkpoint file and its associated files"""
        return await ModelRouteUtils.handle_rename_model(request, self.scanner)

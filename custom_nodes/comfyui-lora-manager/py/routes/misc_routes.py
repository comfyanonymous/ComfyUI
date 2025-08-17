import json
import logging
import os
import sys
import threading
import asyncio
from server import PromptServer # type: ignore
from aiohttp import web
from ..services.settings_manager import settings
from ..utils.usage_stats import UsageStats
from ..utils.lora_metadata import extract_trained_words
from ..config import config
from ..utils.constants import SUPPORTED_MEDIA_EXTENSIONS, NODE_TYPES, DEFAULT_NODE_COLOR
from ..services.service_registry import ServiceRegistry
import re

logger = logging.getLogger(__name__)

standalone_mode = 'nodes' not in sys.modules

# Node registry for tracking active workflow nodes
class NodeRegistry:
    """Thread-safe registry for tracking Lora nodes in active workflows"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._nodes = {}  # node_id -> node_info
        self._registry_updated = threading.Event()
    
    def register_nodes(self, nodes):
        """Register multiple nodes at once, replacing existing registry"""
        with self._lock:
            # Clear existing registry
            self._nodes.clear()
            
            # Register all new nodes
            for node in nodes:
                node_id = node['node_id']
                node_type = node.get('type', '')
                
                # Convert node type name to integer
                type_id = NODE_TYPES.get(node_type, 0)  # 0 for unknown types
                
                # Handle null bgcolor with default color
                bgcolor = node.get('bgcolor')
                if bgcolor is None:
                    bgcolor = DEFAULT_NODE_COLOR
                
                self._nodes[node_id] = {
                    'id': node_id,
                    'bgcolor': bgcolor,
                    'title': node.get('title'),
                    'type': type_id,
                    'type_name': node_type
                }
            
            logger.debug(f"Registered {len(nodes)} nodes in registry")
            
            # Signal that registry has been updated
            self._registry_updated.set()
    
    def get_registry(self):
        """Get current registry information"""
        with self._lock:
            return {
                'nodes': dict(self._nodes),  # Return a copy
                'node_count': len(self._nodes)
            }
    
    def clear_registry(self):
        """Clear the entire registry"""
        with self._lock:
            self._nodes.clear()
            logger.info("Node registry cleared")
    
    def wait_for_update(self, timeout=1.0):
        """Wait for registry update with timeout"""
        self._registry_updated.clear()
        return self._registry_updated.wait(timeout)

# Global registry instance
node_registry = NodeRegistry()

class MiscRoutes:
    """Miscellaneous routes for various utility functions"""
    
    @staticmethod
    def setup_routes(app):
        """Register miscellaneous routes"""
        app.router.add_post('/api/settings', MiscRoutes.update_settings)
        
        # Add new route for clearing cache
        app.router.add_post('/api/clear-cache', MiscRoutes.clear_cache)

        app.router.add_get('/api/health-check', lambda request: web.json_response({'status': 'ok'}))

        # Usage stats routes
        app.router.add_post('/api/update-usage-stats', MiscRoutes.update_usage_stats)
        app.router.add_get('/api/get-usage-stats', MiscRoutes.get_usage_stats)
        
        # Lora code update endpoint
        app.router.add_post('/api/update-lora-code', MiscRoutes.update_lora_code)

        # Add new route for getting trained words
        app.router.add_get('/api/trained-words', MiscRoutes.get_trained_words)
        
        # Add new route for getting model example files
        app.router.add_get('/api/model-example-files', MiscRoutes.get_model_example_files)
        
        # Node registry endpoints
        app.router.add_post('/api/register-nodes', MiscRoutes.register_nodes)
        app.router.add_get('/api/get-registry', MiscRoutes.get_registry)
        
        # Add new route for checking if a model exists in the library
        app.router.add_get('/api/check-model-exists', MiscRoutes.check_model_exists)

    @staticmethod
    async def clear_cache(request):
        """Clear all cache files from the cache folder"""
        try:
            # Get the cache folder path (relative to project directory)
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_folder = os.path.join(project_dir, 'cache')
            
            # Check if cache folder exists
            if not os.path.exists(cache_folder):
                logger.info("Cache folder does not exist, nothing to clear")
                return web.json_response({'success': True, 'message': 'No cache folder found'})
            
            # Get list of cache files before deleting for reporting
            cache_files = [f for f in os.listdir(cache_folder) if os.path.isfile(os.path.join(cache_folder, f))]
            deleted_files = []
            
            # Delete each .msgpack file in the cache folder
            for filename in cache_files:
                if filename.endswith('.msgpack'):
                    file_path = os.path.join(cache_folder, filename)
                    try:
                        os.remove(file_path)
                        deleted_files.append(filename)
                        logger.info(f"Deleted cache file: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to delete {filename}: {e}")
                        return web.json_response({
                            'success': False,
                            'error': f"Failed to delete {filename}: {str(e)}"
                        }, status=500)
            
            return web.json_response({
                'success': True,
                'message': f"Successfully cleared {len(deleted_files)} cache files",
                'deleted_files': deleted_files
            })
            
        except Exception as e:
            logger.error(f"Error clearing cache files: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    @staticmethod
    async def update_settings(request):
        """Update application settings"""
        try:
            data = await request.json()
            
            # Validate and update settings
            for key, value in data.items():
                if value == settings.get(key):
                    # No change, skip
                    continue
                # Special handling for example_images_path - verify path exists
                if key == 'example_images_path' and value:
                    if not os.path.exists(value):
                        return web.json_response({
                            'success': False,
                            'error': f"Path does not exist: {value}"
                        })
                    
                    # Path changed - server restart required for new path to take effect
                    old_path = settings.get('example_images_path')
                    if old_path != value:
                        logger.info(f"Example images path changed to {value} - server restart required")
                
                # Special handling for base_model_path_mappings - parse JSON string
                if (key == 'base_model_path_mappings' or key == 'download_path_templates') and value:
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        return web.json_response({
                            'success': False,
                            'error': f"Invalid JSON format for base_model_path_mappings: {value}"
                        })
                
                # Save to settings
                settings.set(key, value)
            
            return web.json_response({'success': True})
        except Exception as e:
            logger.error(f"Error updating settings: {e}", exc_info=True)
            return web.Response(status=500, text=str(e))
    
    @staticmethod
    async def update_usage_stats(request):
        """
        Update usage statistics based on a prompt_id
        
        Expects a JSON body with:
        {
            "prompt_id": "string"
        }
        """
        try:
            # Parse the request body
            data = await request.json()
            prompt_id = data.get('prompt_id')
            
            if not prompt_id:
                return web.json_response({
                    'success': False,
                    'error': 'Missing prompt_id'
                }, status=400)
            
            # Call the UsageStats to process this prompt_id synchronously
            usage_stats = UsageStats()
            await usage_stats.process_execution(prompt_id)
            
            return web.json_response({
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Failed to update usage stats: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    @staticmethod
    async def get_usage_stats(request):
        """Get current usage statistics"""
        try:
            usage_stats = UsageStats()
            stats = await usage_stats.get_stats()
            
            # Add version information to help clients handle format changes
            stats_response = {
                'success': True,
                'data': stats,
                'format_version': 2  # Indicate this is the new format with history
            }
            
            return web.json_response(stats_response)
            
        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    @staticmethod
    async def update_lora_code(request):
        """
        Update Lora code in ComfyUI nodes
        
        Expects a JSON body with:
        {
            "node_ids": [123, 456], # Optional - List of node IDs to update (for browser mode)
            "lora_code": "<lora:modelname:1.0>", # The Lora code to send
            "mode": "append" # or "replace" - whether to append or replace existing code
        }
        """
        try:
            # Parse the request body
            data = await request.json()
            node_ids = data.get('node_ids')
            lora_code = data.get('lora_code', '')
            mode = data.get('mode', 'append')
            
            if not lora_code:
                return web.json_response({
                    'success': False,
                    'error': 'Missing lora_code parameter'
                }, status=400)
            
            results = []
            
            # Desktop mode: no specific node_ids provided
            if node_ids is None:
                try:
                    # Send broadcast message with id=-1 to all Lora Loader nodes
                    PromptServer.instance.send_sync("lora_code_update", {
                        "id": -1,
                        "lora_code": lora_code,
                        "mode": mode
                    })
                    results.append({
                        'node_id': 'broadcast',
                        'success': True
                    })
                except Exception as e:
                    logger.error(f"Error broadcasting lora code: {e}")
                    results.append({
                        'node_id': 'broadcast',
                        'success': False,
                        'error': str(e)
                    })
            else:
                # Browser mode: send to specific nodes
                for node_id in node_ids:
                    try:
                        # Send the message to the frontend
                        PromptServer.instance.send_sync("lora_code_update", {
                            "id": node_id,
                            "lora_code": lora_code,
                            "mode": mode
                        })
                        results.append({
                            'node_id': node_id,
                            'success': True
                        })
                    except Exception as e:
                        logger.error(f"Error sending lora code to node {node_id}: {e}")
                        results.append({
                            'node_id': node_id,
                            'success': False,
                            'error': str(e)
                        })
            
            return web.json_response({
                'success': True,
                'results': results
            })
            
        except Exception as e:
            logger.error(f"Failed to update lora code: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    @staticmethod
    async def get_trained_words(request):
        """
        Get trained words from a safetensors file, sorted by frequency
        
        Expects a query parameter:
        file_path: Path to the safetensors file
        """
        try:
            # Get file path from query parameters
            file_path = request.query.get('file_path')
            
            if not file_path:
                return web.json_response({
                    'success': False,
                    'error': 'Missing file_path parameter'
                }, status=400)
            
            # Check if file exists and is a safetensors file
            if not os.path.exists(file_path):
                return web.json_response({
                    'success': False,
                    'error': f"File not found: {file_path}"
                }, status=404)
                
            if not file_path.lower().endswith('.safetensors'):
                return web.json_response({
                    'success': False,
                    'error': 'File is not a safetensors file'
                }, status=400)
            
            # Extract trained words and class_tokens
            trained_words, class_tokens = await extract_trained_words(file_path)
            
            # Return result with both trained words and class tokens
            return web.json_response({
                'success': True,
                'trained_words': trained_words,
                'class_tokens': class_tokens
            })
            
        except Exception as e:
            logger.error(f"Failed to get trained words: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    @staticmethod
    async def get_model_example_files(request):
        """
        Get list of example image files for a specific model based on file path
        
        Expects:
        - file_path in query parameters
        
        Returns:
        - List of image files with their paths as static URLs
        """
        try:
            # Get the model file path from query parameters
            file_path = request.query.get('file_path')
            
            if not file_path:
                return web.json_response({
                    'success': False,
                    'error': 'Missing file_path parameter'
                }, status=400)
            
            # Extract directory and base filename
            model_dir = os.path.dirname(file_path)
            model_filename = os.path.basename(file_path)
            model_name = os.path.splitext(model_filename)[0]
            
            # Check if the directory exists
            if not os.path.exists(model_dir):
                return web.json_response({
                    'success': False, 
                    'error': 'Model directory not found',
                    'files': []
                }, status=404)
            
            # Look for files matching the pattern modelname.example.<index>.<ext>
            files = []
            pattern = f"{model_name}.example."
            
            for file in os.listdir(model_dir):
                file_lower = file.lower()
                if file_lower.startswith(pattern.lower()):
                    file_full_path = os.path.join(model_dir, file)
                    if os.path.isfile(file_full_path):
                        # Check if the file is a supported media file
                        file_ext = os.path.splitext(file)[1].lower()
                        if (file_ext in SUPPORTED_MEDIA_EXTENSIONS['images'] or 
                            file_ext in SUPPORTED_MEDIA_EXTENSIONS['videos']):
                            
                            # Extract the index from the filename
                            try:
                                # Extract the part after '.example.' and before file extension
                                index_part = file[len(pattern):].split('.')[0]
                                # Try to parse it as an integer
                                index = int(index_part)
                            except (ValueError, IndexError):
                                # If we can't parse the index, use infinity to sort at the end
                                index = float('inf')
                            
                            # Convert file path to static URL
                            static_url = config.get_preview_static_url(file_full_path)
                            
                            files.append({
                                'name': file,
                                'path': static_url,
                                'extension': file_ext,
                                'is_video': file_ext in SUPPORTED_MEDIA_EXTENSIONS['videos'],
                                'index': index
                            })
            
            # Sort files by their index for consistent ordering
            files.sort(key=lambda x: x['index'])
            # Remove the index field as it's only used for sorting
            for file in files:
                file.pop('index', None)
            
            return web.json_response({
                'success': True,
                'files': files
            })
            
        except Exception as e:
            logger.error(f"Failed to get model example files: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    @staticmethod
    async def register_nodes(request):
        """
        Register multiple Lora nodes at once
        
        Expects a JSON body with:
        {
            "nodes": [
                {
                    "node_id": 123,
                    "bgcolor": "#535",
                    "title": "Lora Loader (LoraManager)"
                },
                ...
            ]
        }
        """
        try:
            data = await request.json()
            
            # Validate required fields
            nodes = data.get('nodes', [])
            
            if not isinstance(nodes, list):
                return web.json_response({
                    'success': False,
                    'error': 'nodes must be a list'
                }, status=400)
            
            # Validate each node
            for i, node in enumerate(nodes):
                if not isinstance(node, dict):
                    return web.json_response({
                        'success': False,
                        'error': f'Node {i} must be an object'
                    }, status=400)
                
                node_id = node.get('node_id')
                if node_id is None:
                    return web.json_response({
                        'success': False,
                        'error': f'Node {i} missing node_id parameter'
                    }, status=400)
                
                # Validate node_id is an integer
                try:
                    node['node_id'] = int(node_id)
                except (ValueError, TypeError):
                    return web.json_response({
                        'success': False,
                        'error': f'Node {i} node_id must be an integer'
                    }, status=400)
            
            # Register all nodes
            node_registry.register_nodes(nodes)
            
            return web.json_response({
                'success': True,
                'message': f'{len(nodes)} nodes registered successfully'
            })
            
        except Exception as e:
            logger.error(f"Failed to register nodes: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    @staticmethod
    async def get_registry(request):
        """Get current node registry information by refreshing from frontend"""
        try:
            # Check if running in standalone mode
            if standalone_mode:
                logger.warning("Registry refresh not available in standalone mode")
                return web.json_response({
                    'success': False,
                    'error': 'Standalone Mode Active',
                    'message': 'Cannot interact with ComfyUI in standalone mode.'
                }, status=503)
            
            # Send message to frontend to refresh registry
            try:
                PromptServer.instance.send_sync("lora_registry_refresh", {})
                logger.debug("Sent registry refresh request to frontend")
            except Exception as e:
                logger.error(f"Failed to send registry refresh message: {e}")
                return web.json_response({
                    'success': False,
                    'error': 'Communication Error',
                    'message': f'Failed to communicate with ComfyUI frontend: {str(e)}'
                }, status=500)
            
            # Wait for registry update with timeout
            def wait_for_registry():
                return node_registry.wait_for_update(timeout=1.0)
            
            # Run the wait in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            registry_updated = await loop.run_in_executor(None, wait_for_registry)
            
            if not registry_updated:
                logger.warning("Registry refresh timeout after 1 second")
                return web.json_response({
                    'success': False,
                    'error': 'Timeout Error',
                    'message': 'Registry refresh timeout - ComfyUI frontend may not be responsive'
                }, status=408)
            
            # Get updated registry
            registry_info = node_registry.get_registry()
            
            return web.json_response({
                'success': True,
                'data': registry_info
            })
            
        except Exception as e:
            logger.error(f"Failed to get registry: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': 'Internal Error',
                'message': str(e)
            }, status=500)

    @staticmethod
    async def check_model_exists(request):
        """
        Check if a model with specified modelId and optionally modelVersionId exists in the library
        
        Expects query parameters:
        - modelId: int - Civitai model ID (required)
        - modelVersionId: int - Civitai model version ID (optional)
        
        Returns:
        - If modelVersionId is provided: JSON with a boolean 'exists' field
        - If modelVersionId is not provided: JSON with a list of modelVersionIds that exist in the library
        """
        try:
            # Get the modelId and modelVersionId from query parameters
            model_id_str = request.query.get('modelId')
            model_version_id_str = request.query.get('modelVersionId')
            
            # Validate modelId parameter (required)
            if not model_id_str:
                return web.json_response({
                    'success': False,
                    'error': 'Missing required parameter: modelId'
                }, status=400)
                
            try:
                # Convert modelId to integer
                model_id = int(model_id_str)
            except ValueError:
                return web.json_response({
                    'success': False,
                    'error': 'Parameter modelId must be an integer'
                }, status=400)
            
            # Get all scanners
            lora_scanner = await ServiceRegistry.get_lora_scanner()
            checkpoint_scanner = await ServiceRegistry.get_checkpoint_scanner()
            embedding_scanner = await ServiceRegistry.get_embedding_scanner()
            
            # If modelVersionId is provided, check for specific version
            if model_version_id_str:
                try:
                    model_version_id = int(model_version_id_str)
                except ValueError:
                    return web.json_response({
                        'success': False,
                        'error': 'Parameter modelVersionId must be an integer'
                    }, status=400)
                
                # Check lora scanner first
                exists = False
                model_type = None

                if await lora_scanner.check_model_version_exists(model_version_id):
                    exists = True
                    model_type = 'lora'
                elif checkpoint_scanner and await checkpoint_scanner.check_model_version_exists(model_version_id):
                    exists = True
                    model_type = 'checkpoint'
                elif embedding_scanner and await embedding_scanner.check_model_version_exists(model_version_id):
                    exists = True
                    model_type = 'embedding'
                
                return web.json_response({
                    'success': True,
                    'exists': exists,
                    'modelType': model_type if exists else None
                })
            
            # If modelVersionId is not provided, return all version IDs for the model
            else:
                lora_versions = await lora_scanner.get_model_versions_by_id(model_id)
                checkpoint_versions = []
                embedding_versions = []

                # 优先lora，其次checkpoint，最后embedding
                if not lora_versions:
                    checkpoint_versions = await checkpoint_scanner.get_model_versions_by_id(model_id)
                if not lora_versions and not checkpoint_versions:
                    embedding_versions = await embedding_scanner.get_model_versions_by_id(model_id)

                model_type = None
                versions = []

                if lora_versions:
                    model_type = 'lora'
                    versions = lora_versions
                elif checkpoint_versions:
                    model_type = 'checkpoint'
                    versions = checkpoint_versions
                elif embedding_versions:
                    model_type = 'embedding'
                    versions = embedding_versions

                return web.json_response({
                    'success': True,
                    'modelId': model_id,
                    'modelType': model_type,
                    'versions': versions
                })
            
        except Exception as e:
            logger.error(f"Failed to check model existence: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

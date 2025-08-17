import asyncio
import logging
from aiohttp import web
from typing import Dict
from server import PromptServer  # type: ignore

from .base_model_routes import BaseModelRoutes
from ..services.lora_service import LoraService
from ..services.service_registry import ServiceRegistry
from ..utils.routes_common import ModelRouteUtils
from ..utils.utils import get_lora_info

logger = logging.getLogger(__name__)

class LoraRoutes(BaseModelRoutes):
    """LoRA-specific route controller"""
    
    def __init__(self):
        """Initialize LoRA routes with LoRA service"""
        # Service will be initialized later via setup_routes
        self.service = None
        self.civitai_client = None
        self.template_name = "loras.html"
    
    async def initialize_services(self):
        """Initialize services from ServiceRegistry"""
        lora_scanner = await ServiceRegistry.get_lora_scanner()
        self.service = LoraService(lora_scanner)
        self.civitai_client = await ServiceRegistry.get_civitai_client()
        
        # Initialize parent with the service
        super().__init__(self.service)
    
    def setup_routes(self, app: web.Application):
        """Setup LoRA routes"""
        # Schedule service initialization on app startup
        app.on_startup.append(lambda _: self.initialize_services())
        
        # Setup common routes with 'loras' prefix (includes page route)
        super().setup_routes(app, 'loras')
    
    def setup_specific_routes(self, app: web.Application, prefix: str):
        """Setup LoRA-specific routes"""
        # LoRA-specific query routes
        app.router.add_get(f'/api/{prefix}/letter-counts', self.get_letter_counts)
        app.router.add_get(f'/api/{prefix}/get-notes', self.get_lora_notes)
        app.router.add_get(f'/api/{prefix}/get-trigger-words', self.get_lora_trigger_words)
        app.router.add_get(f'/api/{prefix}/preview-url', self.get_lora_preview_url)
        app.router.add_get(f'/api/{prefix}/civitai-url', self.get_lora_civitai_url)
        app.router.add_get(f'/api/{prefix}/model-description', self.get_lora_model_description)
        
        # CivitAI integration with LoRA-specific validation
        app.router.add_get(f'/api/{prefix}/civitai/versions/{{model_id}}', self.get_civitai_versions_lora)
        app.router.add_get(f'/api/{prefix}/civitai/model/version/{{modelVersionId}}', self.get_civitai_model_by_version)
        app.router.add_get(f'/api/{prefix}/civitai/model/hash/{{hash}}', self.get_civitai_model_by_hash)
        
        # ComfyUI integration
        app.router.add_post(f'/api/{prefix}/get_trigger_words', self.get_trigger_words)
    
    def _parse_specific_params(self, request: web.Request) -> Dict:
        """Parse LoRA-specific parameters"""
        params = {}
        
        # LoRA-specific parameters
        if 'first_letter' in request.query:
            params['first_letter'] = request.query.get('first_letter')
        
        # Handle fuzzy search parameter name variation
        if request.query.get('fuzzy') == 'true':
            params['fuzzy_search'] = True
        
        # Handle additional filter parameters for LoRAs
        if 'lora_hash' in request.query:
            if not params.get('hash_filters'):
                params['hash_filters'] = {}
            params['hash_filters']['single_hash'] = request.query['lora_hash'].lower()
        elif 'lora_hashes' in request.query:
            if not params.get('hash_filters'):
                params['hash_filters'] = {}
            params['hash_filters']['multiple_hashes'] = [h.lower() for h in request.query['lora_hashes'].split(',')]
        
        return params
    
    # LoRA-specific route handlers
    async def get_letter_counts(self, request: web.Request) -> web.Response:
        """Get count of LoRAs for each letter of the alphabet"""
        try:
            letter_counts = await self.service.get_letter_counts()
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
            lora_name = request.query.get('name')
            if not lora_name:
                return web.Response(text='Lora file name is required', status=400)
            
            notes = await self.service.get_lora_notes(lora_name)
            if notes is not None:
                return web.json_response({
                    'success': True,
                    'notes': notes
                })
            else:
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
            lora_name = request.query.get('name')
            if not lora_name:
                return web.Response(text='Lora file name is required', status=400)
            
            trigger_words = await self.service.get_lora_trigger_words(lora_name)
            return web.json_response({
                'success': True,
                'trigger_words': trigger_words
            })
            
        except Exception as e:
            logger.error(f"Error getting lora trigger words: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def get_lora_preview_url(self, request: web.Request) -> web.Response:
        """Get the static preview URL for a LoRA file"""
        try:
            lora_name = request.query.get('name')
            if not lora_name:
                return web.Response(text='Lora file name is required', status=400)
            
            preview_url = await self.service.get_lora_preview_url(lora_name)
            if preview_url:
                return web.json_response({
                    'success': True,
                    'preview_url': preview_url
                })
            else:
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
            lora_name = request.query.get('name')
            if not lora_name:
                return web.Response(text='Lora file name is required', status=400)
            
            result = await self.service.get_lora_civitai_url(lora_name)
            if result['civitai_url']:
                return web.json_response({
                    'success': True,
                    **result
                })
            else:
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
    
    # CivitAI integration methods
    async def get_civitai_versions_lora(self, request: web.Request) -> web.Response:
        """Get available versions for a Civitai LoRA model with local availability info"""
        try:
            model_id = request.match_info['model_id']
            response = await self.civitai_client.get_model_versions(model_id)
            if not response or not response.get('modelVersions'):
                return web.Response(status=404, text="Model not found")
            
            versions = response.get('modelVersions', [])
            model_type = response.get('type', '')
            
            # Check model type - should be LORA, LoCon, or DORA
            from ..utils.constants import VALID_LORA_TYPES
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
                        version['existsLocally'] = self.service.has_hash(sha256)
                        if version['existsLocally']:
                            version['localPath'] = self.service.get_path_by_hash(sha256)
                        
                        # Also set the model file size at the version level for easier access
                        version['modelSizeKB'] = model_file.get('sizeKB')
                else:
                    # No model file found in this version
                    version['existsLocally'] = False
                    
            return web.json_response(versions)
        except Exception as e:
            logger.error(f"Error fetching LoRA model versions: {e}")
            return web.Response(status=500, text=str(e))
    
    async def get_civitai_model_by_version(self, request: web.Request) -> web.Response:
        """Get CivitAI model details by model version ID"""
        try:
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
            hash = request.match_info.get('hash')
            model = await self.civitai_client.get_model_by_hash(hash)
            return web.json_response(model)
        except Exception as e:
            logger.error(f"Error fetching model details by hash: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def get_lora_model_description(self, request: web.Request) -> web.Response:
        """Get model description for a Lora model"""
        try:
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
                import os
                from ..utils.metadata_manager import MetadataManager
                metadata_path = os.path.splitext(file_path)[0] + '.metadata.json'
                metadata = await ModelRouteUtils.load_local_metadata(metadata_path)
                description = metadata.get('modelDescription')
                tags = metadata.get('tags', [])
                creator = metadata.get('creator', {})
            
            # If description is not in metadata, fetch from CivitAI
            if not description:
                logger.info(f"Fetching model metadata for model ID: {model_id}")
                model_metadata, _ = await self.civitai_client.get_model_metadata(model_id)
                
                if model_metadata:
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

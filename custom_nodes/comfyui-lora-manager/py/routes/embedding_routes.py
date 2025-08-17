import logging
from aiohttp import web

from .base_model_routes import BaseModelRoutes
from ..services.embedding_service import EmbeddingService
from ..services.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)

class EmbeddingRoutes(BaseModelRoutes):
    """Embedding-specific route controller"""
    
    def __init__(self):
        """Initialize Embedding routes with Embedding service"""
        # Service will be initialized later via setup_routes
        self.service = None
        self.civitai_client = None
        self.template_name = "embeddings.html"
    
    async def initialize_services(self):
        """Initialize services from ServiceRegistry"""
        embedding_scanner = await ServiceRegistry.get_embedding_scanner()
        self.service = EmbeddingService(embedding_scanner)
        self.civitai_client = await ServiceRegistry.get_civitai_client()
        
        # Initialize parent with the service
        super().__init__(self.service)
    
    def setup_routes(self, app: web.Application):
        """Setup Embedding routes"""
        # Schedule service initialization on app startup
        app.on_startup.append(lambda _: self.initialize_services())
        
        # Setup common routes with 'embeddings' prefix (includes page route)
        super().setup_routes(app, 'embeddings')
    
    def setup_specific_routes(self, app: web.Application, prefix: str):
        """Setup Embedding-specific routes"""
        # Embedding-specific CivitAI integration
        app.router.add_get(f'/api/{prefix}/civitai/versions/{{model_id}}', self.get_civitai_versions_embedding)
        
        # Embedding info by name
        app.router.add_get(f'/api/{prefix}/info/{{name}}', self.get_embedding_info)
    
    async def get_embedding_info(self, request: web.Request) -> web.Response:
        """Get detailed information for a specific embedding by name"""
        try:
            name = request.match_info.get('name', '')
            embedding_info = await self.service.get_model_info_by_name(name)
            
            if embedding_info:
                return web.json_response(embedding_info)
            else:
                return web.json_response({"error": "Embedding not found"}, status=404)
                
        except Exception as e:
            logger.error(f"Error in get_embedding_info: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_civitai_versions_embedding(self, request: web.Request) -> web.Response:
        """Get available versions for a Civitai embedding model with local availability info"""
        try:
            model_id = request.match_info['model_id']
            response = await self.civitai_client.get_model_versions(model_id)
            if not response or not response.get('modelVersions'):
                return web.Response(status=404, text="Model not found")
            
            versions = response.get('modelVersions', [])
            model_type = response.get('type', '')
            
            # Check model type - should be TextualInversion (Embedding)
            if model_type.lower() not in ['textualinversion', 'embedding']:
                return web.json_response({
                    'error': f"Model type mismatch. Expected TextualInversion/Embedding, got {model_type}"
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
            logger.error(f"Error fetching embedding model versions: {e}")
            return web.Response(status=500, text=str(e))

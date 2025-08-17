import asyncio
import sys
import os
import logging
from pathlib import Path
from server import PromptServer # type: ignore

from .config import config
from .services.model_service_factory import ModelServiceFactory, register_default_model_types
from .routes.recipe_routes import RecipeRoutes
from .routes.stats_routes import StatsRoutes
from .routes.update_routes import UpdateRoutes
from .routes.misc_routes import MiscRoutes
from .routes.example_images_routes import ExampleImagesRoutes
from .services.service_registry import ServiceRegistry
from .services.settings_manager import settings
from .utils.example_images_migration import ExampleImagesMigration
from .services.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

# Check if we're in standalone mode
STANDALONE_MODE = 'nodes' not in sys.modules

class LoraManager:
    """Main entry point for LoRA Manager plugin"""
    
    @classmethod
    def add_routes(cls):
        """Initialize and register all routes using the new refactored architecture"""
        app = PromptServer.instance.app

        # Configure aiohttp access logger to be less verbose
        logging.getLogger('aiohttp.access').setLevel(logging.WARNING)

        # Add specific suppression for connection reset errors
        class ConnectionResetFilter(logging.Filter):
            def filter(self, record):
                # Filter out connection reset errors that are not critical
                if "ConnectionResetError" in str(record.getMessage()):
                    return False
                if "_call_connection_lost" in str(record.getMessage()):
                    return False
                if "WinError 10054" in str(record.getMessage()):
                    return False
                return True

        # Apply the filter to asyncio logger
        asyncio_logger = logging.getLogger("asyncio")
        asyncio_logger.addFilter(ConnectionResetFilter())

        added_targets = set()  # Track already added target paths
        
        # Add static route for example images if the path exists in settings
        example_images_path = settings.get('example_images_path')
        logger.info(f"Example images path: {example_images_path}")
        if example_images_path and os.path.exists(example_images_path):
            app.router.add_static('/example_images_static', example_images_path)
            logger.info(f"Added static route for example images: /example_images_static -> {example_images_path}")
        
        # Add static routes for each lora root
        for idx, root in enumerate(config.loras_roots, start=1):
            preview_path = f'/loras_static/root{idx}/preview'
            
            real_root = root
            if root in config._path_mappings.values():
                for target, link in config._path_mappings.items():
                    if link == root:
                        real_root = target
                        break
            # Add static route for original path
            app.router.add_static(preview_path, real_root)
            logger.info(f"Added static route {preview_path} -> {real_root}")
            
            # Record route mapping
            config.add_route_mapping(real_root, preview_path)
            added_targets.add(real_root)
        
        # Add static routes for each checkpoint root
        for idx, root in enumerate(config.base_models_roots, start=1):
            preview_path = f'/checkpoints_static/root{idx}/preview'
            
            real_root = root
            if root in config._path_mappings.values():
                for target, link in config._path_mappings.items():
                    if link == root:
                        real_root = target
                        break
            # Add static route for original path
            app.router.add_static(preview_path, real_root)
            logger.info(f"Added static route {preview_path} -> {real_root}")
            
            # Record route mapping
            config.add_route_mapping(real_root, preview_path)
            added_targets.add(real_root)
        
        # Add static routes for each embedding root
        for idx, root in enumerate(config.embeddings_roots, start=1):
            preview_path = f'/embeddings_static/root{idx}/preview'
            
            real_root = root
            if root in config._path_mappings.values():
                for target, link in config._path_mappings.items():
                    if link == root:
                        real_root = target
                        break
            # Add static route for original path
            app.router.add_static(preview_path, real_root)
            logger.info(f"Added static route {preview_path} -> {real_root}")
            
            # Record route mapping
            config.add_route_mapping(real_root, preview_path)
            added_targets.add(real_root)
        
        # Add static routes for symlink target paths
        link_idx = {
            'lora': 1,
            'checkpoint': 1,
            'embedding': 1
        }
        
        for target_path, link_path in config._path_mappings.items():
            if target_path not in added_targets:
                # Determine if this is a checkpoint, lora, or embedding link based on path
                is_checkpoint = any(cp_root in link_path for cp_root in config.base_models_roots)
                is_checkpoint = is_checkpoint or any(cp_root in target_path for cp_root in config.base_models_roots)
                is_embedding = any(emb_root in link_path for emb_root in config.embeddings_roots)
                is_embedding = is_embedding or any(emb_root in target_path for emb_root in config.embeddings_roots)
                
                if is_checkpoint:
                    route_path = f'/checkpoints_static/link_{link_idx["checkpoint"]}/preview'
                    link_idx["checkpoint"] += 1
                elif is_embedding:
                    route_path = f'/embeddings_static/link_{link_idx["embedding"]}/preview'
                    link_idx["embedding"] += 1
                else:
                    route_path = f'/loras_static/link_{link_idx["lora"]}/preview'
                    link_idx["lora"] += 1
                
                try:
                    app.router.add_static(route_path, Path(target_path).resolve(strict=False))
                    logger.info(f"Added static route for link target {route_path} -> {target_path}")
                    config.add_route_mapping(target_path, route_path)
                    added_targets.add(target_path)
                except Exception as e:
                    logger.warning(f"Failed to add static route on initialization for {target_path}: {e}")
                    continue
        
        # Add static route for plugin assets
        app.router.add_static('/loras_static', config.static_path)
        
        # Register default model types with the factory
        register_default_model_types()
        
        # Setup all model routes using the factory
        ModelServiceFactory.setup_all_routes(app)
        
        # Setup non-model-specific routes
        stats_routes = StatsRoutes()
        stats_routes.setup_routes(app)
        RecipeRoutes.setup_routes(app)
        UpdateRoutes.setup_routes(app)  
        MiscRoutes.setup_routes(app)
        ExampleImagesRoutes.setup_routes(app)
        
        # Setup WebSocket routes that are shared across all model types
        app.router.add_get('/ws/fetch-progress', ws_manager.handle_connection)
        app.router.add_get('/ws/download-progress', ws_manager.handle_download_connection)
        app.router.add_get('/ws/init-progress', ws_manager.handle_init_connection)
        
        # Schedule service initialization 
        app.on_startup.append(lambda app: cls._initialize_services())
        
        # Add cleanup
        app.on_shutdown.append(cls._cleanup)
        
        logger.info(f"LoRA Manager: Set up routes for {len(ModelServiceFactory.get_registered_types())} model types: {', '.join(ModelServiceFactory.get_registered_types())}")
    
    @classmethod
    async def _initialize_services(cls):
        """Initialize all services using the ServiceRegistry"""
        try:
            # Initialize CivitaiClient first to ensure it's ready for other services
            await ServiceRegistry.get_civitai_client()

            # Register DownloadManager with ServiceRegistry
            await ServiceRegistry.get_download_manager()
            
            # Initialize WebSocket manager
            await ServiceRegistry.get_websocket_manager()
            
            # Initialize scanners in background
            lora_scanner = await ServiceRegistry.get_lora_scanner()
            checkpoint_scanner = await ServiceRegistry.get_checkpoint_scanner()
            embedding_scanner = await ServiceRegistry.get_embedding_scanner()
            
            # Initialize recipe scanner if needed
            recipe_scanner = await ServiceRegistry.get_recipe_scanner()
            
            # Create low-priority initialization tasks
            asyncio.create_task(lora_scanner.initialize_in_background(), name='lora_cache_init')
            asyncio.create_task(checkpoint_scanner.initialize_in_background(), name='checkpoint_cache_init')
            asyncio.create_task(embedding_scanner.initialize_in_background(), name='embedding_cache_init')
            asyncio.create_task(recipe_scanner.initialize_in_background(), name='recipe_cache_init')

            await ExampleImagesMigration.check_and_run_migrations()
            
            logger.info("LoRA Manager: All services initialized and background tasks scheduled")
                
        except Exception as e:
            logger.error(f"LoRA Manager: Error initializing services: {e}", exc_info=True)
    
    @classmethod
    async def _cleanup(cls, app):
        """Cleanup resources using ServiceRegistry"""
        try:
            logger.info("LoRA Manager: Cleaning up services")
                
            # Close CivitaiClient gracefully
            civitai_client = await ServiceRegistry.get_service("civitai_client")
            if civitai_client:
                await civitai_client.close()
                logger.info("Closed CivitaiClient connection")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

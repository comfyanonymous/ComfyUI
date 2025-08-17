from typing import Dict, Type, Any
import logging

logger = logging.getLogger(__name__)

class ModelServiceFactory:
    """Factory for managing model services and routes"""
    
    _services: Dict[str, Type] = {}
    _routes: Dict[str, Type] = {}
    _initialized_services: Dict[str, Any] = {}
    _initialized_routes: Dict[str, Any] = {}
    
    @classmethod
    def register_model_type(cls, model_type: str, service_class: Type, route_class: Type):
        """Register a new model type with its service and route classes
        
        Args:
            model_type: The model type identifier (e.g., 'lora', 'checkpoint')
            service_class: The service class for this model type
            route_class: The route class for this model type
        """
        cls._services[model_type] = service_class
        cls._routes[model_type] = route_class
        logger.info(f"Registered model type '{model_type}' with service {service_class.__name__} and routes {route_class.__name__}")
    
    @classmethod
    def get_service_class(cls, model_type: str) -> Type:
        """Get service class for a model type
        
        Args:
            model_type: The model type identifier
            
        Returns:
            The service class for the model type
            
        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in cls._services:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._services[model_type]
    
    @classmethod
    def get_route_class(cls, model_type: str) -> Type:
        """Get route class for a model type
        
        Args:
            model_type: The model type identifier
            
        Returns:
            The route class for the model type
            
        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in cls._routes:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._routes[model_type]
    
    @classmethod
    def get_route_instance(cls, model_type: str):
        """Get or create route instance for a model type
        
        Args:
            model_type: The model type identifier
            
        Returns:
            The route instance for the model type
        """
        if model_type not in cls._initialized_routes:
            route_class = cls.get_route_class(model_type)
            cls._initialized_routes[model_type] = route_class()
        return cls._initialized_routes[model_type]
    
    @classmethod
    def setup_all_routes(cls, app):
        """Setup routes for all registered model types
        
        Args:
            app: The aiohttp application instance
        """
        logger.info(f"Setting up routes for {len(cls._services)} registered model types")
        
        for model_type in cls._services.keys():
            try:
                routes_instance = cls.get_route_instance(model_type)
                routes_instance.setup_routes(app)
                logger.info(f"Successfully set up routes for {model_type}")
            except Exception as e:
                logger.error(f"Failed to setup routes for {model_type}: {e}", exc_info=True)
    
    @classmethod
    def get_registered_types(cls) -> list:
        """Get list of all registered model types
        
        Returns:
            List of registered model type identifiers
        """
        return list(cls._services.keys())
    
    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """Check if a model type is registered
        
        Args:
            model_type: The model type identifier
            
        Returns:
            True if the model type is registered, False otherwise
        """
        return model_type in cls._services
    
    @classmethod
    def clear_registrations(cls):
        """Clear all registrations - mainly for testing purposes"""
        cls._services.clear()
        cls._routes.clear()
        cls._initialized_services.clear()
        cls._initialized_routes.clear()
        logger.info("Cleared all model type registrations")


def register_default_model_types():
    """Register the default model types (LoRA, Checkpoint, and Embedding)"""
    from ..services.lora_service import LoraService
    from ..services.checkpoint_service import CheckpointService
    from ..services.embedding_service import EmbeddingService
    from ..routes.lora_routes import LoraRoutes
    from ..routes.checkpoint_routes import CheckpointRoutes
    from ..routes.embedding_routes import EmbeddingRoutes
    
    # Register LoRA model type
    ModelServiceFactory.register_model_type('lora', LoraService, LoraRoutes)
    
    # Register Checkpoint model type
    ModelServiceFactory.register_model_type('checkpoint', CheckpointService, CheckpointRoutes)
    
    # Register Embedding model type
    ModelServiceFactory.register_model_type('embedding', EmbeddingService, EmbeddingRoutes)
    
    logger.info("Registered default model types: lora, checkpoint, embedding")
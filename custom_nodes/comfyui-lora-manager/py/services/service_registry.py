import asyncio
import logging
from typing import Optional, Dict, Any, TypeVar, Type

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Define a type variable for service types

class ServiceRegistry:
    """Central registry for managing singleton services"""
    
    _services: Dict[str, Any] = {}
    _locks: Dict[str, asyncio.Lock] = {}
    
    @classmethod
    async def register_service(cls, name: str, service: Any) -> None:
        """Register a service instance with the registry
        
        Args:
            name: Service name identifier
            service: Service instance to register
        """
        cls._services[name] = service
        logger.debug(f"Registered service: {name}")
    
    @classmethod
    async def get_service(cls, name: str) -> Optional[Any]:
        """Get a service instance by name
        
        Args:
            name: Service name identifier
            
        Returns:
            Service instance or None if not found
        """
        return cls._services.get(name)
    
    @classmethod
    def get_service_sync(cls, name: str) -> Optional[Any]:
        """Synchronously get a service instance by name

        Args:
            name: Service name identifier

        Returns:
            Service instance or None if not found
        """
        return cls._services.get(name)
    
    @classmethod
    def _get_lock(cls, name: str) -> asyncio.Lock:
        """Get or create a lock for a service
        
        Args:
            name: Service name identifier
            
        Returns:
            AsyncIO lock for the service
        """
        if name not in cls._locks:
            cls._locks[name] = asyncio.Lock()
        return cls._locks[name]
    
    @classmethod
    async def get_lora_scanner(cls):
        """Get or create LoRA scanner instance"""
        service_name = "lora_scanner"
        
        if service_name in cls._services:
            return cls._services[service_name]
        
        async with cls._get_lock(service_name):
            # Double-check after acquiring lock
            if service_name in cls._services:
                return cls._services[service_name]
            
            # Import here to avoid circular imports
            from .lora_scanner import LoraScanner
            
            scanner = await LoraScanner.get_instance()
            cls._services[service_name] = scanner
            logger.debug(f"Created and registered {service_name}")
            return scanner
    
    @classmethod
    async def get_checkpoint_scanner(cls):
        """Get or create Checkpoint scanner instance"""
        service_name = "checkpoint_scanner"
        
        if service_name in cls._services:
            return cls._services[service_name]
        
        async with cls._get_lock(service_name):
            # Double-check after acquiring lock
            if service_name in cls._services:
                return cls._services[service_name]
            
            # Import here to avoid circular imports
            from .checkpoint_scanner import CheckpointScanner
            
            scanner = await CheckpointScanner.get_instance()
            cls._services[service_name] = scanner
            logger.debug(f"Created and registered {service_name}")
            return scanner
    
    @classmethod
    async def get_recipe_scanner(cls):
        """Get or create Recipe scanner instance"""
        service_name = "recipe_scanner"
        
        if service_name in cls._services:
            return cls._services[service_name]
        
        async with cls._get_lock(service_name):
            # Double-check after acquiring lock
            if service_name in cls._services:
                return cls._services[service_name]
            
            # Import here to avoid circular imports
            from .recipe_scanner import RecipeScanner
            
            scanner = await RecipeScanner.get_instance()
            cls._services[service_name] = scanner
            logger.debug(f"Created and registered {service_name}")
            return scanner
    
    @classmethod
    async def get_civitai_client(cls):
        """Get or create CivitAI client instance"""
        service_name = "civitai_client"
        
        if service_name in cls._services:
            return cls._services[service_name]
        
        async with cls._get_lock(service_name):
            # Double-check after acquiring lock
            if service_name in cls._services:
                return cls._services[service_name]
            
            # Import here to avoid circular imports
            from .civitai_client import CivitaiClient
            
            client = await CivitaiClient.get_instance()
            cls._services[service_name] = client
            logger.debug(f"Created and registered {service_name}")
            return client
    
    @classmethod
    async def get_download_manager(cls):
        """Get or create Download manager instance"""
        service_name = "download_manager"
        
        if service_name in cls._services:
            return cls._services[service_name]
        
        async with cls._get_lock(service_name):
            # Double-check after acquiring lock
            if service_name in cls._services:
                return cls._services[service_name]
            
            # Import here to avoid circular imports
            from .download_manager import DownloadManager
            
            manager = DownloadManager()
            cls._services[service_name] = manager
            logger.debug(f"Created and registered {service_name}")
            return manager
    
    @classmethod
    async def get_websocket_manager(cls):
        """Get or create WebSocket manager instance"""
        service_name = "websocket_manager"
        
        if service_name in cls._services:
            return cls._services[service_name]
        
        async with cls._get_lock(service_name):
            # Double-check after acquiring lock
            if service_name in cls._services:
                return cls._services[service_name]
            
            # Import here to avoid circular imports
            from .websocket_manager import ws_manager
            
            cls._services[service_name] = ws_manager
            logger.debug(f"Registered {service_name}")
            return ws_manager
    
    @classmethod
    async def get_embedding_scanner(cls):
        """Get or create Embedding scanner instance"""
        service_name = "embedding_scanner"
        
        if service_name in cls._services:
            return cls._services[service_name]
        
        async with cls._get_lock(service_name):
            # Double-check after acquiring lock
            if service_name in cls._services:
                return cls._services[service_name]
            
            # Import here to avoid circular imports
            from .embedding_scanner import EmbeddingScanner
            
            scanner = await EmbeddingScanner.get_instance()
            cls._services[service_name] = scanner
            logger.debug(f"Created and registered {service_name}")
            return scanner
    
    @classmethod
    def clear_services(cls):
        """Clear all registered services - mainly for testing"""
        cls._services.clear()
        cls._locks.clear()
        logger.info("Cleared all registered services")
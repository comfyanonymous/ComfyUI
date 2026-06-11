"""API Key Authentication middleware for ComfyUI server"""

from aiohttp import web
from typing import Callable, Awaitable, Optional, Set
import logging
import os


class APIKeyAuth:
    """API Key Authentication handler"""
    
    def __init__(self, api_key: Optional[str] = None, exempt_paths: Optional[Set[str]] = None):
        """
        Initialize API Key Authentication
        
        Args:
            api_key: The API key to validate against. If None, authentication is disabled.
            exempt_paths: Set of paths that don't require authentication (e.g., health check)
        """
        self.api_key = api_key
        self.enabled = api_key is not None and len(api_key) > 0
        self.exempt_paths = exempt_paths or {"/health"}
        
        # Static file extensions that don't require authentication
        self.static_extensions = {
            '.html', '.js', '.css', '.json', '.map', '.png', '.jpg', '.jpeg', 
            '.gif', '.svg', '.ico', '.woff', '.woff2', '.ttf', '.eot', '.webp'
        }
        
        # Path prefixes that serve static content
        self.static_path_prefixes = {
            '/extensions/', '/templates/', '/docs/'
        }
        
        if self.enabled:
            logging.info("[Auth] API Key authentication enabled")
        else:
            logging.info("[Auth] API Key authentication disabled")
    
    def is_path_exempt(self, path: str) -> bool:
        """Check if a path is exempt from authentication"""
        # Exact match for specific exempt paths
        if path in self.exempt_paths:
            return True
        
        # Root path for index.html
        if path == "/":
            return True
        
        # Static file extensions
        for ext in self.static_extensions:
            if path.endswith(ext):
                return True
        
        # Static path prefixes (extensions, templates, docs, etc.)
        for prefix in self.static_path_prefixes:
            if path.startswith(prefix):
                return True
        
        return False
    
    def validate_api_key(self, provided_key: Optional[str]) -> bool:
        """Validate the provided API key"""
        if not self.enabled:
            return True
        
        if not provided_key:
            return False
        
        return provided_key == self.api_key
    
    def extract_api_key(self, request: web.Request) -> Optional[str]:
        """
        Extract API key from request.
        Checks Authorization header (Bearer token) and X-API-Key header.
        """
        # Check Authorization header (Bearer token)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        # Check X-API-Key header
        api_key_header = request.headers.get("X-API-Key", "")
        if api_key_header:
            return api_key_header
        
        # Check query parameter (less secure, but convenient for testing)
        api_key_query = request.query.get("api_key", "")
        if api_key_query:
            return api_key_query
        
        return None


def create_api_key_middleware(api_key: Optional[str] = None, exempt_paths: Optional[Set[str]] = None):
    """
    Create API key authentication middleware
    
    Args:
        api_key: The API key to validate against. If None, authentication is disabled.
        exempt_paths: Set of paths that don't require authentication
    
    Returns:
        Middleware function for aiohttp
    """
    auth = APIKeyAuth(api_key, exempt_paths)
    
    @web.middleware
    async def api_key_middleware(
        request: web.Request, 
        handler: Callable[[web.Request], Awaitable[web.Response]]
    ) -> web.Response:
        """Middleware to validate API key for protected endpoints"""
        
        # Skip authentication if disabled
        if not auth.enabled:
            return await handler(request)
        
        # Check if path is exempt from authentication
        if auth.is_path_exempt(request.path):
            return await handler(request)
        
        # Extract and validate API key
        provided_key = auth.extract_api_key(request)
        
        if not auth.validate_api_key(provided_key):
            logging.warning(f"[Auth] Unauthorized access attempt to {request.path} from {request.remote}")
            return web.json_response(
                {
                    "error": "Unauthorized",
                    "message": "Invalid or missing API key. Provide API key via 'Authorization: Bearer <key>' or 'X-API-Key: <key>' header."
                },
                status=401
            )
        
        # API key is valid, proceed with request
        return await handler(request)
    
    return api_key_middleware

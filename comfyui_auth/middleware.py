from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from jose import JWTError
import time
import logging
from typing import Callable, Awaitable

from .utils import verify_token
from .auth import get_db, get_current_user

logger = logging.getLogger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        excluded_paths: list = None,
        excluded_prefixes: list = None
    ):
        super().__init__(app)
        self.excluded_paths = excluded_paths or ["/api/auth", "/web", "/static", "/favicon.ico"]
        self.excluded_prefixes = excluded_prefixes or ["/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        start_time = time.time()
        
        # Check if path should be excluded from authentication
        path = request.url.path
        
        # Check exact path matches
        if path in self.excluded_paths:
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        
        # Check prefix matches
        for prefix in self.excluded_prefixes:
            if path.startswith(prefix):
                response = await call_next(request)
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(process_time)
                return response
        
        # Check for public API endpoints
        public_endpoints = [
            "/api/auth/register",
            "/api/auth/login",
            "/api/auth/health"
        ]
        if path in public_endpoints:
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authorization header missing or invalid"}
            )
        
        token = auth_header.split(" ")[1]
        
        try:
            # Verify token
            payload = verify_token(token)
            
            # Attach user info to request state
            request.state.user = payload
            
            # Proceed with request
            response = await call_next(request)
            
        except HTTPException as e:
            # Re-raise HTTP exceptions to preserve status code
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
        except JWTError as e:
            logger.error(f"Token verification error: {e}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or expired token"}
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error during authentication"}
            )
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        max_requests: int = 100,
        window_seconds: int = 60
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean up old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                timestamp for timestamp in self.requests[client_ip]
                if now - timestamp < self.window_seconds
            ]
        
        # Check rate limit
        if len(self.requests.get(client_ip, [])) >= self.max_requests:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many requests"}
            )
        
        # Add current request timestamp
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append(now)
        
        response = await call_next(request)
        return response

def require_auth(func: Callable) -> Callable:
    """Decorator to protect routes with authentication"""
    async def wrapper(request: Request, *args, **kwargs):
        # This will be handled by AuthMiddleware
        if not hasattr(request.state, "user"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )
        return await func(request, *args, **kwargs)
    return wrapper

def require_role(required_role: str) -> Callable:
    """Decorator to protect routes with role-based access control"""
    def decorator(func: Callable) -> Callable:
        async def wrapper(request: Request, *args, **kwargs):
            if not hasattr(request.state, "user"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated"
                )
            
            user_role = request.state.user.get("role", "user")
            if user_role != required_role and user_role != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
#!/usr/bin/env python3
"""
Route checking script for ComfyUI authentication module
"""

import sys
import os
from fastapi import FastAPI

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comfyui_auth import create_app

def list_routes(app: FastAPI):
    """List all API routes and their protection status"""
    print("ComfyUI Auth Module - Route Check")
    print("=" * 60)
    
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            routes.append({
                "path": route.path,
                "methods": route.methods,
                "name": route.name,
                "protected": is_protected(route)
            })
    
    # Sort routes by path
    routes.sort(key=lambda x: x["path"])
    
    # Print routes
    for route in routes:
        protected = "✅ Protected" if route["protected"] else "⚠️ Public"
        methods = ", ".join(sorted(route["methods"]))
        print(f"\nPath: {route['path']}")
        print(f"Methods: {methods}")
        print(f"Status: {protected}")
        print(f"Name: {route['name']}")
        print("-" * 40)
    
    # Summary
    protected_count = sum(1 for r in routes if r["protected"])
    public_count = len(routes) - protected_count
    print(f"\nSummary:")
    print(f"Total routes: {len(routes)}")
    print(f"Protected routes: {protected_count}")
    print(f"Public routes: {public_count}")

def is_protected(route) -> bool:
    """Determine if a route is protected by authentication"""
    # Check if route is in auth module
    if "/api/auth" in route.path:
        # Auth endpoints are public except /me, /protected, /admin-only
        if route.path in ["/api/auth/me", "/api/auth/protected", "/api/auth/admin-only"]:
            return True
        if route.path.startswith("/api/auth/logout") or route.path.startswith("/api/auth/refresh"):
            return True
        return False
    
    # Check if route is static or documentation
    if any(prefix in route.path for prefix in ["/web", "/static", "/docs", "/redoc", "/openapi.json"]):
        return False
    
    # Check if route has dependencies
    if hasattr(route, "dependencies") and route.dependencies:
        for dep in route.dependencies:
            if "get_current_user" in str(dep.dependency):
                return True
    
    # Default to protected
    return True

def check_security_config():
    """Check security configuration"""
    print("\n\nSecurity Configuration Check")
    print("=" * 60)
    
    # Check environment variables
    import os
    
    # JWT secret
    jwt_secret = os.getenv("JWT_SECRET", "")
    if jwt_secret == "" or jwt_secret == "your-secret-key-here-change-in-production":
        print("⚠️  WARNING: JWT_SECRET is using default value - change this in production!")
    else:
        print("✅ JWT_SECRET is set")
    
    # Database URL
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        print("⚠️  WARNING: DATABASE_URL not set - using default SQLite")
    else:
        print(f"✅ Database: {db_url.split('://')[0]}")
    
    # Debug mode
    debug = os.getenv("DEBUG", "False").lower() == "true"
    if debug:
        print("⚠️  WARNING: DEBUG mode is enabled - disable in production!")
    else:
        print("✅ DEBUG mode is disabled")
    
    # Allowed origins
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
    if not allowed_origins:
        print("⚠️  WARNING: ALLOWED_ORIGINS not set - using defaults")
    else:
        print(f"✅ Allowed origins: {allowed_origins}")

def main():
    """Main function"""
    app = create_app()
    
    # Load environment variables if .env exists
    if os.path.exists(".env"):
        from dotenv import load_dotenv
        load_dotenv()
    
    print("ComfyUI Authentication Module - Route Checker")
    print("=" * 60)
    
    # List routes
    list_routes(app)
    
    # Check security config
    check_security_config()
    
    print("\n\n✅ Route check completed!")

if __name__ == "__main__":
    main()
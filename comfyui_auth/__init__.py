from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
import os

from .routes import router as auth_router
from .middleware import AuthMiddleware, RateLimitMiddleware
from .auth import get_current_user, get_db
from .models import User

# Configuration
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:8188,http://127.0.0.1:8188').split(',')

def create_app() -> FastAPI:
    app = FastAPI(
        title="ComfyUI Auth",
        description="Authentication module for ComfyUI",
        version="1.0.0",
        debug=DEBUG
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=100,
        window_seconds=60
    )
    
    # Add authentication middleware
    app.add_middleware(
        AuthMiddleware,
        excluded_paths=["/api/auth", "/web", "/static", "/favicon.ico"],
        excluded_prefixes=["/docs", "/redoc", "/openapi.json"]
    )
    
    # Include auth routes
    app.include_router(auth_router)
    
    # Mount static files
    app.mount("/web", StaticFiles(directory="web"), name="web")
    
    # Add main route
    @app.get("/")
    async def root(request: Request, current_user: User = Depends(get_current_user)):
        # Return main page with login status
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ComfyUI</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #eee;
                }
                .user-info {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                button {
                    padding: 8px 16px;
                    background: #667eea;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                button:hover {
                    background: #5568d3;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ComfyUI</h1>
                <div class="user-info">
                    <span>Welcome, {username}!</span>
                    <button onclick="logout()">Logout</button>
                </div>
            </div>
            <h2>Workflows</h2>
            <p>Your ComfyUI workflows will appear here.</p>
            
            <script>
                function logout() {
                    fetch('/api/auth/logout', {
                        method: 'POST',
                        headers: {
                            'Authorization': 'Bearer ' + localStorage.getItem('access_token')
                        }
                    }).then(function() {
                        localStorage.removeItem('access_token');
                        localStorage.removeItem('refresh_token');
                        window.location.href = '/web/login.html';
                    });
                }
            </script>
        </body>
        </html>
        """.format(username=current_user.username)
        return HTMLResponse(content=html_content)
    
    # Add route for login page
    @app.get("/login")
    async def login_page():
        return RedirectResponse(url="/web/login.html")
    
    # Add route for register page
    @app.get("/register")
    async def register_page():
        return RedirectResponse(url="/web/register.html")
    
    # Protected workflow example
    @app.get("/api/workflows")
    async def get_workflows(current_user: User = Depends(get_current_user)):
        return {
            "message": "List of workflows",
            "user": current_user.username,
            "workflows": []
        }
    
    return app

# Initialize database
from .models import engine, Base
Base.metadata.create_all(bind=engine)

# Create app instance
app = create_app()
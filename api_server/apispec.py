from aiohttp_apispec import setup_aiohttp_apispec
from api_server.routes.api_docs import register_api_docs

def register_apispec(app):
    """
    Register Swagger UI and API documentation on the given aiohttp app
    """
    # Register API documentation
    register_api_docs(app)
    
    # Register Swagger UI
    setup_aiohttp_apispec(
        app=app,
        title="ComfyUI API",
        version="1.0.0",
        url="/docs/swagger.json",
        swagger_path="/docs",
        swagger_config={
            "layout": "StandaloneLayout",
            "deepLinking": True,
            "displayRequestDuration": True,
            "defaultModelsExpandDepth": 3,
            "docExpansion": "list",
            "tagsSorter": "alpha",
            "operationsSorter": "alpha"
        }
    ) 
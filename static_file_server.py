from aiohttp import web
import os

def serve_react_app(app: web.Application, build_dir: str, server_url: str, protocol: str, api_key: str = None):
    async def handle_index(request):
        """Serve the index.html with the server URL injected."""
        with open(os.path.join(build_dir, 'index.html'), 'r') as file:
            content = file.read()
        
        content = content.replace('%API_KEY%', api_key)        
        content = content.replace('%SERVER_URL%', server_url)
        content = content.replace('%SERVER_PROTOCOL%', protocol)
        
        return web.Response(text=content, content_type='text/html')

    # Add routes to the existing app
    app.router.add_get('/', handle_index)
    app.router.add_static('/', build_dir, follow_symlinks=True)


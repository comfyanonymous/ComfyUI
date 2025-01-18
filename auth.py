import os
from aiohttp import web

COMFYUI_API_KEY = 'OKZOO_COMFYUI_API_KEY'
CLIENT_API_KEY = 'X-OKZOO-API-KEY'

def validate_api_key(header_api_key):
    api_key = os.getenv(COMFYUI_API_KEY)
    
    if(header_api_key == api_key):
        return True
    return False

def get_api_key_from_client_request(request: web.Request):
    return request.headers.get(CLIENT_API_KEY)

def validate_request(request: web.Request):
    api_key = get_api_key_from_client_request(request)

    if api_key is None:
        raise web.HTTPUnauthorized(text="Unauthorized access: Missing API key")
    if validate_api_key(api_key) == False:
        raise web.HTTPUnauthorized(text="Unauthorized access: Invalid API key")
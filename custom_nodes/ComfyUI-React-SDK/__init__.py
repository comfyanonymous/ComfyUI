from .base import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

import os
import server
from aiohttp import web

# Set your web app name here, this will result access to it via <compfy-url>/<your-app-name>
# Note! you shoud set the same value in app/.env file for react app to build to the same path!.
APP_NAME="root" # <- CHANGE ME

WEBROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

APP_ROOT_URI="/"+APP_NAME
@server.PromptServer.instance.routes.get(APP_ROOT_URI)
def entrance(request):
    return web.FileResponse(os.path.join(WEBROOT, "index.html"))

server.PromptServer.instance.routes.static(APP_ROOT_URI+"/static/css/", path=os.path.join(WEBROOT, "static/css"))
server.PromptServer.instance.routes.static(APP_ROOT_URI+"/static/js/", path=os.path.join(WEBROOT, "static/js"))
# server.PromptServer.instance.routes.static(APP_ROOT_URI+"/static/media/", path=os.path.join(WEBROOT, "static/media"))
server.PromptServer.instance.routes.static(APP_ROOT_URI,WEBROOT)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

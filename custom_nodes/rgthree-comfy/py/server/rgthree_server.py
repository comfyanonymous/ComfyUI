import os
from aiohttp import web
from server import PromptServer

from ..config import get_config_value
from .utils_server import set_default_page_resources, set_default_page_routes
from .routes_config import *
from .routes_model_info import *

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_WEB = os.path.abspath(f'{THIS_DIR}/../../web/')

routes = PromptServer.instance.routes


# Sometimes other pages (link_fixer, etc.) may want to import JS from the comfyui
# directory. To allows TS to resolve like '../comfyui/file.js', we'll also resolve any module HTTP
# to these routes.
set_default_page_resources("comfyui", routes)
set_default_page_resources("common", routes)
set_default_page_resources("lib", routes)

set_default_page_routes("link_fixer", routes)
if get_config_value('unreleased.models_page.enabled') is True:
  set_default_page_routes("models", routes)

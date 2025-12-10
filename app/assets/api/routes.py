from aiohttp import web

from app import user_manager

ROUTES = web.RouteTableDef()
USER_MANAGER: user_manager.UserManager | None = None

def register_assets_system(app: web.Application, user_manager_instance: user_manager.UserManager) -> None:
    global USER_MANAGER
    USER_MANAGER = user_manager_instance
    app.add_routes(ROUTES)

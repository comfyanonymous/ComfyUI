from aiohttp import web
from aiohttp_session import SimpleCookieStorage, session_middleware
from aiohttp_security import check_permission, \
    is_anonymous, remember, forget, \
    setup as setup_security, SessionIdentityPolicy
from aiohttp_security.abc import AbstractAuthorizationPolicy
from server import PromptServer


class SimpleJack_AuthorizationPolicy(AbstractAuthorizationPolicy):
    async def authorized_userid(self, identity):
        """Retrieve authorized user id.
        Return the user_id of the user identified by the identity
        or 'None' if no user exists related to the identity.
        """
        if identity == 'jack':
            return identity

    async def permits(self, identity, permission, context=None):
        """Check user permissions.
        Return True if the identity is allowed the permission
        in the current context, else return False.
        """
        return identity == 'jack' and permission in ('all',)


class PromptServerSecurity(PromptServer):
    def __init__(self, loop):
        super().__init__(loop)

        middleware = session_middleware(SimpleCookieStorage())
        self.app.middlewares.append(middleware)

        setup_security(self.app, SessionIdentityPolicy(), SimpleJack_AuthorizationPolicy())
        new_routes = web.RouteTableDef()
        self.old_routes = self.routes
        self.routes = new_routes
        routes = self.routes

        @routes.get('/login')
        async def handler_login(request):
            redirect_response = web.HTTPFound('/')
            await remember(request, redirect_response, 'jack')
            raise redirect_response

        @routes.get('/logout')
        async def handler_logout(request):
            redirect_response = web.HTTPFound('/')
            await forget(request, redirect_response)
            raise redirect_response

        @routes.get('/l')
        async def handler_root(request):
            is_logged = not await is_anonymous(request)
            return web.Response(text='''<html><head></head><body>
                    Hello, I'm Jack, I'm {logged} logged in.<br /><br />
                    <a href="/login">Log me in</a><br />
                    <a href="/logout">Log me out</a><br /><br />
                    Check my permissions,
                    when i'm logged in and logged out.<br />
                    <a href="/listen">Can I listen?</a><br />
                    <a href="/speak">Can I speak?</a><br />
                </body></html>'''.format(
                logged='' if is_logged else 'NOT',
            ), content_type='text/html')

    def add_routes(self):
        # super().add_routes()
        # self.route

        # self.app.router.add_get('/login', handler_login)
        # self.app.router.add_post('/login', handler_login)
        # self.app.router.add_get('/logout', handler_logout)
        # Use app.router.routes() to get the list of routes
        # Iterate through each route:
        # Check if route should be secured based on path
        # If so, use app.router.add_get() / add_post() to add a new secured version of the route with @check_permission

        old_routes = self.old_routes

        secure_routes = ['/','/infer', '/prompt']
        self.functions = {}
        for old_route in old_routes:
            if old_route.path in secure_routes:
                # If so, use app.router.add_get() / add_post() to add a new secured version of the route with @check_permission
                # check if post or get
                # also we are not using decorators for security we are using an await call
                # so we need to return a new function that calls the check_permission before the handler
                if old_route.method == 'POST':

                    async def wrapped_func(request):
                        await check_permission(request, "all")
                        prev_func = old_route.handler
                        return await prev_func(request)

                    self.functions[old_route.path+"_"+"post"] = wrapped_func
                    self.routes.post(old_route.path)(self.functions[old_route.path+"_"+"post"])

                elif old_route.method == 'GET':

                    async def wrapped_func(request):
                        await check_permission(request, "all")
                        prev_func = old_route.handler
                        return await prev_func(request)

                    #self.routes.get(old_route.path)(old_route.handler)
                    # self.routes.get(old_route.path)(wrapped_func)
                    self.functions[old_route.path+"_"+"get"] = wrapped_func
                    self.routes.get(old_route.path)(self.functions[old_route.path+"_"+"get"])

            else:
                # if not secured, just add the route
                if old_route.method == 'POST':
                    self.routes.post(old_route.path)(old_route.handler)
                elif old_route.method == 'GET':
                    self.routes.get(old_route.path)(old_route.handler)

        self.app.add_routes(self.routes)
        self.app.add_routes([
            web.static('/', self.web_root, follow_symlinks=True),
        ])

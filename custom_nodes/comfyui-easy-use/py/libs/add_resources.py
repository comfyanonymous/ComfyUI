import urllib.parse
from os import PathLike
from aiohttp import web
from aiohttp.web_urldispatcher import AbstractRoute, UrlDispatcher
from server import PromptServer
from pathlib import Path

# 文件限制大小（MB）
max_size = 50
def suffix_limiter(self: web.StaticResource, request: web.Request):
    suffixes = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg", ".ico", ".apng", ".tif", ".hdr", ".exr"}
    rel_url = request.match_info["filename"]
    try:
        filename = Path(rel_url)
        if filename.anchor:
            raise web.HTTPForbidden()
        filepath = self._directory.joinpath(filename).resolve()
        if filepath.exists() and filepath.suffix.lower() not in suffixes:
            raise web.HTTPForbidden(reason="File type is not allowed")
    finally:
        pass

def filesize_limiter(self: web.StaticResource, request: web.Request):
    rel_url = request.match_info["filename"]
    try:
        filename = Path(rel_url)
        filepath = self._directory.joinpath(filename).resolve()
        if filepath.exists() and filepath.stat().st_size > max_size * 1024 * 1024:
            raise web.HTTPForbidden(reason="File size is too large")
    finally:
        pass
class LimitResource(web.StaticResource):
    limiters = []

    def push_limiter(self, limiter):
        self.limiters.append(limiter)

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        try:
            for limiter in self.limiters:
                limiter(self, request)
        except (ValueError, FileNotFoundError) as error:
            raise web.HTTPNotFound() from error

        return await super()._handle(request)

    def __repr__(self) -> str:
        name = "'" + self.name + "'" if self.name is not None else ""
        return f'<LimitResource {name} {self._prefix} -> {self._directory!r}>'

class LimitRouter(web.StaticDef):
    def __repr__(self) -> str:
        info = []
        for name, value in sorted(self.kwargs.items()):
            info.append(f", {name}={value!r}")
        return f'<LimitRouter {self.prefix} -> {self.path}{"".join(info)}>'

    def register(self, router: UrlDispatcher) -> list[AbstractRoute]:
        # resource = router.add_static(self.prefix, self.path, **self.kwargs)
        def add_static(
            self: UrlDispatcher,
            prefix: str,
            path: PathLike,
            *,
            name=None,
            expect_handler=None,
            chunk_size: int = 256 * 1024,
            show_index: bool = False,
            follow_symlinks: bool = False,
            append_version: bool = False,
        ) -> web.AbstractResource:
            assert prefix.startswith("/")
            if prefix.endswith("/"):
                prefix = prefix[:-1]
            resource = LimitResource(
                prefix,
                path,
                name=name,
                expect_handler=expect_handler,
                chunk_size=chunk_size,
                show_index=show_index,
                follow_symlinks=follow_symlinks,
                append_version=append_version,
            )
            resource.push_limiter(suffix_limiter)
            resource.push_limiter(filesize_limiter)
            self.register_resource(resource)
            return resource
        resource = add_static(router, self.prefix, self.path, **self.kwargs)
        routes = resource.get_info().get("routes", {})
        return list(routes.values())

def path_to_url(path):
    if not path:
        return path
    path = path.replace("\\", "/")
    if not path.startswith("/"):
        path = "/" + path
    while path.startswith("//"):
        path = path[1:]
    path = path.replace("//", "/")
    return path

def add_static_resource(prefix, path,limit=False):
    app = PromptServer.instance.app
    prefix = path_to_url(prefix)
    prefix = urllib.parse.quote(prefix)
    prefix = path_to_url(prefix)
    if limit:
        route = LimitRouter(prefix, path, {"follow_symlinks": True})
    else:
        route = web.static(prefix, path, follow_symlinks=True)
    app.add_routes([route])
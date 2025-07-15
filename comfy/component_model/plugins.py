from typing import Callable, NamedTuple


class RouteTuple(NamedTuple):
    method: str
    path: str
    func: Callable


class _RoutesWrapper:
    def __init__(self):
        self.routes: list[RouteTuple] = []

    def _decorator_factory(self, method):
        def decorator(path):
            def wrapper(func):
                from ..cmd.server import PromptServer
                if PromptServer.instance is not None and not isinstance(PromptServer.instance.routes, _RoutesWrapper):
                    getattr(PromptServer.instance.routes, method)(path)(func)
                else:
                    self.routes.append(RouteTuple(method, path, func))
                return func

            return wrapper

        return decorator

    def get(self, path):
        return self._decorator_factory('get')(path)

    def post(self, path):
        return self._decorator_factory('post')(path)

    def put(self, path):
        return self._decorator_factory('put')(path)

    def delete(self, path):
        return self._decorator_factory('delete')(path)

    def patch(self, path):
        return self._decorator_factory('patch')(path)

    def head(self, path):
        return self._decorator_factory('head')(path)

    def options(self, path):
        return self._decorator_factory('options')(path)

    def route(self, method, path):
        return self._decorator_factory(method.lower())(path)

    def clear(self):
        self.routes.clear()


prompt_server_instance_routes = _RoutesWrapper()

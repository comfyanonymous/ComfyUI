import sys
from functools import wraps

def create_module_properties():
    """
    Example:
        >>> _module_properties = create_module_properties()

        >>> @_module_properties.getter
        >>> def _nodes():
        >>>     return ...

    This creates nodes as a property
    :return:
    """
    properties = {}
    patched_modules = set()

    def patch_module(module):
        if module in patched_modules:
            return

        def base_getattr(name):
            raise AttributeError(f"module '{module.__name__}' has no attribute '{name}'")

        old_getattr = getattr(module, '__getattr__', base_getattr)

        def new_getattr(name):
            if name in properties:
                return properties[name]()
            else:
                return old_getattr(name)

        module.__getattr__ = new_getattr
        patched_modules.add(module)

    class ModuleProperties:
        @staticmethod
        def getter(func):
            @wraps(func)
            def wrapper():
                return func()

            name = func.__name__
            properties[name[1:]] = wrapper

            module = sys.modules[func.__module__]
            patch_module(module)

            return wrapper

    return ModuleProperties()
import os
import sys
import inspect
from importlib.machinery import PathFinder

MAIN_PY = 'main_py'
CURRENT_DIRECTORY = 'cwd'
SITE_PACKAGES = 'site'


class ImportContext:
    def __init__(self, *module_names, order):
        self.module_names = module_names
        self.order = order
        self.original_modules = {}
        self.finder = None

    def __enter__(self):
        try:
            main_frame = next(f for f in reversed(inspect.stack()) if f.frame.f_globals.get('__name__') == '__main__')
            self.main_py_dir = os.path.dirname(os.path.abspath(main_frame.filename))
        except (StopIteration, AttributeError):
            self.main_py_dir = None

        self.original_modules = {name: sys.modules.get(name) for name in self.module_names}
        for name in self.module_names:
            if name in sys.modules:
                for mod_name in list(sys.modules.keys()):
                    if mod_name == name or mod_name.startswith(f"{name}."):
                        del sys.modules[mod_name]

        self.finder = self._CustomFinder(self.module_names, self.order, self.main_py_dir)
        sys.meta_path.insert(0, self.finder)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.finder in sys.meta_path:
            sys.meta_path.remove(self.finder)

        for name in self.module_names:
            for mod_name in list(sys.modules.keys()):
                if mod_name == name or mod_name.startswith(f"{name}."):
                    del sys.modules[mod_name]

        for name, mod in self.original_modules.items():
            if mod:
                sys.modules[name] = mod
            elif name in sys.modules:
                del sys.modules[name]

    class _CustomFinder(PathFinder):
        def __init__(self, names, order, main_py_dir):
            self.module_names = set(names)
            self.order = order
            self.main_py_dir = main_py_dir

        def find_spec(self, fullname, path=None, target=None):
            if fullname not in self.module_names:
                return None

            cwd_path = os.getcwd()
            site_paths = [p for p in sys.path if 'site-packages' in p]
            other_paths = [p for p in sys.path if p != cwd_path and p not in site_paths]

            search_path = []
            for source in self.order:
                if source == CURRENT_DIRECTORY:
                    search_path.append(cwd_path)
                elif source == MAIN_PY and self.main_py_dir:
                    search_path.append(self.main_py_dir)
                elif source == SITE_PACKAGES:
                    search_path.extend(site_paths)

            search_path.extend(other_paths)

            return super().find_spec(fullname, path=search_path, target=target)

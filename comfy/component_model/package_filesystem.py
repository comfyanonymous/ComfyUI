import importlib.resources
from fsspec.spec import AbstractFileSystem
from fsspec.registry import register_implementation


class PkgResourcesFileSystem(AbstractFileSystem):
    """
    An fsspec filesystem for reading Python package resources.

    Paths are expected in the format:
    pkg://<package.name>/path/to/resource.txt
    """

    protocol = "pkg"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._traversables = {}

    def _get_traversable(self, package_name):
        """Get or cache the root Traversable for a package."""
        if package_name not in self._traversables:
            try:
                # Get the Traversable object for the root of the package
                self._traversables[package_name] = importlib.resources.files(package_name)
            except ModuleNotFoundError as e:
                raise FileNotFoundError(f"Package '{package_name}' not found.") from e
        return self._traversables[package_name]

    def _resolve_path(self, path):
        """Split a pkg:// path into package name and resource path."""
        # Remove protocol and leading slashes
        path_no_proto = self._strip_protocol(path).lstrip('/')

        if not path_no_proto:
            raise ValueError("Path must include a package name.")

        parts = path_no_proto.split('/', 1)
        package_name = parts[0]

        resource_path = parts[1] if len(parts) > 1 else ""

        root = self._get_traversable(package_name)

        # Resolve the final resource Traversable
        if resource_path:
            resource = root.joinpath(resource_path)
        else:
            resource = root

        return resource

    def _open(
            self,
            path,
            mode="rb",
            **kwargs,
    ):
        """Open a file for reading."""
        if "w" in mode or "a" in mode or "x" in mode:
            raise NotImplementedError("Only read mode is supported.")

        try:
            resource = self._resolve_path(path)
            if not resource.is_file():
                raise FileNotFoundError(f"Path is not a file: {path}")
            return resource.open("rb")
        except (ModuleNotFoundError, FileNotFoundError):
            raise FileNotFoundError(f"Resource not found: {path}")
        except Exception as e:
            raise IOError(f"Failed to open resource {path}: {e}") from e

    def ls(self, path, detail=False, **kwargs):
        """List contents of a package directory."""
        try:
            resource = self._resolve_path(path)
            if not resource.is_dir():
                # If it's a file, 'ls' should return info on that file
                return [self.info(path)] if detail else [path]

            items = []
            for item in resource.iterdir():
                item_path = f"{path.rstrip('/')}/{item.name}"
                if detail:
                    items.append(self.info(item_path))
                else:
                    items.append(item_path)
            return items
        except (ModuleNotFoundError, FileNotFoundError):
            raise FileNotFoundError(f"Resource path not found: {path}")

    def info(self, path, **kwargs):
        """Get info about a resource."""
        try:
            resource = self._resolve_path(path)
            resource_type = "directory" if resource.is_dir() else "file"

            size = None
            if resource_type == 'file':
                # This is inefficient but demonstrates the principle
                try:
                    with resource.open('rb') as f:
                        size = len(f.read())
                except Exception:
                    size = None  # Could fail for some reason

            return {
                "name": path,
                "type": resource_type,
                "size": size,
            }
        except (ModuleNotFoundError, FileNotFoundError):
            raise FileNotFoundError(f"Resource not found: {path}")


# Register the filesystem with fsspec
register_implementation(PkgResourcesFileSystem.protocol, PkgResourcesFileSystem)

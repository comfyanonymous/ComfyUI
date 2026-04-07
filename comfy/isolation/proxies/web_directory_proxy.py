"""WebDirectoryProxy — serves isolated node web assets via RPC.

Child side: enumerates and reads files from the extension's web/ directory.
Host side: gets an RPC proxy that fetches file listings and contents on demand.

Only files with allowed extensions (.js, .html, .css) are served.
Directory traversal is rejected. File contents are base64-encoded for
safe JSON-RPC transport.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

from pyisolate import ProxiedSingleton

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = frozenset({".js", ".html", ".css"})

MIME_TYPES = {
    ".js": "application/javascript",
    ".html": "text/html",
    ".css": "text/css",
}


class WebDirectoryProxy(ProxiedSingleton):
    """Proxy for serving isolated extension web directories.

    On the child side, this class has direct filesystem access to the
    extension's web/ directory.  On the host side, callers get an RPC
    proxy whose method calls are forwarded to the child.
    """

    # {extension_name: absolute_path_to_web_dir}
    _web_dirs: dict[str, str] = {}

    @classmethod
    def register_web_dir(cls, extension_name: str, web_dir_path: str) -> None:
        """Register an extension's web directory (child-side only)."""
        cls._web_dirs[extension_name] = web_dir_path
        logger.info(
            "][ WebDirectoryProxy: registered %s -> %s",
            extension_name,
            web_dir_path,
        )

    def list_web_files(self, extension_name: str) -> List[Dict[str, str]]:
        """Return a list of servable files in the extension's web directory.

        Each entry is {"relative_path": "js/foo.js", "content_type": "application/javascript"}.
        Only files with allowed extensions are included.
        """
        web_dir = self._web_dirs.get(extension_name)
        if not web_dir:
            return []

        root = Path(web_dir)
        if not root.is_dir():
            return []

        result: List[Dict[str, str]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue
            rel = path.relative_to(root)
            result.append({
                "relative_path": str(PurePosixPath(rel)),
                "content_type": MIME_TYPES[ext],
            })
        return result

    def get_web_file(
        self, extension_name: str, relative_path: str
    ) -> Dict[str, Any]:
        """Return the contents of a single web file as base64.

        Raises ValueError for traversal attempts or disallowed file types.
        Returns {"content": <base64 str>, "content_type": <MIME str>}.
        """
        _validate_path(relative_path)

        web_dir = self._web_dirs.get(extension_name)
        if not web_dir:
            raise FileNotFoundError(
                f"No web directory registered for {extension_name}"
            )

        root = Path(web_dir)
        target = (root / relative_path).resolve()

        # Ensure resolved path is under the web directory
        if not str(target).startswith(str(root.resolve())):
            raise ValueError(f"Path escapes web directory: {relative_path}")

        if not target.is_file():
            raise FileNotFoundError(f"File not found: {relative_path}")

        ext = target.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Disallowed file type: {ext}")

        content_type = MIME_TYPES[ext]
        raw = target.read_bytes()

        return {
            "content": base64.b64encode(raw).decode("ascii"),
            "content_type": content_type,
        }


def _validate_path(relative_path: str) -> None:
    """Reject directory traversal and absolute paths."""
    if os.path.isabs(relative_path):
        raise ValueError(f"Absolute paths are not allowed: {relative_path}")
    if ".." in PurePosixPath(relative_path).parts:
        raise ValueError(f"Directory traversal is not allowed: {relative_path}")


# ---------------------------------------------------------------------------
# Host-side cache and aiohttp handler
# ---------------------------------------------------------------------------


class WebDirectoryCache:
    """Host-side in-memory cache for proxied web directory contents.

    Populated lazily via RPC calls to the child's WebDirectoryProxy.
    Once a file is cached, subsequent requests are served from memory.
    """

    def __init__(self) -> None:
        # {extension_name: {relative_path: {"content": bytes, "content_type": str}}}
        self._file_cache: dict[str, dict[str, dict[str, Any]]] = {}
        # {extension_name: [{"relative_path": str, "content_type": str}, ...]}
        self._listing_cache: dict[str, list[dict[str, str]]] = {}
        # {extension_name: WebDirectoryProxy (RPC proxy instance)}
        self._proxies: dict[str, Any] = {}

    def register_proxy(self, extension_name: str, proxy: Any) -> None:
        """Register an RPC proxy for an extension's web directory."""
        self._proxies[extension_name] = proxy
        logger.info(
            "][ WebDirectoryCache: registered proxy for %s", extension_name
        )

    @property
    def extension_names(self) -> list[str]:
        return list(self._proxies.keys())

    def list_files(self, extension_name: str) -> list[dict[str, str]]:
        """List servable files for an extension (cached after first call)."""
        if extension_name not in self._listing_cache:
            proxy = self._proxies.get(extension_name)
            if proxy is None:
                return []
            try:
                self._listing_cache[extension_name] = proxy.list_web_files(
                    extension_name
                )
            except Exception:
                logger.warning(
                    "][ WebDirectoryCache: failed to list files for %s",
                    extension_name,
                    exc_info=True,
                )
                return []
        return self._listing_cache[extension_name]

    def get_file(
        self, extension_name: str, relative_path: str
    ) -> dict[str, Any] | None:
        """Get file content (cached after first fetch). Returns None on miss."""
        ext_cache = self._file_cache.get(extension_name)
        if ext_cache and relative_path in ext_cache:
            return ext_cache[relative_path]

        proxy = self._proxies.get(extension_name)
        if proxy is None:
            return None

        try:
            result = proxy.get_web_file(extension_name, relative_path)
        except (FileNotFoundError, ValueError):
            return None
        except Exception:
            logger.warning(
                "][ WebDirectoryCache: failed to fetch %s/%s",
                extension_name,
                relative_path,
                exc_info=True,
            )
            return None

        decoded = {
            "content": base64.b64decode(result["content"]),
            "content_type": result["content_type"],
        }

        if extension_name not in self._file_cache:
            self._file_cache[extension_name] = {}
        self._file_cache[extension_name][relative_path] = decoded
        return decoded


# Global cache instance — populated during isolation loading
_web_directory_cache = WebDirectoryCache()


def get_web_directory_cache() -> WebDirectoryCache:
    return _web_directory_cache

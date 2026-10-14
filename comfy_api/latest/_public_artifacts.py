"""Declared, opaque model artifacts. This module never deserializes model code."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time
import urllib.parse
import urllib.request


_CATALOGUES = frozenset({"nlf"})
_DELIVERY_HOSTS = frozenset({
    "github.com", "api.github.com", "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
})
_CHUNK = 1024 * 1024
_METADATA_MAX = 1024 * 1024
_INSTALL_LOCK = threading.Lock()


@dataclass(frozen=True)
class GitHubReleaseArtifact:
    """Digest/size-pinned opaque TorchScript, declared in SDK_REQUIRED_ARTIFACTS.

    ``nlf`` is the first enabled opaque catalogue. Its files are never passed
    to a host tensor loader or JIT. Loading code is exclusively guest work.
    There is no caller-selected URL, token, endpoint or destination path.
    """
    repo_id: str
    release_id: int
    release_tag: str
    asset_id: int
    filename: str
    folder: str
    sha256: str
    size: int
    on_demand: bool = True

    def __post_init__(self):
        component = r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
        if (not isinstance(self.repo_id, str) or len(self.repo_id) > 193
                or re.fullmatch(component + "/" + component, self.repo_id) is None
                or ".." in self.repo_id or "--" in self.repo_id
                or self.repo_id.endswith(".git")):
            raise ValueError("artifact repo_id must name a GitHub owner/repository")
        for field in ("release_id", "asset_id", "size"):
            value = getattr(self, field)
            if type(value) is not int or not 0 < value <= 2**63 - 1:
                raise ValueError(f"artifact {field} must be a positive integer")
        if (not isinstance(self.release_tag, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.release_tag) is None
                or ".." in self.release_tag):
            raise ValueError("artifact release_tag must be a confined release tag")
        if (not isinstance(self.filename, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", self.filename) is None
                or ".." in self.filename or not self.filename.endswith(".torchscript")):
            raise ValueError("opaque artifacts require a confined .torchscript basename")
        if not isinstance(self.folder, str) or self.folder not in _CATALOGUES:
            raise ValueError("artifact folder must be an enabled opaque catalogue")
        if (not isinstance(self.sha256, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256) is None):
            raise ValueError("artifact sha256 must be 64 hex digits")
        if type(self.on_demand) is not bool:
            raise TypeError("artifact on_demand must be a bool")
        object.__setattr__(self, "sha256", self.sha256.lower())

    @property
    def catalogue_name(self) -> str:
        return (f"github-release/{self.repo_id}/{self.release_id}/"
                f"{self.asset_id}/{self.sha256}/{self.filename}")

    @property
    def source_url(self) -> str:
        return (f"https://github.com/{self.repo_id}/releases/download/"
                f"{self.release_tag}/{self.filename}")


def _checked_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in _DELIVERY_HOSTS
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError("artifact redirect must remain on public GitHub asset delivery")
    return url


class _PublicGitHubRedirects(urllib.request.HTTPRedirectHandler):
    max_repeats = 2
    max_redirections = 5

    def redirect_request(self, request, fp, code, message, headers, newurl):
        _checked_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


@dataclass(frozen=True)
class _Acquisition:
    end: float
    cancelled: threading.Event


def _remaining(deadline: _Acquisition) -> float:
    if deadline.cancelled.is_set():
        raise InterruptedError("public artifact acquisition cancelled")
    remaining = deadline.end - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("public artifact acquisition deadline expired")
    return min(remaining, 10.0)


def _open(opener, url: str, accept: str, deadline: _Acquisition):
    request = urllib.request.Request(_checked_url(url), headers={
        "Accept": accept, "Accept-Encoding": "identity",
        "User-Agent": "ComfyUI-public-model-artifact",
    })
    response = opener.open(request, timeout=_remaining(deadline))
    try:
        _checked_url(response.geturl())
        if response.status != 200:
            raise ValueError("public artifact request did not return HTTP 200")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("public artifact content encoding must be identity")
        return response
    except BaseException:
        response.close()
        raise


def _chunks(response, limit: int, deadline: _Acquisition):
    total = 0
    while True:
        _remaining(deadline)
        # read1 returns available bytes instead of waiting to fill a large
        # buffer; a slow trickle therefore cannot renew the absolute deadline.
        chunk = response.read1(min(_CHUNK, limit - total + 1))
        _remaining(deadline)
        if not chunk:
            return
        total += len(chunk)
        if total > limit:
            raise ValueError("public artifact response exceeds its byte limit")
        yield chunk


def _metadata(opener, url: str, deadline: _Acquisition):
    with _open(opener, url, "application/vnd.github+json", deadline) as response:
        value = json.loads(b"".join(_chunks(response, _METADATA_MAX, deadline)))
    if not isinstance(value, dict):
        raise ValueError("GitHub release metadata must be an object")
    return value


def _destination(artifact: GitHubReleaseArtifact) -> Path:
    import folder_paths

    registered = folder_paths.folder_names_and_paths.get(artifact.folder)
    if not registered or not registered[0] or ".torchscript" not in registered[1]:
        raise ValueError("opaque artifact catalogue is not registered for .torchscript")
    root = Path(registered[0][0]).resolve()
    destination = root.joinpath(*artifact.catalogue_name.split("/"))
    # Reject linked descendants even if their resolved location is inside the
    # root. The root itself is trusted host catalogue configuration.
    for part in (destination, *destination.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("artifact catalogue descendants must not be symlinks")
    if not destination.resolve().is_relative_to(root):
        raise ValueError("artifact destination escapes its catalogue")
    return destination


def _verify_cached(path: Path, artifact: GitHubReleaseArtifact, deadline: _Acquisition):
    digest = hashlib.sha256()
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("cached public artifact must be a regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as file:
        before = os.fstat(file.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != artifact.size:
            raise ValueError("cached public artifact size/type does not match")
        for chunk in iter(lambda: file.read(_CHUNK), b""):
            _remaining(deadline)
            digest.update(chunk)
        after = os.fstat(file.fileno())
    identity = lambda value: (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(path.stat()):
        raise ValueError("cached public artifact changed during verification")
    if digest.hexdigest() != artifact.sha256:
        raise ValueError("cached public artifact sha256 does not match")


def download_github_release_artifact(
    artifact: GitHubReleaseArtifact, cancelled: threading.Event | None = None,
) -> str:
    """Bounded explicit acquisition; atomically publish only verified bytes."""
    if not isinstance(artifact, GitHubReleaseArtifact):
        raise TypeError("public artifact must be a GitHubReleaseArtifact declaration")
    max_bytes = int(os.environ.get("COMFY_SECURE_PUBLIC_ARTIFACT_MAX", str(512 * 1024 * 1024)))
    max_bytes = min(max_bytes, int(os.environ.get("COMFY_SECURE_HF_WEIGHT_MAX", str(64 * 1024**3))))
    seconds = int(os.environ.get("COMFY_SECURE_PUBLIC_ARTIFACT_SECONDS", "300"))
    if max_bytes <= 0 or not 0 < seconds <= 900:
        raise ValueError("public artifact byte/deadline limits must be positive and finite")
    if artifact.size > max_bytes:
        raise ValueError("public artifact exceeds the configured per-artifact byte limit")
    deadline = _Acquisition(time.monotonic() + seconds, cancelled or threading.Event())
    if not _INSTALL_LOCK.acquire(timeout=_remaining(deadline)):
        raise TimeoutError("public artifact installer is busy")
    temporary = None
    try:
        destination = _destination(artifact)
        if destination.exists():
            _verify_cached(destination, artifact, deadline)
            return artifact.catalogue_name
        # No ambient proxies, credentials, cookies or caller-supplied headers.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _PublicGitHubRedirects())
        release = _metadata(opener, f"https://api.github.com/repos/{artifact.repo_id}/releases/{artifact.release_id}", deadline)
        if (type(release.get("id")) is not int or release.get("id") != artifact.release_id
                or release.get("tag_name") != artifact.release_tag):
            raise ValueError("GitHub release identity does not match the declaration")
        assets = release.get("assets")
        if not isinstance(assets, list):
            raise ValueError("GitHub release assets must be a list")
        selected = [item for item in assets if isinstance(item, dict)
                    and type(item.get("id")) is int and item.get("id") == artifact.asset_id]
        if len(selected) != 1:
            raise ValueError("GitHub release does not contain the declared asset")
        asset = selected[0]
        if (asset.get("name") != artifact.filename or type(asset.get("size")) is not int
                or asset.get("size") != artifact.size
                or asset.get("browser_download_url") != artifact.source_url):
            raise ValueError("GitHub asset identity/length does not match the declaration")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Recheck descendants after mkdir before opening the private temporary.
        if _destination(artifact) != destination:
            raise ValueError("artifact destination changed")
        descriptor, temporary = tempfile.mkstemp(prefix=".public-artifact-", dir=destination.parent)
        digest = hashlib.sha256(); total = 0
        with os.fdopen(descriptor, "wb") as output:
            with _open(opener, artifact.source_url, "application/octet-stream", deadline) as response:
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdecimal() or int(length) != artifact.size):
                    raise ValueError("GitHub artifact Content-Length does not match")
                for chunk in _chunks(response, artifact.size, deadline):
                    output.write(chunk); digest.update(chunk); total += len(chunk)
            if total != artifact.size or digest.hexdigest() != artifact.sha256:
                raise ValueError("public artifact sha256/length does not match")
            output.flush(); os.fsync(output.fileno())
        _remaining(deadline)
        if _destination(artifact) != destination:
            raise ValueError("artifact destination changed before publication")
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
        temporary = None
        return artifact.catalogue_name
    finally:
        try:
            if temporary is not None:
                os.unlink(temporary)
        finally:
            _INSTALL_LOCK.release()

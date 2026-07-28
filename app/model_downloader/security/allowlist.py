"""URL allowlist for server-side model fetches.

Default-deny. A URL is downloadable only when its parsed host + scheme are
allowlisted AND (unless explicitly relaxed) its final filename ends in a
known model extension.

The built-in host defaults mirror the frontend's ``isModelDownloadable``
allowlist so the two flows agree on what is eligible; ``--download-allowed-hosts``
extends it for self-hosted mirrors. Matching is done on ``urlparse().hostname``
(never a raw string prefix) so userinfo tricks like
``http://127.0.0.1@169.254.169.254/x.safetensors`` — whose real host is the
metadata IP — cannot slip past.
"""

from __future__ import annotations

from urllib.parse import urlparse

from comfy.cli_args import args

# host -> set of allowed schemes. Frontend parity (HuggingFace / Civitai /
# localhost). Extra hosts from --download-allowed-hosts are https-only.
_DEFAULT_ALLOWED_HOSTS: dict[str, set[str]] = {
    "huggingface.co": {"https"},
    "civitai.com": {"https"},
    "localhost": {"http", "https"},
    "127.0.0.1": {"http", "https"},
}

# Hosts for which loopback addresses are intentionally permitted (the localhost
# "download a local model" feature). Every other host's loopback resolution is
# rejected by the SSRF resolver.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# Known model file extensions (frontend parity). Checked on the final filename.
ALLOWED_MODEL_EXTENSIONS = (
    ".safetensors",
    ".sft",
    ".ckpt",
    ".pth",
    ".pt",
    ".gguf",
    ".bin",
)


def _allowed_hosts() -> dict[str, set[str]]:
    hosts = {h: set(s) for h, s in _DEFAULT_ALLOWED_HOSTS.items()}
    for extra in getattr(args, "download_allowed_hosts", []) or []:
        host = extra.strip().lower()
        if host:
            hosts.setdefault(host, set()).add("https")
    return hosts


def is_host_allowed(host: str | None, scheme: str | None) -> bool:
    """True iff ``host`` is allowlisted for ``scheme``.

    Used both for the initial URL and re-checked on every redirect hop,
    so a whitelisted URL cannot 30x into an off-list host.
    """
    if not host or not scheme:
        return False
    allowed = _allowed_hosts().get(host.lower())
    return allowed is not None and scheme.lower() in allowed


def has_allowed_extension(path: str, allow_any_extension: bool = False) -> bool:
    if allow_any_extension:
        return True
    return path.lower().endswith(ALLOWED_MODEL_EXTENSIONS)


def filename_extension(name: str) -> str:
    """Lowercased extension (including the leading dot) of a bare filename.

    Returns ``""`` when there is no extension. A leading-dot name
    (``.safetensors``) is treated as having no extension (all stem), matching
    ``os.path.splitext`` semantics so dotfiles aren't mistaken for typed files.
    """
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    dot = base.rfind(".")
    if dot <= 0:
        return ""
    return base[dot:].lower()


def is_allowed_extension_name(name: str) -> bool:
    """True iff ``name`` ends in one of the known model extensions."""
    return name.lower().endswith(ALLOWED_MODEL_EXTENSIONS)


def is_host_allowed_url(url: str) -> bool:
    """True iff ``url`` parses and its host+scheme are allowlisted."""
    if not isinstance(url, str) or not url:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return is_host_allowed(parsed.hostname, parsed.scheme)


def url_path_extension(url: str) -> str:
    """Extension of the URL *path* basename (query ignored), or ``""``."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    return filename_extension(parsed.path)


def is_url_downloadable(url: str) -> bool:
    """Coarse enqueue gate: host/scheme allowed and extension not disallowed.

    Unlike :func:`is_url_allowed` (which demands a known extension *in the URL*),
    this also admits URLs whose path carries no extension at all — e.g. a Civitai
    ``/api/download/models/<id>`` endpoint whose real filename only shows up in
    the redirect target / ``Content-Disposition``. The true extension is then
    resolved from the network and re-validated before the download is admitted.
    A path bearing an explicit *non-model* extension (``.zip``, ``.html``, ...)
    is still rejected here.
    """
    if not is_host_allowed_url(url):
        return False
    ext = url_path_extension(url)
    return ext == "" or ext in ALLOWED_MODEL_EXTENSIONS


def is_url_allowed(url: str, allow_any_extension: bool = False) -> bool:
    """Check whether ``url`` is permitted as a server-side download source."""
    if not isinstance(url, str) or not url:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if not is_host_allowed(parsed.hostname, parsed.scheme):
        return False
    return has_allowed_extension(parsed.path, allow_any_extension)

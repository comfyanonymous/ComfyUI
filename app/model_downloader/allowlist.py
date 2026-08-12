"""URL allowlist for server-side model fetches.

Mirrors the frontend's ``isModelDownloadable`` allowlist so the two flows
agree on which URLs are eligible for download. Server-side allowlisting is
the primary SSRF defense for this subsystem — workflow JSON is untrusted
input (anyone can hand-craft one), so we never let the server fetch URLs
outside this list.
"""

from urllib.parse import urlparse

# Frontend parity: ``missingModelDownload-*.js`` exports the same triple
# (Civitai / HuggingFace / localhost). Keyed by exact hostname → allowed
# schemes, and matched against the *parsed* host (not a raw string prefix),
# so URL-userinfo tricks can't slip past — see ``is_url_allowed``.
_ALLOWED_HOSTS = {
    "huggingface.co": {"https"},
    "civitai.com": {"https"},
    "localhost": {"http"},
    "127.0.0.1": {"http"},
}

# Frontend parity: same set as ``a = [...]`` in the bundle.
_ALLOWED_MODEL_EXTENSIONS = (
    ".safetensors",
    ".sft",
    ".ckpt",
    ".pth",
    ".pt",
)


def is_url_allowed(url: str) -> bool:
    """Check whether ``url`` is permitted as a server-side download source.

    True only when the parsed host + scheme are allowlisted AND the path ends
    in a model extension. Matching on ``parsed.hostname`` (not a string prefix)
    defeats userinfo tricks like ``http://127.0.0.1:80@169.254.169.254/x.safetensors``,
    whose real host is ``169.254.169.254``; the extension check rejects non-model
    URLs on allowed hosts (e.g. ``huggingface.co/api/...``).
    """
    if not isinstance(url, str) or not url:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = parsed.hostname
    if host is None or parsed.scheme not in _ALLOWED_HOSTS.get(host, ()):
        return False
    return any(parsed.path.endswith(ext) for ext in _ALLOWED_MODEL_EXTENSIONS)

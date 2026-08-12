"""Parsers for the ``huggingface.co`` URL shape we accept in workflows.

The download API accepts URLs of the form
``https://huggingface.co/<org>/<repo>/resolve/<rev>/<path/to/file>``.
We need to recover ``<org>/<repo>`` (the *repo_id*) from such URLs for
``huggingface_hub`` API calls (notably ``HfApi.auth_check``).
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

_HF_HOST = "huggingface.co"


def is_hf_url(url: str) -> bool:
    """Cheap host check — does this URL point at huggingface.co?"""
    try:
        return urlparse(url).hostname == _HF_HOST
    except ValueError:
        return False


def repo_id_from_url(url: str) -> Optional[str]:
    """Extract ``<org>/<repo>`` from an HF model file URL.

    Returns ``None`` if the URL isn't on huggingface.co or doesn't look
    like a model-file URL. The expected shape is
    ``/<org>/<repo>/resolve/<rev>/<path>`` — anything else
    (datasets, spaces, /tree/, /blob/, …) we treat as out of scope here.
    """
    if not is_hf_url(url):
        return None
    parts = urlparse(url).path.lstrip("/").split("/")
    if len(parts) < 4 or parts[2] != "resolve":
        return None
    org, repo = parts[0], parts[1]
    if not org or not repo:
        return None
    return f"{org}/{repo}"

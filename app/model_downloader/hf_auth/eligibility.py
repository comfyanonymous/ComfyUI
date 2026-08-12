"""Whether this deployment is allowed to do interactive HF OAuth.

We only let the server hold a HuggingFace token under a strict trust
assumption: this is a *single tenant local* install. Concretely:

  - The server is bound to a loopback address. SSH tunneling /
    reverse-proxies can defeat this, but it's the strongest signal
    we have without an authentication system.
  - ``--multi-user`` is off. A shared token used implicitly by multiple
    declared users would be a footgun — one user's gated downloads
    would silently authenticate as another.

Anything else and the frontend hides the HF login UI entirely; gated
models continue to show the "acquire it manually" message.
"""

from __future__ import annotations

import ipaddress
import socket

from comfy.cli_args import args


def _is_loopback(host: str | None) -> bool:
    """Duplicates ``server.is_loopback`` (small, no shared module yet).

    Resolves a host or IP literal to whether it lives on the loopback
    interface (127.0.0.0/8 for IPv4, ::1 for IPv6). Returns False for
    ``0.0.0.0`` / ``::`` because those are bind-all wildcards, not
    loopback.
    """
    if host is None:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass

    loopback = False
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            r = socket.getaddrinfo(host, None, family, socket.SOCK_STREAM)
            for _family, _, _, _, sockaddr in r:
                if not ipaddress.ip_address(sockaddr[0]).is_loopback:
                    return loopback
                loopback = True
        except socket.gaierror:
            pass
    return loopback


def is_hf_auth_eligible() -> bool:
    """True iff this deployment may surface the HF OAuth flow."""
    return _is_loopback(args.listen) and not args.multi_user

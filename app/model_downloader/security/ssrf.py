"""SSRF / exfiltration defenses.

Two cooperating layers:

1. :class:`ValidatingResolver` is installed on the shared connector. Every
   connection — the initial probe and every segment GET, including ones made
   after a redirect — resolves its host through this resolver, which rejects
   any address that lands on a private / special-use IP range. Because the
   resolve and the connect happen together inside the connector, there is no
   check-then-connect window for DNS rebinding to exploit.

2. :func:`check_redirect_hop` re-validates every hop. The host allowlist gates
   only the *initial* user-supplied URL (anti-SSRF for arbitrary input);
   legitimate downloads from allowlisted origins redirect to presigned CDN
   hosts that are deliberately NOT on the allowlist (HF ->
   ``cdn-lfs*.huggingface.co``, Civitai -> signed Cloudflare/S3), so hops are
   instead screened for scheme, embedded credentials, and — via the resolver
   above — private IPs. Credentials are only ever attached when a hop's host
   exactly matches a stored credential, so they are dropped on the CDN hop.
   Loopback (the "download a local model" feature) is exempt from IP filtering
   only for the initial URL: a *redirect* may never target a loopback host or
   a blocked IP-literal, which the resolver alone can't enforce (it exempts
   loopback literals and never sees IP literals through DNS).
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from aiohttp.abc import AbstractResolver
from aiohttp.resolver import DefaultResolver

from app.model_downloader.security.allowlist import LOOPBACK_HOSTS

# Cap the redirect chain length a hop may use.
MAX_REDIRECTS = 5


class SSRFError(Exception):
    """A hop failed an SSRF / allowlist check."""


def is_scheme_allowed(scheme: str | None, host: str | None) -> bool:
    """True iff ``scheme`` is permitted for ``host`` on a download hop.

    https is always allowed; plain http only for loopback/approved dev hosts.
    """
    if not scheme:
        return False
    scheme = scheme.lower()
    if scheme == "https":
        return True
    if scheme == "http":
        return bool(host) and host.lower() in LOOPBACK_HOSTS
    return False


def is_blocked_ip(ip_str: str) -> bool:
    """True for any address we refuse to connect to.

    Covers loopback, link-local (incl. 169.254.169.254 cloud metadata),
    RFC1918 private ranges, unique-local (ULA), unspecified (0.0.0.0/::),
    multicast and other reserved ranges.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable -> refuse
    # On CPython before the gh-113171 fix (backported to 3.12.4/3.11.9/
    # 3.10.14/3.9.19) the is_* properties don't see through IPv4-mapped IPv6
    # (e.g. ::ffff:169.254.169.254), so resolve and re-check the embedded IPv4
    # to keep mapped metadata/private addresses from slipping past the filter.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


class ValidatingResolver(AbstractResolver):
    """Delegating resolver that drops blocked IPs from every resolution.

    If a hostname resolves only to blocked addresses, the connection fails
    closed with an :class:`OSError`, which aiohttp surfaces as a connection
    error to the caller.
    """

    def __init__(self) -> None:
        self._inner = DefaultResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        infos = await self._inner.resolve(host, port, family)
        # localhost/127.0.0.1 are an explicit, opt-in allowlist feature.
        if isinstance(host, str) and host.lower() in LOOPBACK_HOSTS:
            return infos
        safe = [info for info in infos if not is_blocked_ip(info["host"])]
        if not safe:
            raise OSError(
                f"refusing to connect to {host!r}: resolves only to "
                f"private/special-use addresses"
            )
        return safe

    async def close(self) -> None:
        await self._inner.close()


def check_redirect_hop(url: str, *, is_initial_url: bool = False) -> str:
    """Validate one hop's URL.

    Returns the URL unchanged on success; raises :class:`SSRFError` otherwise.
    Requires https for external hosts (http only for loopback/approved dev
    hosts) and forbids credentials-in-URL. The host is NOT re-checked against
    the allowlist (CDN redirect targets are off-list by design); credential
    leakage is prevented by exact host matching at attach time, and the landing
    filename's extension is gated separately by the caller.

    Loopback/blocked-IP screening: the connector's resolver filters resolvable
    hostnames but exempts literal loopback hosts (``localhost``/``127.0.0.1``/
    ``::1``) and never sees IP literals through DNS. That loopback exemption is
    legitimate only for the *initial* user-supplied URL (``is_initial_url``);
    on a redirect hop we reject loopback hosts and any blocked IP-literal here,
    so a 30x can't steer a server-side GET at loopback/internal services.
    """
    try:
        parsed = urlparse(url)
    except ValueError as e:
        raise SSRFError(f"unparseable redirect URL {url!r}: {e}") from e
    host = parsed.hostname
    if not host:
        raise SSRFError(f"redirect URL has no host: {url!r}")
    if not is_scheme_allowed(parsed.scheme, host):
        raise SSRFError(
            f"redirect to disallowed scheme {parsed.scheme!r} for host "
            f"{host!r} (https required for external hosts)"
        )
    if parsed.username or parsed.password:
        raise SSRFError("credentials-in-URL are not allowed")
    host_is_loopback = host.lower() in LOOPBACK_HOSTS
    if not is_initial_url and host_is_loopback:
        raise SSRFError(f"redirect to loopback host {host!r} is not allowed")
    # IP-literal targets never go through DNS, so the connector's resolver can't
    # screen them — check them directly. The only blocked IP allowed through is
    # a loopback literal on the initial URL (handled by the exemption above).
    try:
        ipaddress.ip_address(host)
    except ValueError:
        is_ip_literal = False
    else:
        is_ip_literal = True
    if is_ip_literal and is_blocked_ip(host) and not (
        is_initial_url and host_is_loopback
    ):
        raise SSRFError(f"redirect to blocked internal address {host!r}")
    return url

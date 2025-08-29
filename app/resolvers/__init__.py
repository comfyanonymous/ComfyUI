import contextlib
from dataclasses import dataclass
from typing import Protocol, Optional, Mapping


@dataclass
class ResolveResult:
    provider: str                       # e.g., "gcs"
    download_url: str                   # fully-qualified URL to fetch bytes
    headers: Mapping[str, str]          # optional auth headers etc
    expected_size: Optional[int] = None
    tags: Optional[list[str]] = None    # e.g. ["models","vae","subdir"]
    filename: Optional[str] = None      # preferred basename

class AssetResolver(Protocol):
    provider: str
    async def resolve(self, asset_hash: str) -> Optional[ResolveResult]: ...


_REGISTRY: list[AssetResolver] = []


def register_resolver(resolver: AssetResolver) -> None:
    """Append Resolver with simple de-dup per provider."""
    global _REGISTRY
    _REGISTRY = [r for r in _REGISTRY if r.provider != resolver.provider] + [resolver]


async def resolve_asset(asset_hash: str) -> Optional[ResolveResult]:
    for r in _REGISTRY:
        with contextlib.suppress(Exception):  # For Resolver failure we just try the next one
            res = await r.resolve(asset_hash)
            if res:
                return res
    return None

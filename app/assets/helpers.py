from datetime import datetime, timezone


def escape_sql_like_string(s: str, escape: str = "!") -> tuple[str, str]:
    """Escapes %, _ and the escape char in a LIKE prefix.

    Returns (escaped_prefix, escape_char).
    """
    s = s.replace(escape, escape + escape)  # escape the escape char first
    s = s.replace("%", escape + "%").replace("_", escape + "_")  # escape LIKE wildcards
    return s, escape


def get_utc_now() -> datetime:
    """Naive UTC timestamp (no tzinfo). We always treat DB datetimes as UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def normalize_tags(tags: list[str] | None) -> list[str]:
    """
    Normalize a list of tags by:
      - Stripping whitespace.
      - Removing exact duplicates while preserving order and case.
    """
    return list(dict.fromkeys(t.strip() for t in (tags or []) if (t or "").strip()))


def to_stored_hash(digest: str) -> str:
    """Convert a bare BLAKE3 digest to its stored form."""
    return f"blake3:{digest}"


def validate_blake3_hash(s: str) -> str:
    """Validate and normalize a blake3 hash string.

    Returns canonical 'blake3:<hex>' or raises ValueError.
    """
    s = s.strip().lower()
    if not s or ":" not in s:
        raise ValueError("hash must be 'blake3:<hex>'")
    algo, digest = s.split(":", 1)
    if (
        algo != "blake3"
        or len(digest) != 64
        or any(c for c in digest if c not in "0123456789abcdef")
    ):
        raise ValueError("hash must be 'blake3:<hex>'")
    return f"{algo}:{digest}"

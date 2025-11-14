def escape_like_prefix(s: str, escape: str = "!") -> tuple[str, str]:
    """Escapes %, _ and the escape char itself in a LIKE prefix.
    Returns (escaped_prefix, escape_char). Caller should append '%' and pass escape=escape_char to .like().
    """
    s = s.replace(escape, escape + escape)  # escape the escape char first
    s = s.replace("%", escape + "%").replace("_", escape + "_")  # escape LIKE wildcards
    return s, escape

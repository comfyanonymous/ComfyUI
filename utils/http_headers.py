import re
import urllib.parse


_CONTROL_CHARS_RE = re.compile(r"[\r\n]+")
_QUOTED_FILENAME_UNSAFE_RE = re.compile(r'["\\]')
_NON_ASCII_RE = re.compile(r"[^\x20-\x7E]")


def content_disposition_header(filename: str, disposition: str) -> str:
    if disposition not in {"inline", "attachment"}:
        raise ValueError(f"Unsupported Content-Disposition type: {disposition}")

    safe_filename = _CONTROL_CHARS_RE.sub("_", filename or "")
    safe_filename = _QUOTED_FILENAME_UNSAFE_RE.sub("_", safe_filename)
    fallback_filename = _NON_ASCII_RE.sub("_", safe_filename)
    encoded_filename = urllib.parse.quote(safe_filename, safe="")
    return (
        f'{disposition}; filename="{fallback_filename}"; '
        f"filename*=UTF-8''{encoded_filename}"
    )

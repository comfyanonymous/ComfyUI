import os
from urllib.parse import quote


def content_disposition_for_file(filename: str, disposition_type: str = "inline") -> str:
    filename = filename.replace("\r", "_").replace("\n", "_")
    if filename.isascii():
        fallback_filename = filename
    else:
        _, ext = os.path.splitext(filename)
        fallback_filename = f"download{ext}" if ext.isascii() else "download"

    escaped_filename = fallback_filename.replace("\\", "\\\\").replace('"', '\\"')
    header = f'{disposition_type}; filename="{escaped_filename}"'

    if fallback_filename != filename:
        header += f"; filename*=UTF-8''{quote(filename, safe='')}"

    return header

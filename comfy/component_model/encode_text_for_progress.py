from __future__ import annotations

import struct


def encode_text_for_progress(node_id, text):
    if isinstance(text, str):
        text = text.encode("utf-8")
    node_id_bytes = str(node_id).encode("utf-8")
    # Pack the node_id length as a 4-byte unsigned integer, followed by the node_id bytes
    message = struct.pack(">I", len(node_id_bytes)) + node_id_bytes + text
    return message

import asyncio

try:
    from collections.abc import Buffer
except ImportError:
    from typing_extensions import Buffer
from io import BytesIO
from typing import Literal, AsyncGenerator

import ijson
import aiofiles
import sys
import shlex


async def stream_json_objects(source_path_or_stdin: str | Literal["-"]) -> AsyncGenerator[dict, None]:
    """
    Asynchronously yields JSON objects from a given source.
    The source can be a file path or "-" for stdin.
    Assumes the input stream contains concatenated JSON objects (e.g., {}{}{}).
    """
    if source_path_or_stdin is None or len(source_path_or_stdin) == 0:
        return
    elif source_path_or_stdin == "-":
        async for obj in ijson.items_async(aiofiles.stdin_bytes, '', multiple_values=True):
            yield obj
    else:
        # Handle file path or literal JSON
        if "{" in source_path_or_stdin[:2]:
            # literal string
            encode: Buffer = source_path_or_stdin.encode("utf-8")
            source_path_or_stdin = BytesIO(encode)
            for obj in ijson.items(source_path_or_stdin, '', multiple_values=True):
                yield obj
        else:
            async with aiofiles.open(source_path_or_stdin, mode='rb') as f:
                # 'rb' mode is important as ijson expects byte streams.
                # The prefix '' targets root-level objects.
                # multiple_values=True allows parsing of multiple top-level JSON values.
                async for obj in ijson.items_async(f, '', multiple_values=True):
                    yield obj

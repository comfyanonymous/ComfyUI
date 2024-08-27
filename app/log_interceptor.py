import io
import threading
from datetime import datetime

from collections import deque
logs = deque(maxlen=300)

class LogInterceptor(io.TextIOWrapper):
    def __init__(self, stream,  *args, **kwargs):
        buffer = stream.buffer
        encoding = stream.encoding
        super().__init__(buffer, *args, **kwargs, encoding=encoding)
        self._lock = threading.Lock()

    def write(self, data):
        with self._lock:
            logs.append((datetime.now(), data))
        super().write(data)

    def flush(self):
        super().flush()

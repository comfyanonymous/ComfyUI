from collections import deque
from datetime import datetime
import io
import logging
import sys
import threading

logs = None
stdout_interceptor = None
stderr_interceptor = None


class LogInterceptor(io.TextIOWrapper):
    def __init__(self, stream,  *args, **kwargs):
        buffer = stream.buffer
        encoding = stream.encoding
        super().__init__(buffer, *args, **kwargs, encoding=encoding, line_buffering=stream.line_buffering)
        self._lock = threading.Lock()
        self._flush_callbacks = []
        self._logs_since_flush = []

    def write(self, data):
        entry = {"t": datetime.now().isoformat(), "m": data}
        with self._lock:
            self._logs_since_flush.append(entry)

            # Simple handling for cr to overwrite the last output if it isnt a full line
            # else logs just get full of progress messages
            if isinstance(data, str) and data.startswith("\r") and not logs[-1]["m"].endswith("\n"):
                logs.pop()
            logs.append(entry)
        super().write(data)

    def flush(self):
        super().flush()
        for cb in self._flush_callbacks:
            cb(self._logs_since_flush)
            self._logs_since_flush = []

    def on_flush(self, callback):
        self._flush_callbacks.append(callback)


def get_logs():
    return logs


def on_flush(callback):
    if stdout_interceptor is not None:
        stdout_interceptor.on_flush(callback)
    if stderr_interceptor is not None:
        stderr_interceptor.on_flush(callback)

def setup_logger(log_level: str = 'INFO', capacity: int = 300):
    global logs
    if logs:
        return

    # Override output streams and log to buffer
    logs = deque(maxlen=capacity)

    global stdout_interceptor
    global stderr_interceptor
    stdout_interceptor = sys.stdout = LogInterceptor(sys.stdout)
    stderr_interceptor = sys.stderr = LogInterceptor(sys.stderr)

    # Setup default global logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

import io
import logging
import sys
import threading
from collections import deque
from datetime import datetime

# initialize with sane defaults
logs = deque(maxlen=1000)
stdout_interceptor = sys.stdout
stderr_interceptor = sys.stderr

logger = logging.getLogger(__name__)

class LogInterceptor(io.TextIOWrapper):
    def __init__(self, stream, *args, **kwargs):
        buffer = stream.buffer
        encoding = stream.encoding
        # Use 'replace' error handling to avoid Unicode encoding errors on Windows
        super().__init__(buffer, *args, **kwargs, encoding=encoding, errors='replace', line_buffering=stream.line_buffering)
        self._lock = threading.Lock()
        self._flush_callbacks = []
        self._logs_since_flush = []

    def write(self, data):
        entry = {"t": datetime.now().isoformat(), "m": data}
        with self._lock:
            self._logs_since_flush.append(entry)

            # Simple handling for cr to overwrite the last output if it isnt a full line
            # else logs just get full of progress messages
            if isinstance(data, str) and data.startswith("\r") and len(logs) > 0 and not logs[-1]["m"].endswith("\n"):
                logs.pop()
            logs.append(entry)
        if not self.closed:
            try:
                super().write(data)
            except UnicodeEncodeError:
                # some random bs in custom nodes will trigger errors on Windows
                super().write(data.encode(self.encoding, errors='replace').decode(self.encoding))

    def flush(self):
        if not self.closed:
            super().flush()
        for cb in self._flush_callbacks:
            cb(self._logs_since_flush)
            self._logs_since_flush = []

    def on_flush(self, callback):
        self._flush_callbacks.append(callback)


def get_logs():
    return logs


def on_flush(callback):
    if stdout_interceptor is not None and hasattr(stdout_interceptor, "on_flush"):
        stdout_interceptor.on_flush(callback)
    if stderr_interceptor is not None and hasattr(stderr_interceptor, "on_flush"):
        stderr_interceptor.on_flush(callback)


class StackTraceLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        if not stack_info and level >= logging.ERROR and exc_info is None:
            # create a stack even when there is no exception
            stack_info = True
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel=stacklevel + 1)


def setup_logger(log_level: str = 'INFO', capacity: int = 300, use_stdout: bool = False):
    global logs
    if logs:
        return

    # workaround for google colab
    if not hasattr(sys.stdout, "buffer") or not hasattr(sys.stdout, "encoding") or not hasattr(sys.stdout, "line_buffering"):
        return

    # Override output streams and log to buffer
    logs = deque(maxlen=capacity)

    global stdout_interceptor
    global stderr_interceptor
    stdout_interceptor = sys.stdout = LogInterceptor(sys.stdout)
    stderr_interceptor = sys.stderr = LogInterceptor(sys.stderr)

    # Setup default global logger
    logging.setLoggerClass(StackTraceLogger)
    logger = logging.getLogger()
    logger.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    if use_stdout:
        # Only errors and critical to stderr
        stream_handler.addFilter(lambda record: not record.levelno < logging.ERROR)

        # Lesser to stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))
        stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
        logger.addHandler(stdout_handler)

    logger.addHandler(stream_handler)


STARTUP_WARNINGS = []


def log_startup_warning(msg):
    logger.warning(msg)
    STARTUP_WARNINGS.append(msg)


def print_startup_warnings():
    for s in STARTUP_WARNINGS:
        logger.warning(s)
    STARTUP_WARNINGS.clear()

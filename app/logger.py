import logging
from logging.handlers import MemoryHandler
from collections import deque

logs = None
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def get_logs():
    return "\n".join([formatter.format(x) for x in logs])


def setup_logger(log_level: str = 'INFO', capacity: int = 300):
    global logs
    if logs:
        return

    # Setup default global logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    # Create a memory handler with a deque as its buffer
    logs = deque(maxlen=capacity)
    memory_handler = MemoryHandler(capacity, flushLevel=logging.INFO)
    memory_handler.buffer = logs
    memory_handler.setFormatter(formatter)
    logger.addHandler(memory_handler)

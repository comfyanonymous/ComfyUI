import logging
from logging.handlers import MemoryHandler
from collections import deque
from rich import pretty
from rich.theme import Theme
from rich.console import Console
from rich.logging import RichHandler


logs = None
LOGFMT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)s] %(message)s"
formatter = logging.Formatter(LOGFMT)


def get_logs():
    return "\n".join([formatter.format(x) for x in logs])


def setup_logger(log_level: str = 'INFO', capacity: int = 300):
    global logs
    if logs:
        return

    # Setup default global logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    stream_handler = get_rich_hander(log_level)
    logger.addHandler(stream_handler)

    # Create a memory handler with a deque as its buffer
    logs = deque(maxlen=capacity)
    memory_handler = MemoryHandler(capacity, flushLevel=logging.INFO)
    memory_handler.buffer = logs
    memory_handler.setFormatter(formatter)
    logger.addHandler(memory_handler)


def get_rich_hander(log_level: str = "INFO") -> RichHandler:
    console = Console(
        log_time=True,
        log_time_format="%Y-%m-%d %H:%M:%S",
        theme=Theme({
            "log.time": "green",
            "inspect.value.border": "black",
            "traceback.border": "black",
            "traceback.border.syntax_error": "black",
        }),
    )

    pretty.install(console=console)
    handler = RichHandler(
        console=console,
        level=log_level,
        markup=False,
        show_time=True,
        show_level=True,
        show_path=True,
        rich_tracebacks=True,
        enable_link_path=False,
        omit_repeated_times=False,
        log_time_format="%Y-%m-%d %H:%M:%S",
    )
    return handler

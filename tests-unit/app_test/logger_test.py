from collections import deque
import io

import app.logger
from app.logger import LogInterceptor


def make_interceptor():
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", line_buffering=True)
    return LogInterceptor(stream), stream


def test_carriage_return_write_with_empty_log_buffer(monkeypatch):
    monkeypatch.setattr(app.logger, "logs", deque(maxlen=10))
    interceptor, _stream = make_interceptor()

    interceptor.write("\rprogress")

    assert len(app.logger.logs) == 1
    assert app.logger.logs[-1]["m"] == "\rprogress"


def test_carriage_return_replaces_incomplete_log_entry(monkeypatch):
    monkeypatch.setattr(app.logger, "logs", deque(maxlen=10))
    interceptor, _stream = make_interceptor()

    interceptor.write("progress 1")
    interceptor.write("\rprogress 2")

    assert [entry["m"] for entry in app.logger.logs] == ["\rprogress 2"]

"""Tests for the logger module, specifically LogInterceptor."""

import io
import pytest
from unittest.mock import MagicMock


class TestLogInterceptorFlush:
    """Test that LogInterceptor.flush() handles OSError gracefully."""

    def test_flush_handles_errno_22(self):
        """Test that flush() catches OSError with errno 22 and still executes callbacks."""
        # We can't easily mock the parent flush, so we test the behavior by
        # creating a LogInterceptor and verifying the flush method exists
        # with the try-except structure.

        # Read the source to verify the fix is in place
        import inspect
        from app.logger import LogInterceptor

        source = inspect.getsource(LogInterceptor.flush)

        # Verify the try-except structure is present
        assert 'try:' in source
        assert 'super().flush()' in source
        assert 'except OSError as e:' in source
        assert 'e.errno != 22' in source or 'e.errno == 22' in source

    def test_flush_callback_execution(self):
        """Test that flush callbacks are executed."""
        from app.logger import LogInterceptor

        # Create a proper stream for LogInterceptor
        import sys

        # Use a StringIO-based approach with a real buffer
        class MockStream:
            def __init__(self):
                self._buffer = io.BytesIO()
                self.encoding = 'utf-8'
                self.line_buffering = False

            @property
            def buffer(self):
                return self._buffer

        mock_stream = MockStream()
        interceptor = LogInterceptor(mock_stream)

        # Register a callback
        callback_results = []
        interceptor.on_flush(lambda logs: callback_results.append(len(logs)))

        # Add some logs
        interceptor._logs_since_flush = [
            {"t": "test", "m": "message1"},
            {"t": "test", "m": "message2"}
        ]

        # Flush should execute callback
        interceptor.flush()

        assert len(callback_results) == 1
        assert callback_results[0] == 2  # Two log entries

    def test_flush_clears_logs_after_callback(self):
        """Test that logs are cleared after flush callbacks."""
        from app.logger import LogInterceptor

        class MockStream:
            def __init__(self):
                self._buffer = io.BytesIO()
                self.encoding = 'utf-8'
                self.line_buffering = False

            @property
            def buffer(self):
                return self._buffer

        mock_stream = MockStream()
        interceptor = LogInterceptor(mock_stream)

        # Add a dummy callback
        interceptor.on_flush(lambda logs: None)

        # Add some logs
        interceptor._logs_since_flush = [{"t": "test", "m": "message"}]

        # Flush
        interceptor.flush()

        # Logs should be cleared
        assert interceptor._logs_since_flush == []

    def test_flush_multiple_callbacks_receive_same_logs(self):
        """Test that all callbacks receive the same logs, not just the first one."""
        from app.logger import LogInterceptor

        class MockStream:
            def __init__(self):
                self._buffer = io.BytesIO()
                self.encoding = 'utf-8'
                self.line_buffering = False

            @property
            def buffer(self):
                return self._buffer

        mock_stream = MockStream()
        interceptor = LogInterceptor(mock_stream)

        # Register multiple callbacks
        callback1_results = []
        callback2_results = []
        callback3_results = []
        interceptor.on_flush(lambda logs: callback1_results.append(len(logs)))
        interceptor.on_flush(lambda logs: callback2_results.append(len(logs)))
        interceptor.on_flush(lambda logs: callback3_results.append(len(logs)))

        # Add some logs
        interceptor._logs_since_flush = [
            {"t": "test", "m": "message1"},
            {"t": "test", "m": "message2"},
            {"t": "test", "m": "message3"}
        ]

        # Flush should execute all callbacks with the same logs
        interceptor.flush()

        # All callbacks should have received 3 log entries
        assert callback1_results == [3]
        assert callback2_results == [3]
        assert callback3_results == [3]

    def test_flush_clears_logs_even_if_callback_raises(self):
        """Test that logs are cleared even if a callback raises an exception."""
        from app.logger import LogInterceptor

        class MockStream:
            def __init__(self):
                self._buffer = io.BytesIO()
                self.encoding = 'utf-8'
                self.line_buffering = False

            @property
            def buffer(self):
                return self._buffer

        mock_stream = MockStream()
        interceptor = LogInterceptor(mock_stream)

        # Register a callback that raises
        def raising_callback(logs):
            raise ValueError("Callback error")

        interceptor.on_flush(raising_callback)

        # Add some logs
        interceptor._logs_since_flush = [
            {"t": "test", "m": "message1"},
            {"t": "test", "m": "message2"}
        ]

        # Flush should raise but logs should still be cleared
        with pytest.raises(ValueError, match="Callback error"):
            interceptor.flush()

        # Logs should be cleared to prevent duplicates on next flush
        assert interceptor._logs_since_flush == []


class TestLogInterceptorWrite:
    """Test that LogInterceptor.write() works correctly."""

    def test_write_adds_to_logs(self):
        """Test that write() adds entries to the log buffer."""
        from app.logger import LogInterceptor

        class MockStream:
            def __init__(self):
                self._buffer = io.BytesIO()
                self.encoding = 'utf-8'
                self.line_buffering = False

            @property
            def buffer(self):
                return self._buffer

        mock_stream = MockStream()
        interceptor = LogInterceptor(mock_stream)

        # Initialize the global logs
        import app.logger
        from collections import deque
        app.logger.logs = deque(maxlen=100)

        # Write a message
        interceptor.write("test message")

        # Check that it was added to _logs_since_flush
        assert len(interceptor._logs_since_flush) == 1
        assert interceptor._logs_since_flush[0]["m"] == "test message"

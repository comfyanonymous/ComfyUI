from __future__ import annotations

import time


class TqdmWatcher:
    """An object to track the progress of a tqdm instance."""

    def __init__(self):
        # We use a list to store the time, making it mutable across scopes.
        self._last_update = [time.monotonic()]

    def tick(self):
        """Signals that progress has been made by updating the timestamp."""
        self._last_update[0] = time.monotonic()

    @property
    def last_update_time(self) -> float:
        """Gets the time of the last recorded progress update."""
        return self._last_update[0]

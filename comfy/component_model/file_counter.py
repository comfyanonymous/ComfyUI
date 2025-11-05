import shutil
from contextlib import contextmanager
from pathlib import Path

import filelock


class ContextWrapper:
    """A wrapper to hold context manager values for entry and exit."""

    def __init__(self, value):
        self.value = value
        self.ctr = None

    def __int__(self):
        return self.value


class FileCounter:
    def __init__(self, path):
        self.path = Path(path)

    async def __aenter__(self):
        wrapper = ContextWrapper(self.get_and_increment())
        self._context_wrapper = wrapper
        return wrapper

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._context_wrapper.ctr = self.decrement_and_get()

    def __enter__(self):
        """Increment on entering the context and return a wrapper."""
        wrapper = ContextWrapper(self.get_and_increment())
        self._context_wrapper = wrapper
        return wrapper

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Decrement on exiting the context and update the wrapper."""
        self._context_wrapper.ctr = self.decrement_and_get()

    def _read_and_write(self, operation):
        lock = filelock.FileLock(f"{self.path}.lock")
        with lock:
            count = 0
            try:
                with open(self.path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        count = int(content)
            except FileNotFoundError:
                # File doesn't exist, will be created with initial value.
                pass
            except ValueError:
                # File is corrupt or empty, treat as 0 and overwrite.
                pass

            original_count = count
            new_count = operation(count)

            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, 'w') as f:
                f.write(str(new_count))

            return original_count, new_count

    def get_and_increment(self):
        """Atomically reads the current value, increments it, and returns the original value."""
        original_count, _ = self._read_and_write(lambda x: x + 1)
        return original_count

    def decrement_and_get(self):
        """Atomically decrements the value and returns the new value."""
        _, new_count = self._read_and_write(lambda x: x - 1)
        return new_count


@contextmanager
def cleanup_temp():
    from ..cli_args import args
    from ..cmd import folder_paths
    tmp_dir = Path(args.temp_directory or folder_paths.get_temp_directory())
    counter_path = tmp_dir / "counter.txt"
    fc_i = -1
    try:
        with FileCounter(counter_path) as fc:
            yield
        fc_i = fc.ctr
    finally:
        if fc_i == 0 and tmp_dir.is_dir():
            shutil.rmtree(tmp_dir, ignore_errors=True)

import pickle
import logging

# Allowlist of safe modules for unpickling model checkpoints
# These are the minimum required for loading PyTorch model weights
ALLOWED_MODULE_PREFIXES = frozenset({
    'torch',
    'numpy',
    'collections',
    '_codecs',
    'codecs',
})

# Blocklist of dangerous names that should never be unpickled
BLOCKED_NAMES = frozenset({
    'eval', 'exec', 'compile', 'open', 'input', '__import__',
    'getattr', 'setattr', 'delattr', 'globals', 'locals', 'vars',
    'system', 'popen', 'spawn', 'fork', 'execv', 'execve',
    'subprocess', 'Popen', 'call', 'check_output', 'run',
})

class Empty:
    """Placeholder class for blocked types during unpickling."""
    pass

class RestrictedUnpickler(pickle.Unpickler):
    """
    A restricted unpickler that only allows safe modules to be loaded.

    This helps prevent arbitrary code execution when loading untrusted
    checkpoint files. Only modules needed for loading model weights
    (torch, numpy, collections) are allowed.
    """

    def find_class(self, module: str, name: str):
        # Block dangerous function/class names regardless of module
        if name in BLOCKED_NAMES:
            logging.warning(f"Blocked unpickling of dangerous name: {module}.{name}")
            return Empty

        # Block pytorch_lightning (known to cause issues)
        if module.startswith("pytorch_lightning"):
            return Empty

        # Allow only safe module prefixes
        if any(module.startswith(prefix) for prefix in ALLOWED_MODULE_PREFIXES):
            return super().find_class(module, name)

        # Block everything else
        logging.warning(f"Blocked unpickling of untrusted class: {module}.{name}")
        return Empty

# For backwards compatibility, provide a restricted load function
def load(file, **kwargs):
    """Load a pickle file using the restricted unpickler."""
    return RestrictedUnpickler(file, **kwargs).load()

# Alias for compatibility
Unpickler = RestrictedUnpickler

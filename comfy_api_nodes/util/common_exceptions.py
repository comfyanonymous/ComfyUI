class NetworkError(Exception):
    """Base exception for network-related errors with diagnostic information."""


class LocalNetworkError(NetworkError):
    """Exception raised when local network connectivity issues are detected."""


class ApiServerError(NetworkError):
    """Exception raised when the API server is unreachable but internet is working."""


class ProcessingInterrupted(Exception):
    """Operation was interrupted by user/runtime via processing_interrupted()."""

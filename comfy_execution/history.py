class HistoryStorageError(RuntimeError):
    """The requested history operation did not commit.

    Only the closed reason vocabulary is safe for logs, never the exception text.
    """

    REASONS = frozenset(
        {
            "storage_error",
            "unsupported_json",
            "invalid_record",
            "record_too_large",
            "serialization_failed",
            "unsupported_format",
            "invalid_json",
            "database_error",
            "database_locked",
            "disk_full",
            "disk_io",
            "unable_to_open",
            "database_corrupt",
            "permission_denied",
            "file_locked",
            "read_only",
        }
    )

    def __init__(self, message, record=None, *, reason="storage_error"):
        super().__init__(message)
        self.record = record
        self.reason = (
            reason
            if type(reason) is str and reason in self.REASONS
            else "storage_error"
        )

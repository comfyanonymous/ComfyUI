class HistoryStorageError(RuntimeError):
    """The requested history operation did not commit."""

    def __init__(self, message, record=None):
        super().__init__(message)
        self.record = record

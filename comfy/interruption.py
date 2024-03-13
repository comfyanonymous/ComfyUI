import threading

_interrupt_processing_mutex = threading.RLock()
_interrupt_processing = False


class InterruptProcessingException(Exception):
    pass


def interrupt_current_processing(value=True):
    global _interrupt_processing
    global _interrupt_processing_mutex
    with _interrupt_processing_mutex:
        _interrupt_processing = value


def processing_interrupted():
    global _interrupt_processing
    global _interrupt_processing_mutex
    with _interrupt_processing_mutex:
        return _interrupt_processing


def throw_exception_if_processing_interrupted():
    global _interrupt_processing
    global _interrupt_processing_mutex
    with _interrupt_processing_mutex:
        if _interrupt_processing:
            _interrupt_processing = False
            raise InterruptProcessingException()

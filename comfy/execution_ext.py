import importlib


def import_exception_class(fqn: str):
    """
    Imports an exception class from its fully qualified name.
    Example: 'torch.cuda.OutOfMemoryError' -> torch.cuda.OutOfMemoryError

    Args:
        fqn: Fully qualified name of the exception class

    Returns:
        The exception class

    Raises:
        ValueError: If the class cannot be imported or is not a subclass of Exception
    """
    try:
        module_path, class_name = fqn.rsplit('.', 1)
        module = importlib.import_module(module_path)
        exc_class = getattr(module, class_name)

        if not isinstance(exc_class, type) or not issubclass(exc_class, Exception):
            raise ValueError(f"{fqn} is not an exception class")

        return exc_class
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not import exception class {fqn}: {str(e)}")


def should_panic_on_exception(exc: Exception, panic_classes: list[str]) -> bool:
    """
    Checks if the given exception matches any of the specified panic classes.

    Args:
        exc: The exception to check
        panic_classes: List of fully qualified exception class names

    Returns:
        True if the exception is an instance of one of the specified classes
    """
    # Handle comma-separated lists (from config files or env vars)
    expanded_classes = []
    for class_spec in panic_classes:
        expanded_classes.extend(name.strip() for name in class_spec.split(','))

    # Import all exception classes
    try:
        exception_types = [import_exception_class(name)
                           for name in expanded_classes if name]
    except ValueError as e:
        print(f"Warning: {str(e)}")
        return False

    # Check if exception matches any of the specified types
    return any(isinstance(exc, exc_type) for exc_type in exception_types)

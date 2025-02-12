def merge_json_recursive(base, update):
    """Recursively merge two JSON-like objects.
    - Dictionaries are merged recursively
    - Lists are concatenated
    - Other types are overwritten by the update value

    Args:
        base: Base JSON-like object
        update: Update JSON-like object to merge into base

    Returns:
        Merged JSON-like object
    """
    if not isinstance(base, dict) or not isinstance(update, dict):
        if isinstance(base, list) and isinstance(update, list):
            return base + update
        return update

    merged = base.copy()
    for key, value in update.items():
        if key in merged:
            merged[key] = merge_json_recursive(merged[key], value)
        else:
            merged[key] = value

    return merged

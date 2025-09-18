from decimal import Decimal


def is_scalar(v):
    if v is None:
        return True
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, Decimal, str)):
        return True
    return False


def project_kv(key: str, value):
    """
    Turn a metadata key/value into typed projection rows.
    Returns list[dict] with keys:
      key, ordinal, and one of val_str / val_num / val_bool / val_json (others None)
    """
    rows: list[dict] = []

    def _null_row(ordinal: int) -> dict:
        return {
            "key": key, "ordinal": ordinal,
            "val_str": None, "val_num": None, "val_bool": None, "val_json": None
        }

    if value is None:
        rows.append(_null_row(0))
        return rows

    if is_scalar(value):
        if isinstance(value, bool):
            rows.append({"key": key, "ordinal": 0, "val_bool": bool(value)})
        elif isinstance(value, (int, float, Decimal)):
            num = value if isinstance(value, Decimal) else Decimal(str(value))
            rows.append({"key": key, "ordinal": 0, "val_num": num})
        elif isinstance(value, str):
            rows.append({"key": key, "ordinal": 0, "val_str": value})
        else:
            rows.append({"key": key, "ordinal": 0, "val_json": value})
        return rows

    if isinstance(value, list):
        if all(is_scalar(x) for x in value):
            for i, x in enumerate(value):
                if x is None:
                    rows.append(_null_row(i))
                elif isinstance(x, bool):
                    rows.append({"key": key, "ordinal": i, "val_bool": bool(x)})
                elif isinstance(x, (int, float, Decimal)):
                    num = x if isinstance(x, Decimal) else Decimal(str(x))
                    rows.append({"key": key, "ordinal": i, "val_num": num})
                elif isinstance(x, str):
                    rows.append({"key": key, "ordinal": i, "val_str": x})
                else:
                    rows.append({"key": key, "ordinal": i, "val_json": x})
            return rows
        for i, x in enumerate(value):
            rows.append({"key": key, "ordinal": i, "val_json": x})
        return rows

    rows.append({"key": key, "ordinal": 0, "val_json": value})
    return rows

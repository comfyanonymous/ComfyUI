from app.assets.database.queries.records import (
    create_content,
    create_record,
    delete_record,
    get_record_by_id,
    list_records_page,
    mark_content_missing,
    rename_record,
    unset_content_missing,
)

__all__ = [
    "create_content",
    "create_record",
    "delete_record",
    "get_record_by_id",
    "list_records_page",
    "mark_content_missing",
    "rename_record",
    "unset_content_missing",
]


def __getattr__(name: str):
    from importlib import import_module

    for module_name in ("asset", "asset_reference", "tags"):
        module = import_module(f"app.assets.database.queries.{module_name}")
        candidate = getattr(module, name, None)
        if candidate is not None:
            return candidate
    raise AttributeError(name)

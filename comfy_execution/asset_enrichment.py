import logging
import os


def enrich_output_with_assets(output_ui: dict) -> dict:
    from comfy.cli_args import args
    if not args.enable_assets:
        return output_ui

    import folder_paths
    from app.assets.database.queries.records import get_record_by_path_or_none
    from app.database.db import create_session

    enriched = {}
    for key, entries in output_ui.items():
        if not isinstance(entries, list):
            enriched[key] = entries
            continue
        new_entries = []
        for entry in entries:
            if not isinstance(entry, dict) or "filename" not in entry or "type" not in entry:
                new_entries.append(entry)
                continue
            try:
                base = folder_paths.get_directory_by_type(entry["type"])
                if base is None:
                    new_entries.append(entry)
                    continue
                base_abs = os.path.abspath(base)
                abs_path = os.path.abspath(os.path.join(base_abs, entry.get("subfolder") or "", entry["filename"]))
                try:
                    if os.path.commonpath([base_abs, abs_path]) != base_abs:
                        raise ValueError("escapes base")
                except ValueError:
                    logging.warning("Asset enrichment skipped (path escapes base): %s", entry.get("filename"))
                    new_entries.append(entry)
                    continue
                if not os.path.isfile(abs_path):
                    new_entries.append(entry)
                    continue

                with create_session() as session:
                    record = get_record_by_path_or_none(session, abs_path)
                if record is not None:
                    entry = dict(entry)
                    entry["id"] = record.id
            except Exception:
                logging.warning("Failed to enrich output entry with asset id: %s", entry.get("filename"), exc_info=True)
            new_entries.append(entry)
        enriched[key] = new_entries
    return enriched

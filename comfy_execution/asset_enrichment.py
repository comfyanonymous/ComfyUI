import copy
import logging
import os


def _resolve_output_path(entry: dict) -> str | None:
    """Resolve an output entry to an absolute, in-base, on-disk file path.

    Returns ``None`` (skip, no registration) when the type is unknown, the
    resolved path escapes its base directory, or the file does not exist.
    """
    import folder_paths

    base = folder_paths.get_directory_by_type(entry["type"])
    if base is None:
        return None
    base_abs = os.path.abspath(base)
    abs_path = os.path.abspath(os.path.join(base_abs, entry.get("subfolder") or "", entry["filename"]))
    try:
        if os.path.commonpath([base_abs, abs_path]) != base_abs:
            return None
    except ValueError:
        return None
    if not os.path.isfile(abs_path):
        return None
    return abs_path


def _enrich_in_place(output_ui: dict, job_id, register) -> None:
    """Register each output file entry and attach the returned id, in place.

    The caller always passes a *copy* it owns, so mutating here never touches
    the argument the adapter received. S10.6: producers that write the same
    output path are not coalesced (unsupported) - every entry is registered
    independently. S10.4: a registration failure is logged, leaves the entry
    without an id, and never propagates.
    """
    for entries in output_ui.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "filename" not in entry or "type" not in entry:
                continue
            try:
                abs_path = _resolve_output_path(entry)
                if abs_path is None:
                    continue
                result = register(abs_path, job_id=job_id)
                if result is not None:
                    entry["id"] = result.id
            except Exception:
                logging.warning("Asset registration failed for output: %s", entry.get("filename"), exc_info=True)


def _strip_ids(output_ui: dict) -> None:
    """Drop any pre-existing ``id`` keys so a replay never reuses a stale id."""
    for entries in output_ui.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                entry.pop("id", None)


def register_executed_outputs(output_ui: dict, job_id) -> dict:
    """Register a freshly-executed node's outputs and return an enriched COPY.

    Pure: deep-copies ``output_ui`` and enriches the copy, so the raw dict the
    caller stores in the cache stays id-free (S10.5). Gated on
    ``args.enable_assets``; when disabled the copy is returned unenriched.
    """
    from comfy.cli_args import args

    enriched = copy.deepcopy(output_ui)
    if not args.enable_assets:
        return enriched
    from app.assets.services.ingest import register_executed_output

    _enrich_in_place(enriched, job_id, register_executed_output)
    return enriched


def register_cached_outputs(ui_wrapper, job_id):
    """Register a replayed cached node's outputs and return an enriched COPY.

    ``ui_wrapper`` is the cache UI wrapper ``{"meta": ..., "output": output_ui}``.
    Pure: deep-copies the whole wrapper, strips any legacy ids from the copy,
    then enriches ``copy["output"]`` as a cached-replay delivery. The argument
    (``cached.ui``) is never mutated (S10.5). Returns ``None`` for a ``None``
    wrapper; gated on ``args.enable_assets``.
    """
    if ui_wrapper is None:
        return None

    enriched = copy.deepcopy(ui_wrapper)
    output_ui = enriched.get("output")
    if not isinstance(output_ui, dict):
        return enriched
    _strip_ids(output_ui)

    from comfy.cli_args import args

    if not args.enable_assets:
        return enriched
    from app.assets.services.ingest import register_cached_output

    _enrich_in_place(output_ui, job_id, register_cached_output)
    return enriched


def emit_cached_output(server, node_id, display_node_id, cached, prompt_id, ui_outputs) -> None:
    """Register + deliver a cached node's UI at emission time.

    Registers the replay (even with no connected client), publishes the enriched
    COPY to ``ui_outputs`` (never ``cached.ui`` itself), and only then returns
    early when no client is connected - so the client send is the sole part that
    depends on ``client_id``. Double-emission guard (D6d): a node already present
    in ``ui_outputs`` is skipped, giving exactly one delivery record per cached
    node per prompt.
    """
    if node_id in ui_outputs:
        return
    enriched = register_cached_outputs(cached.ui, prompt_id)
    if enriched is not None:
        ui_outputs[node_id] = enriched
    if server.client_id is None:
        return
    output = enriched.get("output") if enriched is not None else None
    server.send_sync(
        "executed",
        {"node": node_id, "display_node": display_node_id, "output": output, "prompt_id": prompt_id},
        server.client_id,
    )

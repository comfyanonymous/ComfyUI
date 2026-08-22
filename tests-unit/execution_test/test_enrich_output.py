import importlib
import os
import sys
import types
from unittest.mock import MagicMock, patch


_DEFAULT_BASE = os.path.join(__import__("tempfile").gettempdir(), "asset-enrichment-test-base")


def _record(record_id: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(id=record_id)


def _call(
    output_ui: dict,
    *,
    enable_assets: bool = True,
    file_exists: bool = True,
    record: types.SimpleNamespace | None = None,
    directory: str | None = _DEFAULT_BASE,
    lookup: MagicMock | None = None,
) -> tuple[dict, MagicMock]:
    lookup = lookup or MagicMock(return_value=record)
    session = MagicMock()
    session_context = MagicMock()
    session_context.__enter__.return_value = session
    session_context.__exit__.return_value = False
    mocked_modules = {
        "comfy.cli_args": MagicMock(args=types.SimpleNamespace(enable_assets=enable_assets)),
        "folder_paths": MagicMock(get_directory_by_type=MagicMock(return_value=directory)),
        "app.assets.database.queries.records": MagicMock(
            get_record_by_path_or_none=lookup
        ),
        "app.assets.services.ingest": MagicMock(
            register_file_in_place=MagicMock(side_effect=AssertionError("must not register")),
            DependencyMissingError=type("DependencyMissingError", (Exception,), {}),
        ),
        "app.database.db": MagicMock(create_session=MagicMock(return_value=session_context)),
    }
    with patch.dict(sys.modules, mocked_modules), patch("os.path.isfile", return_value=file_exists):
        import comfy_execution.asset_enrichment as module

        importlib.reload(module)
        return module.enrich_output_with_assets(output_ui), lookup


def test_disabled_returns_original_output() -> None:
    output = {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}

    result, _ = _call(output, enable_assets=False)

    assert result is output


def test_non_list_value_is_passed_through() -> None:
    result, _ = _call({"text": "hello"})

    assert result["text"] == "hello"


def test_entry_without_file_fields_is_unchanged() -> None:
    result, _ = _call({"latent": [{"subfolder": "", "type": "output"}]})

    assert "id" not in result["latent"][0]


def test_missing_file_is_unchanged() -> None:
    result, lookup = _call(
        {"images": [{"filename": "missing.png", "subfolder": "", "type": "output"}]},
        file_exists=False,
    )

    assert "id" not in result["images"][0]
    lookup.assert_not_called()


def test_registered_output_receives_its_existing_record_id() -> None:
    output = {"images": [{"filename": "new.png", "subfolder": "", "type": "output"}]}

    result, lookup = _call(output, record=_record("existing-record"))

    assert result["images"][0]["id"] == "existing-record"
    lookup.assert_called_once()


def test_unregistered_output_is_unchanged() -> None:
    output = {"images": [{"filename": "pending.png", "subfolder": "", "type": "output"}]}

    result, _ = _call(output)

    assert "id" not in result["images"][0]


def test_original_entry_is_not_mutated() -> None:
    entry = {"filename": "a.png", "subfolder": "", "type": "output"}

    _call({"images": [entry]}, record=_record("record"))

    assert "id" not in entry


def test_lookup_error_does_not_block_a_sibling_entry() -> None:
    lookup = MagicMock(side_effect=[RuntimeError("boom"), _record("good-record")])
    output = {
        "images": [
            {"filename": "bad.png", "subfolder": "", "type": "output"},
            {"filename": "good.png", "subfolder": "", "type": "output"},
        ]
    }

    result, _ = _call(output, lookup=lookup)

    assert "id" not in result["images"][0]
    assert result["images"][1]["id"] == "good-record"


def test_multiple_output_keys_are_looked_up() -> None:
    output = {
        "images": [{"filename": "a.png", "subfolder": "", "type": "output"}],
        "videos": [{"filename": "b.mp4", "subfolder": "", "type": "output"}],
    }

    result, lookup = _call(output, record=_record("record"))

    assert result["images"][0]["id"] == "record"
    assert result["videos"][0]["id"] == "record"
    assert lookup.call_count == 2


def test_path_outside_its_base_is_ignored() -> None:
    output = {"images": [{"filename": "passwd", "subfolder": "../../etc", "type": "output"}]}

    result, lookup = _call(output, record=_record("record"))

    assert "id" not in result["images"][0]
    lookup.assert_not_called()

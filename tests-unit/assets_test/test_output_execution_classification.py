from pathlib import Path

from app.assets.services.output_registration import (
    OutputExecution,
    OutputFileRegistration,
    collect_output_registrations,
)


def test_static_output_from_executed_node_is_executed(tmp_path: Path) -> None:
    directories = {"output": str(tmp_path / "output")}
    history_result: dict[str, object] = {
        "outputs": {
            "static-node": {
                "images": [
                    {
                        "filename": "image.png",
                        "subfolder": "nested",
                        "type": "output",
                    }
                ]
            }
        }
    }

    registrations = collect_output_registrations(
        history_result, {"static-node"}, directories.get
    )

    assert registrations == (
        OutputFileRegistration(
            path=str(tmp_path / "output" / "nested" / "image.png"),
            execution=OutputExecution.EXECUTED,
        ),
    )


def test_static_output_from_unexecuted_node_is_cached(tmp_path: Path) -> None:
    directories = {"output": str(tmp_path / "output")}
    history_result: dict[str, object] = {
        "outputs": {
            "static-node": {
                "images": [
                    {"filename": "image.png", "subfolder": "", "type": "output"}
                ]
            }
        }
    }

    registrations = collect_output_registrations(history_result, set(), directories.get)

    assert registrations == (
        OutputFileRegistration(
            path=str(tmp_path / "output" / "image.png"),
            execution=OutputExecution.CACHED,
        ),
    )


def test_runtime_created_node_absent_from_executed_ids_is_cached(
    tmp_path: Path,
) -> None:
    directories = {"output": str(tmp_path / "output")}
    history_result: dict[str, object] = {
        "outputs": {
            "expanded-node.7": {
                "images": [
                    {
                        "filename": "expanded.png",
                        "subfolder": "dynamic",
                        "type": "output",
                    }
                ]
            }
        }
    }

    registrations = collect_output_registrations(
        history_result, {"wrapper-node"}, directories.get
    )

    assert registrations == (
        OutputFileRegistration(
            path=str(tmp_path / "output" / "dynamic" / "expanded.png"),
            execution=OutputExecution.CACHED,
        ),
    )


def test_executed_producer_dominates_cached_producer_for_same_path(
    tmp_path: Path,
) -> None:
    directories = {"output": str(tmp_path / "output")}
    shared_item = {"filename": "shared.png", "subfolder": "", "type": "output"}
    history_result: dict[str, object] = {
        "outputs": {
            "executed-node": {"images": [shared_item]},
            "cached-node": {"images": [shared_item]},
        }
    }

    registrations = collect_output_registrations(
        history_result, {"executed-node"}, directories.get
    )

    assert registrations == (
        OutputFileRegistration(
            path=str(tmp_path / "output" / "shared.png"),
            execution=OutputExecution.EXECUTED,
        ),
    )


def test_same_filename_under_distinct_roots_stays_distinct(tmp_path: Path) -> None:
    directories = {
        "output": str(tmp_path / "output"),
        "temp": str(tmp_path / "temp"),
    }
    history_result: dict[str, object] = {
        "outputs": {
            "output-node": {
                "images": [
                    {"filename": "same.png", "subfolder": "", "type": "output"}
                ]
            },
            "temp-node": {
                "images": [
                    {"filename": "same.png", "subfolder": "", "type": "temp"}
                ]
            },
        }
    }

    registrations = collect_output_registrations(history_result, set(), directories.get)

    assert registrations == (
        OutputFileRegistration(
            path=str(tmp_path / "output" / "same.png"),
            execution=OutputExecution.CACHED,
        ),
        OutputFileRegistration(
            path=str(tmp_path / "temp" / "same.png"),
            execution=OutputExecution.CACHED,
        ),
    )


def test_traversal_and_unknown_output_type_are_skipped(tmp_path: Path) -> None:
    directories = {"output": str(tmp_path / "output")}
    history_result: dict[str, object] = {
        "outputs": {
            "node": {
                "images": [
                    {
                        "filename": "escape.png",
                        "subfolder": "..",
                        "type": "output",
                    },
                    {
                        "filename": "unknown.png",
                        "subfolder": "",
                        "type": "unknown",
                    },
                ]
            }
        }
    }

    registrations = collect_output_registrations(
        history_result, {"node"}, directories.get
    )

    assert registrations == ()


def test_symlinked_and_canonical_roots_are_not_coalesced(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    symlink_root = tmp_path / "symlink"
    symlink_root.symlink_to(canonical_root, target_is_directory=True)
    directories = {
        "output": str(canonical_root),
        "temp": str(symlink_root),
    }
    history_result: dict[str, object] = {
        "outputs": {
            "node": {
                "images": [
                    {"filename": "same.png", "subfolder": "", "type": "output"},
                    {"filename": "same.png", "subfolder": "", "type": "temp"},
                ]
            }
        }
    }

    registrations = collect_output_registrations(history_result, set(), directories.get)

    assert registrations == (
        OutputFileRegistration(
            path=str(canonical_root / "same.png"),
            execution=OutputExecution.CACHED,
        ),
        OutputFileRegistration(
            path=str(symlink_root / "same.png"),
            execution=OutputExecution.CACHED,
        ),
    )


def test_executed_node_reemitting_existing_locator_is_classified_executed(
    tmp_path: Path,
) -> None:
    directories = {"output": str(tmp_path / "output")}
    existing_item = {
        "filename": "existing.png",
        "subfolder": "",
        "type": "output",
    }
    history_result: dict[str, object] = {
        "outputs": {
            "cached-delivery": {"images": [existing_item]},
            "executed-producer": {"images": [existing_item]},
        }
    }

    registrations = collect_output_registrations(
        history_result, {"executed-producer"}, directories.get
    )

    assert registrations == (
        OutputFileRegistration(
            path=str(tmp_path / "output" / "existing.png"),
            execution=OutputExecution.EXECUTED,
        ),
    )

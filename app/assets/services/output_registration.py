import enum
import os
from collections.abc import Callable, Mapping, Set as AbstractSet
from dataclasses import dataclass
from typing import TypeGuard


class OutputExecution(enum.Enum):
    EXECUTED = enum.auto()
    CACHED = enum.auto()


@dataclass(frozen=True, slots=True)
class OutputFileRegistration:
    path: str
    execution: OutputExecution


def _is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def collect_output_registrations(
    history_result: Mapping[str, object],
    executed_node_ids: AbstractSet[str],
    directory_by_type: Callable[[str], str | None],
) -> tuple[OutputFileRegistration, ...]:
    """Collect output paths and classify them by their producer execution state."""
    outputs = history_result.get("outputs")
    if not _is_object_mapping(outputs):
        return ()

    executed_by_path: dict[str, bool] = {}
    for node_id, node_output in outputs.items():
        if not isinstance(node_id, str) or not _is_object_mapping(node_output):
            continue
        node_was_executed = node_id in executed_node_ids
        for output_items in node_output.values():
            if not _is_object_list(output_items):
                continue
            for file_info in output_items:
                if not _is_object_mapping(file_info):
                    continue
                output_type = file_info.get("type")
                filename = file_info.get("filename")
                subfolder = file_info.get("subfolder", "")
                if (
                    not isinstance(output_type, str)
                    or not isinstance(filename, str)
                    or not filename
                    or not isinstance(subfolder, str)
                ):
                    continue
                base_directory = directory_by_type(output_type)
                if base_directory is None:
                    continue
                absolute_base = os.path.abspath(base_directory)
                absolute_path = os.path.abspath(
                    os.path.join(base_directory, subfolder, filename)
                )
                if (
                    absolute_path != absolute_base
                    and not absolute_path.startswith(absolute_base + os.sep)
                ):
                    continue
                executed_by_path[absolute_path] = (
                    executed_by_path.get(absolute_path, False) or node_was_executed
                )

    return tuple(
        OutputFileRegistration(
            path=path,
            execution=(
                OutputExecution.EXECUTED
                if producer_was_executed
                else OutputExecution.CACHED
            ),
        )
        for path, producer_was_executed in executed_by_path.items()
    )

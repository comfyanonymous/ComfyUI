"""Semantics step 1: re-derive the reference state that comes from the path.

Three columns are pure functions of a file's location and whether it is there:
``loader_path``, the backend tags a path implies, and ``is_missing``. Every one
of them was computed once, when the reference was first written, by whatever
rules were current that day. ``loader_path`` in particular was added to existing
databases as a NULL column and no path has ever filled it in, so references
older than that column serve a null loader path forever.

This step recomputes those three from the filesystem as it stands now. It reads
no file contents: ``verify_file_unchanged`` answers whether a file still matches
what the database recorded, and the only thing that answer is used for is
deciding *not* to touch content-derived state. Nothing here re-hashes, so a
library of untouched 500GB models costs one ``stat`` each.

What the step will not touch:

- ``hash`` and ``size_bytes`` -- content facts, unrecoverable without reading
  the file. A file that has changed underneath its row is handed to the existing
  ``needs_verify`` path instead, exactly as the scanner would flag it.
- Anything a person chose: manual and upload-origin tags, ``user_metadata``,
  ``preview_id``, ``deleted_at``, ``job_id``, ``name``.
- References whose file is gone, or whose path is not under any root this
  install currently knows about. Retiring the former is the scanner's job under
  its own rules; the latter cannot be classified at all, and a misconfigured
  ``extra_model_paths.yaml`` must not be able to strip tags off assets that are
  merely out of view.
"""

import logging
import os
from dataclasses import dataclass

from app.assets.database.queries import (
    bulk_update_is_missing,
    bulk_update_needs_verify,
)
from app.assets.database.queries.semantics import (
    AUTOMATIC_TAG_ORIGIN,
    DerivedStateRow,
    bulk_add_automatic_tags,
    bulk_remove_automatic_tags,
    bulk_set_loader_paths,
    get_file_backed_references_page,
    get_tags_by_reference,
)
from app.assets.helpers import normalize_tags
from app.assets.semantics.step import InterruptCheck, SemanticsStepInterrupted
from app.assets.services.file_utils import verify_file_unchanged
from app.assets.services.path_utils import (
    compute_loader_path,
    get_path_derived_tag_vocabulary,
    get_path_derived_tags_from_path,
)
from app.database.db import create_session

# Bounds each transaction. Small enough that an interrupt loses little work,
# large enough that the walk is not dominated by session setup.
_BATCH_SIZE = 500


@dataclass
class ReprojectionSummary:
    """What a reprojection pass did, for the log line."""

    scanned: int = 0
    unchanged_files: int = 0
    changed_files: int = 0
    absent_files: int = 0
    unclassified_paths: int = 0
    loader_paths_rewritten: int = 0
    tags_added: int = 0
    tags_removed: int = 0
    missing_flags_cleared: int = 0
    verify_flags_set: int = 0

    def __str__(self) -> str:
        return (
            f"scanned={self.scanned} unchanged={self.unchanged_files} "
            f"changed={self.changed_files} absent={self.absent_files} "
            f"unclassified={self.unclassified_paths} "
            f"loader_paths={self.loader_paths_rewritten} "
            f"tags+{self.tags_added}/-{self.tags_removed} "
            f"unflagged_missing={self.missing_flags_cleared} "
            f"flagged_verify={self.verify_flags_set}"
        )


def reproject_derived_state(
    interrupt_check: InterruptCheck | None = None,
) -> ReprojectionSummary:
    """Re-derive path-derived state for every file-backed reference.

    Walks the reference table in keyset-paginated batches, committing each one,
    so a kill mid-walk leaves committed batches reprojected and the rest
    untouched -- both states the step handles on its next run.
    """
    summary = ReprojectionSummary()
    vocabulary = get_path_derived_tag_vocabulary()
    after_id: str | None = None

    while True:
        if interrupt_check is not None and interrupt_check():
            raise SemanticsStepInterrupted(
                f"interrupted after {summary.scanned} references"
            )

        with create_session() as session:
            rows = get_file_backed_references_page(
                session, after_id=after_id, limit=_BATCH_SIZE
            )
            if not rows:
                return summary
            after_id = rows[-1].reference_id
            _reproject_batch(session, rows, vocabulary, summary)
            session.commit()


def _reproject_batch(
    session,
    rows: list[DerivedStateRow],
    vocabulary: set[str],
    summary: ReprojectionSummary,
) -> None:
    stored_tags = get_tags_by_reference(session, [row.reference_id for row in rows])

    loader_paths: dict[str, str | None] = {}
    tags_to_add: list[tuple[str, str]] = []
    tags_to_remove: list[tuple[str, str]] = []
    clear_missing: list[str] = []
    set_needs_verify: list[str] = []

    for row in rows:
        summary.scanned += 1
        present, unchanged = _classify_file(row)

        if not present:
            # The scanner owns retiring these, under its own missing semantics.
            summary.absent_files += 1
            continue

        if unchanged:
            summary.unchanged_files += 1
        else:
            # The file moved on from what the row records, so its hash, size and
            # mime no longer describe it. Re-deriving those means reading the
            # file, which this step does not do; flag it for the path that does.
            summary.changed_files += 1
            if not row.needs_verify:
                set_needs_verify.append(row.reference_id)

        try:
            derived_tags = set(
                normalize_tags(get_path_derived_tags_from_path(row.file_path))
            )
        except ValueError:
            summary.unclassified_paths += 1
            continue

        loader_path = compute_loader_path(row.file_path)
        if loader_path != row.loader_path:
            loader_paths[row.reference_id] = loader_path
            summary.loader_paths_rewritten += 1

        current_tags = stored_tags.get(row.reference_id, {})
        for tag_name in sorted(derived_tags - set(current_tags)):
            tags_to_add.append((row.reference_id, tag_name))
        for tag_name, origin in sorted(current_tags.items()):
            if (
                origin == AUTOMATIC_TAG_ORIGIN
                and tag_name in vocabulary
                and tag_name not in derived_tags
            ):
                tags_to_remove.append((row.reference_id, tag_name))

        if row.is_missing:
            # The file is right there. Nothing else un-flags this: the scanner
            # excludes already-missing references from its temp reconciliation.
            clear_missing.append(row.reference_id)

    bulk_set_loader_paths(session, loader_paths)
    bulk_add_automatic_tags(session, tags_to_add)
    bulk_remove_automatic_tags(session, tags_to_remove)
    bulk_update_is_missing(session, clear_missing, value=False)
    bulk_update_needs_verify(session, set_needs_verify, value=True)

    summary.tags_added += len(tags_to_add)
    summary.tags_removed += len(tags_to_remove)
    summary.missing_flags_cleared += len(clear_missing)
    summary.verify_flags_set += len(set_needs_verify)


def _classify_file(row: DerivedStateRow) -> tuple[bool, bool]:
    """Return (file is present, file still matches what the row recorded).

    Mirrors the scanner's stat handling so the two agree about what counts as a
    present file: a permission error means the file is there but unreadable, any
    other OS error means treat it as gone.
    """
    try:
        stat_result = os.stat(row.file_path, follow_symlinks=True)
    except FileNotFoundError:
        return False, False
    except PermissionError:
        logging.debug("Permission denied accessing %s", row.file_path)
        return True, False
    except OSError as error:
        logging.debug("OSError checking %s: %s", row.file_path, error)
        return False, False

    return True, verify_file_unchanged(
        mtime_db=row.mtime_ns,
        size_db=row.size_bytes,
        stat_result=stat_result,
    )

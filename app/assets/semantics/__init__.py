"""Reset steps that bring stale asset rows forward to the current semantics.

Alembic migrates the *shape* of the assets tables. Nothing migrates their
*meaning*: a row whose columns are structurally current can still hold values
that a superseded rule computed, and the steady-state scan will not repair one,
because it only ever looks at paths it has not seen before.

A reset step closes that gap. Each step is numbered, applied in order from the
version stamped in the database up to ``CURRENT_SEMANTICS_VERSION``, and stamped
only once it finishes, so an interrupted run resumes rather than half-applying.
Steps must be idempotent: re-running one on rows it has already fixed changes
nothing.

The registry, not any individual step, is the durable part. A step is free to be
as broad or as surgical as its particular drift calls for.
"""

import logging
import time

from app.assets.database.queries.semantics import (
    get_semantics_version,
    set_semantics_version,
)
from app.assets.semantics.reproject_derived import reproject_derived_state
from app.assets.semantics.step import (
    InterruptCheck,
    SemanticsStep,
    SemanticsStepInterrupted,
)
from app.database.db import can_create_session, create_session

__all__ = [
    "CURRENT_SEMANTICS_VERSION",
    "SEMANTICS_STEPS",
    "InterruptCheck",
    "SemanticsStep",
    "SemanticsStepInterrupted",
    "run_pending_semantics_steps",
]

SEMANTICS_STEPS: tuple[SemanticsStep, ...] = (
    SemanticsStep(
        version=1,
        description="reproject path-derived reference state",
        apply=reproject_derived_state,
    ),
)

CURRENT_SEMANTICS_VERSION = SEMANTICS_STEPS[-1].version


def run_pending_semantics_steps(interrupt_check: InterruptCheck | None = None) -> int:
    """Apply every reset step this database has not been stamped for.

    A database already at ``CURRENT_SEMANTICS_VERSION`` costs one indexed row
    read and nothing else -- no filesystem is touched to discover there is
    nothing to do, which is the overwhelmingly common case.

    Returns the number of steps applied.
    """
    if not can_create_session():
        return 0

    try:
        with create_session() as session:
            stored_version = get_semantics_version(session)
    except Exception:
        logging.exception(
            "Could not read the asset semantics version; skipping semantics reset"
        )
        return 0

    pending = [step for step in SEMANTICS_STEPS if step.version > stored_version]
    if not pending:
        return 0

    logging.info(
        "Asset semantics at version %d, current is %d: applying %d step(s)",
        stored_version,
        CURRENT_SEMANTICS_VERSION,
        len(pending),
    )

    applied = 0
    for step in pending:
        started = time.perf_counter()
        try:
            summary = step.apply(interrupt_check)
        except SemanticsStepInterrupted:
            logging.info(
                "Asset semantics step %d (%s) interrupted; resuming on next start",
                step.version,
                step.description,
            )
            return applied
        except Exception:
            logging.exception(
                "Asset semantics step %d (%s) failed; database stays at version %d",
                step.version,
                step.description,
                stored_version + applied,
            )
            return applied

        with create_session() as session:
            set_semantics_version(session, step.version)
            session.commit()

        applied += 1
        logging.info(
            "Asset semantics step %d (%s) applied in %.3fs: %s",
            step.version,
            step.description,
            time.perf_counter() - started,
            summary,
        )

    return applied

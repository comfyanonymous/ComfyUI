"""What a semantics reset step is. See ``app.assets.semantics`` for the why."""

from dataclasses import dataclass
from typing import Callable

InterruptCheck = Callable[[], bool]


class SemanticsStepInterrupted(Exception):
    """A step stopped early on request.

    Its partial work stands -- steps are idempotent, so a resumed run repeats it
    harmlessly -- but the version is not stamped, so the step runs again.
    """


@dataclass(frozen=True)
class SemanticsStep:
    """One numbered reset step.

    ``apply`` takes an optional interrupt check and returns a summary of what it
    did, which is logged. It must be idempotent and must raise
    ``SemanticsStepInterrupted`` rather than return if it stops early.
    """

    version: int
    description: str
    apply: Callable[[InterruptCheck | None], object]

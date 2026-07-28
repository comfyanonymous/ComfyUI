"""Segment planning.

Split a known byte range into S roughly-equal segments, each fetched by its
own coroutine with ``Range: bytes=start-end``. Falls back to a single segment
when the server doesn't support ranges or the size is unknown/too small for
segmentation to be worthwhile.
"""

from __future__ import annotations

from dataclasses import dataclass

# Below this size, the per-connection setup cost outweighs any parallelism.
_MIN_SEGMENT_BYTES = 1 * 1024 * 1024


@dataclass(frozen=True)
class SegmentPlan:
    idx: int
    start: int
    end: int  # inclusive

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def effective_segment_count(
    total_bytes: int | None, accept_ranges: bool, configured: int
) -> int:
    """How many segments to actually use for this file."""
    if not accept_ranges or total_bytes is None or total_bytes <= 0:
        return 1
    by_size = max(1, total_bytes // _MIN_SEGMENT_BYTES)
    return max(1, min(configured, by_size))


def plan_segments(total_bytes: int, num_segments: int) -> list[SegmentPlan]:
    """Return ``num_segments`` contiguous, inclusive byte ranges covering [0, total)."""
    if total_bytes <= 0 or num_segments <= 1:
        return [SegmentPlan(idx=0, start=0, end=max(0, total_bytes - 1))]
    base = total_bytes // num_segments
    plans: list[SegmentPlan] = []
    start = 0
    for i in range(num_segments):
        # Last segment soaks up the remainder.
        length = base if i < num_segments - 1 else total_bytes - start
        end = start + length - 1
        plans.append(SegmentPlan(idx=i, start=start, end=end))
        start = end + 1
    return plans

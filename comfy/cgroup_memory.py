"""Container (cgroup) aware system memory reporting.

ComfyUI sizes its RAM caches and its pinned memory budget from psutil, which
reads /proc/meminfo. That file is not namespaced, so inside a container with a
memory limit psutil reports the host's RAM rather than the limit the process is
actually killed at. The RAM pressure logic then compares against host-wide
numbers, never frees anything, and the kernel OOM-kills the process instead.

These helpers clamp the reported values to the active cgroup limit. When there
is no limit -- bare metal, or an unconstrained container -- they return psutil's
values untouched, so behaviour on those systems is unchanged.
"""

import logging
import os
import sys

import psutil

CGROUP_V2_ROOT = "/sys/fs/cgroup"
CGROUP_V1_MEMORY_ROOT = "/sys/fs/cgroup/memory"

# cgroup v1 spells "unlimited" as a large sentinel rather than a keyword.
CGROUP_V1_UNLIMITED = 1 << 62

# Probed once; the cgroup layout and limit do not change while we run.
# Usage is deliberately not cached -- it is the value that moves.
_LIMIT = None
_CANDIDATE_DIRS = None
_USAGE_DIR = None


def _read(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except (OSError, ValueError):
        return None


def _read_int(path):
    raw = _read(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_stat_field(path, field):
    """Read a single key out of a cgroup memory.stat style file."""
    content = _read(path)
    if content is None:
        return None
    for line in content.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == field:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def _own_cgroup_dirs():
    """Resolve this process' own memory cgroup directory.

    With a private cgroup namespace -- the Docker default -- the container's
    cgroup is mounted at the root of /sys/fs/cgroup and the root paths below
    are enough. This covers the other case (for example --cgroupns=host, or
    cgroup v1), where the process sits in a sub-path such as /docker/<id>.
    """
    content = _read("/proc/self/cgroup")
    if content is None:
        return []

    dirs = []
    for line in content.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        path = path.lstrip("/")
        if not path:
            # "0::/" -- a process in the root cgroup, i.e. not constrained.
            continue
        if controllers == "":  # cgroup v2 unified hierarchy
            dirs.append(os.path.join(CGROUP_V2_ROOT, path))
        elif "memory" in controllers.split(","):
            dirs.append(os.path.join(CGROUP_V1_MEMORY_ROOT, path))
    return dirs


def _candidate_dirs():
    global _CANDIDATE_DIRS
    if _CANDIDATE_DIRS is None:
        _CANDIDATE_DIRS = [CGROUP_V2_ROOT, CGROUP_V1_MEMORY_ROOT] + _own_cgroup_dirs()
    return _CANDIDATE_DIRS


def _limit_from_dir(directory):
    # cgroup v2. Note the v2 root cgroup has no memory.max at all, so on bare
    # metal this simply finds nothing.
    raw = _read(os.path.join(directory, "memory.max"))
    if raw is not None and raw != "max":
        try:
            return int(raw)
        except ValueError:
            pass

    # cgroup v1
    limit = _read_int(os.path.join(directory, "memory.limit_in_bytes"))
    if limit is not None and limit < CGROUP_V1_UNLIMITED:
        return limit

    return None


def cgroup_memory_limit():
    """Effective memory limit in bytes, or None when unconstrained.

    A limit at or above physical RAM tells us nothing the host total doesn't
    already, so it is treated as no limit.
    """
    global _LIMIT
    if _LIMIT is not None:
        return _LIMIT[0]

    limit = None
    if sys.platform.startswith("linux"):
        for directory in _candidate_dirs():
            limit = _limit_from_dir(directory)
            if limit is not None:
                break

        if limit is not None and limit >= psutil.virtual_memory().total:
            limit = None

        if limit is not None:
            logging.info("Detected cgroup memory limit {:0.0f} MB".format(limit / (1024 * 1024)))

    _LIMIT = (limit,)
    return limit


def _usage_from_dir(directory):
    """Non-reclaimable usage for a cgroup directory.

    memory.current counts the page cache, which the kernel reclaims long before
    it resorts to killing anything. Subtracting inactive_file gives the working
    set, matching what cAdvisor and the kubelet use for the same decision.
    """
    usage = _read_int(os.path.join(directory, "memory.current"))  # cgroup v2
    if usage is not None:
        inactive_file = _read_stat_field(os.path.join(directory, "memory.stat"), "inactive_file") or 0
        return max(0, usage - inactive_file)

    usage = _read_int(os.path.join(directory, "memory.usage_in_bytes"))  # cgroup v1
    if usage is not None:
        inactive_file = _read_stat_field(os.path.join(directory, "memory.stat"), "total_inactive_file") or 0
        return max(0, usage - inactive_file)

    return None


def cgroup_memory_usage():
    """Current non-reclaimable usage in bytes, or None if unavailable."""
    global _USAGE_DIR
    if _USAGE_DIR is not None:
        return _usage_from_dir(_USAGE_DIR)

    for directory in _candidate_dirs():
        usage = _usage_from_dir(directory)
        if usage is not None:
            _USAGE_DIR = directory
            return usage
    return None


def virtual_memory_total():
    """psutil.virtual_memory().total, clamped to the cgroup limit."""
    total = psutil.virtual_memory().total
    limit = cgroup_memory_limit()
    if limit is None:
        return total
    return min(total, limit)


def virtual_memory_available():
    """psutil.virtual_memory().available, clamped to what the cgroup allows.

    Swap is deliberately not counted. A cgroup may permit unlimited swap while
    the host's swap is already exhausted, which would put us straight back to
    over-reporting.
    """
    available = psutil.virtual_memory().available
    limit = cgroup_memory_limit()
    if limit is None:
        return available

    usage = cgroup_memory_usage()
    if usage is None:
        return min(available, limit)

    return max(0, min(available, limit - usage))

"""Container aware system memory reporting.

ComfyUI sizes its RAM caches and its pinned memory budget from psutil, which
reads /proc/meminfo. That file is not namespaced, so inside a container with a
memory limit psutil reports the host's RAM rather than the limit the process is
actually killed at. The RAM pressure logic then compares against host-wide
numbers, never frees anything, and the kernel OOM-kills the process instead.

These helpers clamp the reported values to the limit that actually applies. When
there is no limit -- bare metal, or an unconstrained container -- they return
psutil's values untouched, so behaviour on those systems is unchanged.

Callers only need virtual_memory_total() and virtual_memory_available(). The
cgroup specifics are an implementation detail of this module.
"""

import logging
import os
import sys

import psutil

CGROUP_V2_ROOT = "/sys/fs/cgroup"
CGROUP_V1_MEMORY_ROOT = "/sys/fs/cgroup/memory"

# cgroup v1 spells "unlimited" as a large sentinel rather than a keyword.
CGROUP_V1_UNLIMITED = 1 << 62

# The discovered hierarchy is intentionally not cached. A process can be moved
# between cgroups and an orchestrator can resize a cgroup while ComfyUI runs.
# Tests may set this to avoid touching the real /proc filesystem.
_HIERARCHY_DIRS = None

# Unlike the hierarchy itself, this only suppresses duplicate log messages.
# Keep an explicit unset state so a transition from a finite limit to no limit
# is still recorded and a later finite limit is logged again.
_UNSET = object()
_LOGGED_LIMIT = _UNSET


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


def _own_cgroup_scopes():
    """[(mount_root, cgroup_dir)] for this process, one entry per hierarchy.

    With a private cgroup namespace -- the Docker default -- /proc/self/cgroup
    reads "0::/" and the container's own cgroup is mounted at the root, so the
    mount root is the right directory. This also covers --cgroupns=host and
    cgroup v1, where the process sits in a sub-path such as /docker/<id>.
    """
    content = _read("/proc/self/cgroup")
    if content is None:
        return []

    scopes = []
    for line in content.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        path = path.lstrip("/")

        if controllers == "":  # cgroup v2 unified hierarchy
            root = CGROUP_V2_ROOT
        elif "memory" in controllers.split(","):
            root = CGROUP_V1_MEMORY_ROOT
        else:
            continue

        scopes.append((root, os.path.join(root, path) if path else root))
    return scopes


def _ancestors(mount_root, directory):
    """directory, then each parent up to and including mount_root."""
    root = os.path.normpath(mount_root)
    current = os.path.normpath(directory)

    out = []
    while True:
        out.append(current)
        if current == root:
            break
        parent = os.path.dirname(current)
        if parent == current or not parent.startswith(root):
            break
        current = parent
    return out


def _hierarchy_dirs():
    """Directories to consult, nearest cgroup first, then its ancestors.

    Ordering matters. Reading the mount roots first would let a limit found on
    the process' own cgroup be paired with usage read from the host root, which
    on a host cgroup namespace reports host-wide usage and can drive the
    available figure to zero.
    """
    if _HIERARCHY_DIRS is not None:
        return _HIERARCHY_DIRS

    dirs = []
    for root, own in _own_cgroup_scopes():
        for directory in _ancestors(root, own):
            if directory not in dirs:
                dirs.append(directory)

    # /proc/self/cgroup unreadable (or no memory controller): fall back to the
    # well-known roots so a namespaced container still works.
    if not dirs:
        dirs.extend((CGROUP_V2_ROOT, CGROUP_V1_MEMORY_ROOT))

    return dirs


def _limit_from_dir(directory):
    # cgroup v2. Note the v2 root cgroup has no memory.max at all, so on bare
    # metal this simply finds nothing.
    raw = _read(os.path.join(directory, "memory.max"))
    if raw is not None and raw != "max":
        try:
            value = int(raw)
            if value >= 0:
                return value
            # Negative is malformed. Treat it as absent rather than letting it
            # propagate into a negative "total", and try the v1 file below.
        except ValueError:
            pass

    # cgroup v1
    limit = _read_int(os.path.join(directory, "memory.limit_in_bytes"))
    if limit is not None and 0 <= limit < CGROUP_V1_UNLIMITED:
        return limit

    return None


def _resolve_scope():
    """(limit, directory) for the most binding finite limit in the hierarchy.

    A cgroup inherits its ancestors' limits, so the effective ceiling is the
    smallest one found walking up. Returning the directory alongside it lets
    usage be read from the same level, which is what keeps the two consistent.
    """
    global _LOGGED_LIMIT

    limit = None
    directory = None

    if sys.platform.startswith("linux"):
        for candidate in _hierarchy_dirs():
            found = _limit_from_dir(candidate)
            if found is not None and (limit is None or found < limit):
                limit, directory = found, candidate

        # A limit at or above physical RAM tells us nothing the host total does
        # not already, so it is treated as no limit.
        if limit is not None and limit >= psutil.virtual_memory().total:
            limit, directory = None, None

        if limit != _LOGGED_LIMIT:
            _LOGGED_LIMIT = limit
            if limit is not None:
                logging.info("Detected cgroup memory limit {:0.0f} MB".format(limit / (1024 * 1024)))

    return limit, directory


def cgroup_memory_limit():
    """Effective memory limit in bytes, or None when unconstrained."""
    return _resolve_scope()[0]


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
    """Current non-reclaimable usage in bytes, or None if unavailable.

    Read from the same cgroup the limit came from. Any other directory would
    describe a different scope, and subtracting one from the other would be
    meaningless.
    """
    directory = _resolve_scope()[1]
    if directory is None:
        return None
    return _usage_from_dir(directory)


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
    limit, directory = _resolve_scope()
    if limit is None:
        return available

    usage = None if directory is None else _usage_from_dir(directory)
    if usage is None:
        return min(available, limit)

    return max(0, min(available, limit - usage))

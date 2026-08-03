import sys

import psutil
import pytest

from comfy import cgroup_memory


@pytest.fixture(autouse=True)
def reset_probe_cache():
    """The module probes the cgroup layout once and caches it."""
    cgroup_memory._LIMIT = None
    cgroup_memory._CANDIDATE_DIRS = None
    cgroup_memory._USAGE_DIR = None
    yield
    cgroup_memory._LIMIT = None
    cgroup_memory._CANDIDATE_DIRS = None
    cgroup_memory._USAGE_DIR = None


def use_dirs(monkeypatch, *dirs):
    monkeypatch.setattr(cgroup_memory, "_CANDIDATE_DIRS", [str(d) for d in dirs])


def write(directory, name, content):
    (directory / name).write_text(content)


# --- no limit: must fall through to psutil untouched -------------------------

def test_no_cgroup_files_falls_back_to_psutil(monkeypatch, tmp_path):
    """Bare metal / Windows / macOS: nothing to read, psutil is authoritative."""
    use_dirs(monkeypatch, tmp_path / "missing")

    assert cgroup_memory.cgroup_memory_limit() is None
    assert cgroup_memory.virtual_memory_total() == psutil.virtual_memory().total


def test_v2_root_has_no_memory_max(monkeypatch, tmp_path):
    """The cgroup v2 root cgroup exposes memory.stat but no memory.max, which
    is exactly what an unconstrained host looks like."""
    write(tmp_path, "memory.stat", "inactive_file 1024\n")
    use_dirs(monkeypatch, tmp_path)

    assert cgroup_memory.cgroup_memory_limit() is None


def test_v2_max_keyword_means_unconstrained(monkeypatch, tmp_path):
    write(tmp_path, "memory.max", "max\n")
    use_dirs(monkeypatch, tmp_path)

    assert cgroup_memory.cgroup_memory_limit() is None


def test_v1_unlimited_sentinel(monkeypatch, tmp_path):
    write(tmp_path, "memory.limit_in_bytes", "9223372036854771712\n")
    use_dirs(monkeypatch, tmp_path)

    assert cgroup_memory.cgroup_memory_limit() is None


def test_limit_at_or_above_physical_ram_is_ignored(monkeypatch, tmp_path):
    write(tmp_path, "memory.max", str(psutil.virtual_memory().total * 2))
    use_dirs(monkeypatch, tmp_path)

    assert cgroup_memory.cgroup_memory_limit() is None
    assert cgroup_memory.virtual_memory_total() == psutil.virtual_memory().total


def test_root_cgroup_path_yields_no_candidates(monkeypatch):
    """An unconstrained process reports "0::/" -- no sub-path to inspect."""
    monkeypatch.setattr(cgroup_memory, "_read", lambda path: "0::/" if path == "/proc/self/cgroup" else None)

    assert cgroup_memory._own_cgroup_dirs() == []


def test_non_linux_never_probes(monkeypatch, tmp_path):
    write(tmp_path, "memory.max", str(8 * 1024 ** 3))
    use_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")

    assert cgroup_memory.cgroup_memory_limit() is None


# --- with a limit ------------------------------------------------------------

def test_v2_limit_and_usage(monkeypatch, tmp_path):
    write(tmp_path, "memory.max", str(8 * 1024 ** 3))
    write(tmp_path, "memory.current", str(6 * 1024 ** 3))
    write(tmp_path, "memory.stat", "anon 123\ninactive_file 0\n")
    use_dirs(monkeypatch, tmp_path)

    assert cgroup_memory.cgroup_memory_limit() == 8 * 1024 ** 3
    assert cgroup_memory.cgroup_memory_usage() == 6 * 1024 ** 3
    assert cgroup_memory.virtual_memory_total() == 8 * 1024 ** 3


def test_v1_limit_and_usage(monkeypatch, tmp_path):
    write(tmp_path, "memory.limit_in_bytes", str(8 * 1024 ** 3))
    write(tmp_path, "memory.usage_in_bytes", str(6 * 1024 ** 3))
    write(tmp_path, "memory.stat", "total_inactive_file 0\n")
    use_dirs(monkeypatch, tmp_path)

    assert cgroup_memory.cgroup_memory_limit() == 8 * 1024 ** 3
    assert cgroup_memory.cgroup_memory_usage() == 6 * 1024 ** 3


def test_reclaimable_page_cache_is_not_counted_as_used(monkeypatch, tmp_path):
    """memory.current includes the page cache, which the kernel reclaims before
    it OOM-kills. Counting it as used reports a nearly-full cgroup and causes
    constant, pointless cache eviction.

    Numbers taken from the report in Comfy-Org/ComfyUI#14938: a 90 GiB limit
    sitting at 89.9 GiB current, of which 46 GiB is reclaimable page cache.
    """
    limit, current, inactive_file = 96636764160, 96570064896, 49504276480
    write(tmp_path, "memory.max", str(limit))
    write(tmp_path, "memory.current", str(current))
    write(tmp_path, "memory.stat", "anon 45096034304\ninactive_file {}\n".format(inactive_file))
    use_dirs(monkeypatch, tmp_path)

    assert cgroup_memory.cgroup_memory_usage() == current - inactive_file

    # Naive limit - memory.current would leave ~64 MB; the working set leaves ~46 GB.
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _fake_vmem(total=limit * 2, available=limit * 2))
    assert cgroup_memory.virtual_memory_available() == limit - (current - inactive_file)
    assert cgroup_memory.virtual_memory_available() > 46 * 1024 ** 3


def test_available_never_exceeds_psutil(monkeypatch, tmp_path):
    """The host can be under pressure even when the cgroup is not."""
    write(tmp_path, "memory.max", str(8 * 1024 ** 3))
    write(tmp_path, "memory.current", str(1 * 1024 ** 3))
    write(tmp_path, "memory.stat", "inactive_file 0\n")
    use_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _fake_vmem(total=64 * 1024 ** 3, available=512 * 1024 ** 2))

    assert cgroup_memory.virtual_memory_available() == 512 * 1024 ** 2


def test_available_is_never_negative(monkeypatch, tmp_path):
    """Usage can exceed the limit briefly before the kernel reclaims."""
    write(tmp_path, "memory.max", str(8 * 1024 ** 3))
    write(tmp_path, "memory.current", str(9 * 1024 ** 3))
    write(tmp_path, "memory.stat", "inactive_file 0\n")
    use_dirs(monkeypatch, tmp_path)

    assert cgroup_memory.virtual_memory_available() == 0


def test_usage_unreadable_still_clamps_total(monkeypatch, tmp_path):
    write(tmp_path, "memory.max", str(8 * 1024 ** 3))
    use_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _fake_vmem(total=64 * 1024 ** 3, available=32 * 1024 ** 3))

    assert cgroup_memory.cgroup_memory_usage() is None
    assert cgroup_memory.virtual_memory_available() == 8 * 1024 ** 3


# --- malformed input ---------------------------------------------------------

@pytest.mark.parametrize("value", ["", "garbage", "12 34", "-"])
def test_malformed_limit_does_not_raise(monkeypatch, tmp_path, value):
    write(tmp_path, "memory.max", value)
    use_dirs(monkeypatch, tmp_path)

    assert cgroup_memory.cgroup_memory_limit() is None
    assert cgroup_memory.virtual_memory_available() == psutil.virtual_memory().available


def test_trailing_newline_is_handled(monkeypatch, tmp_path):
    """cgroup files end in a newline; the value must still parse."""
    write(tmp_path, "memory.max", "34359738368\n")
    use_dirs(monkeypatch, tmp_path)

    assert cgroup_memory.cgroup_memory_limit() == 34359738368


def test_missing_memory_stat_treats_page_cache_as_zero(monkeypatch, tmp_path):
    write(tmp_path, "memory.max", str(8 * 1024 ** 3))
    write(tmp_path, "memory.current", str(3 * 1024 ** 3))
    use_dirs(monkeypatch, tmp_path)

    assert cgroup_memory.cgroup_memory_usage() == 3 * 1024 ** 3


def _fake_vmem(total, available):
    class _VMem:
        pass

    mem = _VMem()
    mem.total = total
    mem.available = available
    return mem

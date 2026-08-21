import logging
import sys

import psutil
import pytest

from comfy import system_memory


@pytest.fixture(autouse=True)
def reset_probe_cache(monkeypatch):
    """Run cgroup behavior tests as Linux tests on every host."""
    monkeypatch.setattr(system_memory.sys, "platform", "linux")
    system_memory._HIERARCHY_DIRS = None
    system_memory._LOGGED_LIMIT = system_memory._UNSET
    yield
    system_memory._HIERARCHY_DIRS = None
    system_memory._LOGGED_LIMIT = system_memory._UNSET


def use_dirs(monkeypatch, *dirs):
    monkeypatch.setattr(system_memory, "_HIERARCHY_DIRS", [str(d) for d in dirs])


def write(directory, name, content):
    (directory / name).write_text(content)


# --- no limit: must fall through to psutil untouched -------------------------

def test_no_cgroup_files_falls_back_to_psutil(monkeypatch, tmp_path):
    """Bare metal / Windows / macOS: nothing to read, psutil is authoritative."""
    use_dirs(monkeypatch, tmp_path / "missing")

    assert system_memory.cgroup_memory_limit() is None
    assert system_memory.virtual_memory_total() == psutil.virtual_memory().total


def test_v2_root_has_no_memory_max(monkeypatch, tmp_path):
    """The cgroup v2 root cgroup exposes memory.stat but no memory.max, which
    is exactly what an unconstrained host looks like."""
    write(tmp_path, "memory.stat", "inactive_file 1024\n")
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_limit() is None


def test_v2_max_keyword_means_unconstrained(monkeypatch, tmp_path):
    write(tmp_path, "memory.max", "max\n")
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_limit() is None


def test_v1_unlimited_sentinel(monkeypatch, tmp_path):
    write(tmp_path, "memory.limit_in_bytes", "9223372036854771712\n")
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_limit() is None


def test_limit_at_or_above_physical_ram_is_ignored(monkeypatch, tmp_path):
    write(tmp_path, "memory.max", str(psutil.virtual_memory().total * 2))
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_limit() is None
    assert system_memory.virtual_memory_total() == psutil.virtual_memory().total


def test_root_cgroup_path_resolves_to_the_mount_root(monkeypatch):
    """A process reports "0::/" both when unconstrained and, more importantly,
    inside a private cgroup namespace -- the Docker default -- where its own
    limited cgroup is mounted at the root. The mount root is the directory to
    read in both cases; whether a limit exists there is decided separately."""
    monkeypatch.setattr(system_memory, "_read", lambda path: "0::/" if path == "/proc/self/cgroup" else None)

    assert system_memory._own_cgroup_scopes() == [
        (system_memory.CGROUP_V2_ROOT, system_memory.CGROUP_V2_ROOT)
    ]


def test_v1_memory_controller_scope(monkeypatch):
    """Only the v1 memory controller should produce a memory scope."""
    content = "3:memory:/docker/abc123\n2:cpu,cpuacct:/docker/abc123\n"
    monkeypatch.setattr(system_memory, "_read",
                        lambda path: content if path == "/proc/self/cgroup" else None)

    assert system_memory._own_cgroup_scopes() == [
        (system_memory.CGROUP_V1_MEMORY_ROOT,
         system_memory.CGROUP_V1_MEMORY_ROOT + "/docker/abc123")
    ]


def test_non_linux_never_probes(monkeypatch, tmp_path):
    write(tmp_path, "memory.max", str(8 * 1024 ** 3))
    use_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")

    assert system_memory.cgroup_memory_limit() is None


# --- with a limit ------------------------------------------------------------

def test_v2_limit_and_usage(monkeypatch, tmp_path):
    write(tmp_path, "memory.max", str(8 * 1024 ** 3))
    write(tmp_path, "memory.current", str(6 * 1024 ** 3))
    write(tmp_path, "memory.stat", "anon 123\ninactive_file 0\n")
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_limit() == 8 * 1024 ** 3
    assert system_memory.cgroup_memory_usage() == 6 * 1024 ** 3
    assert system_memory.virtual_memory_total() == 8 * 1024 ** 3


def test_v1_limit_and_usage(monkeypatch, tmp_path):
    write(tmp_path, "memory.limit_in_bytes", str(8 * 1024 ** 3))
    write(tmp_path, "memory.usage_in_bytes", str(6 * 1024 ** 3))
    write(tmp_path, "memory.stat", "total_inactive_file 0\n")
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_limit() == 8 * 1024 ** 3
    assert system_memory.cgroup_memory_usage() == 6 * 1024 ** 3


def test_reclaimable_page_cache_is_not_counted_as_used(monkeypatch, tmp_path):
    """memory.current includes the page cache, which the kernel reclaims before
    it OOM-kills. Counting it as used reports a nearly-full cgroup and causes
    constant, pointless cache eviction.

    Numbers taken from the report in Comfy-Org/ComfyUI#14938: a 90 GiB limit
    sitting at 89.9 GiB current, of which 46 GiB is reclaimable page cache.
    """
    limit, current, inactive_file = 96636764160, 96570064896, 49504276480
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: _fake_vmem(total=128 * 1024 ** 3, available=128 * 1024 ** 3))
    write(tmp_path, "memory.max", str(limit))
    write(tmp_path, "memory.current", str(current))
    write(tmp_path, "memory.stat", "anon 45096034304\ninactive_file {}\n".format(inactive_file))
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_usage() == current - inactive_file

    # Naive limit - memory.current would leave ~64 MB; the working set leaves ~46 GB.
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _fake_vmem(total=limit * 2, available=limit * 2))
    assert system_memory.virtual_memory_available() == limit - (current - inactive_file)
    assert system_memory.virtual_memory_available() > 46 * 1024 ** 3


def test_available_never_exceeds_psutil(monkeypatch, tmp_path):
    """The host can be under pressure even when the cgroup is not."""
    write(tmp_path, "memory.max", str(8 * 1024 ** 3))
    write(tmp_path, "memory.current", str(1 * 1024 ** 3))
    write(tmp_path, "memory.stat", "inactive_file 0\n")
    use_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _fake_vmem(total=64 * 1024 ** 3, available=512 * 1024 ** 2))

    assert system_memory.virtual_memory_available() == 512 * 1024 ** 2


def test_available_is_never_negative(monkeypatch, tmp_path):
    """Usage can exceed the limit briefly before the kernel reclaims."""
    write(tmp_path, "memory.max", str(8 * 1024 ** 3))
    write(tmp_path, "memory.current", str(9 * 1024 ** 3))
    write(tmp_path, "memory.stat", "inactive_file 0\n")
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.virtual_memory_available() == 0


def test_usage_unreadable_still_clamps_total(monkeypatch, tmp_path):
    write(tmp_path, "memory.max", str(8 * 1024 ** 3))
    use_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _fake_vmem(total=64 * 1024 ** 3, available=32 * 1024 ** 3))

    assert system_memory.cgroup_memory_usage() is None
    assert system_memory.virtual_memory_available() == 8 * 1024 ** 3


# --- malformed input ---------------------------------------------------------

@pytest.mark.parametrize("value", ["", "garbage", "12 34", "-"])
def test_malformed_limit_does_not_raise(monkeypatch, tmp_path, value):
    fixed_memory = _fake_vmem(total=64 * 1024 ** 3, available=32 * 1024 ** 3)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: fixed_memory)
    write(tmp_path, "memory.max", value)
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_limit() is None
    assert system_memory.virtual_memory_available() == fixed_memory.available


def test_trailing_newline_is_handled(monkeypatch, tmp_path):
    """cgroup files end in a newline; the value must still parse."""
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: _fake_vmem(total=64 * 1024 ** 3, available=32 * 1024 ** 3))
    write(tmp_path, "memory.max", "34359738368\n")
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_limit() == 34359738368


def test_missing_memory_stat_treats_page_cache_as_zero(monkeypatch, tmp_path):
    write(tmp_path, "memory.max", str(8 * 1024 ** 3))
    write(tmp_path, "memory.current", str(3 * 1024 ** 3))
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_usage() == 3 * 1024 ** 3


def _fake_vmem(total, available):
    class _VMem:
        pass

    mem = _VMem()
    mem.total = total
    mem.available = available
    return mem


# --- hierarchy: limit and usage must come from the same cgroup ----------------

def test_constrained_child_below_unconstrained_root(monkeypatch, tmp_path):
    """The case that motivates walking the hierarchy at all.

    With --cgroupns=host the mount root is the *host* root: unconstrained, but
    reporting host-wide usage. The container's limit lives on a child. Reading
    the root first would pair the child's limit with the host's usage.
    """
    root = tmp_path / "root"
    child = root / "docker" / "abc123"
    child.mkdir(parents=True)

    # Host root: no limit, and usage far larger than the child's limit.
    write(root, "memory.max", "max")
    write(root, "memory.current", str(40 * 1024 ** 3))
    write(root, "memory.stat", "inactive_file 0\n")

    # Container: 8 GB limit, 2 GB in use.
    write(child, "memory.max", str(8 * 1024 ** 3))
    write(child, "memory.current", str(2 * 1024 ** 3))
    write(child, "memory.stat", "inactive_file 0\n")

    use_dirs(monkeypatch, child, root)

    assert system_memory.cgroup_memory_limit() == 8 * 1024 ** 3
    # The bug this guards: 40 GB of host usage against an 8 GB limit would
    # clamp available to zero and make every allocation look impossible.
    assert system_memory.cgroup_memory_usage() == 2 * 1024 ** 3
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: _fake_vmem(total=64 * 1024 ** 3, available=50 * 1024 ** 3))
    assert system_memory.virtual_memory_available() == 6 * 1024 ** 3


def test_inherited_parent_limit(monkeypatch, tmp_path):
    """A cgroup with no limit of its own still inherits its parent's."""
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)

    write(parent, "memory.max", str(4 * 1024 ** 3))
    write(parent, "memory.current", str(3 * 1024 ** 3))
    write(parent, "memory.stat", "inactive_file 0\n")

    write(child, "memory.max", "max")
    write(child, "memory.current", str(1 * 1024 ** 3))
    write(child, "memory.stat", "inactive_file 0\n")

    use_dirs(monkeypatch, child, parent)

    assert system_memory.cgroup_memory_limit() == 4 * 1024 ** 3
    # Usage is read from the level that owns the binding limit, not the child.
    assert system_memory.cgroup_memory_usage() == 3 * 1024 ** 3


def test_most_binding_limit_wins(monkeypatch, tmp_path):
    """When several ancestors set a limit, the smallest is the one that kills."""
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)

    write(parent, "memory.max", str(16 * 1024 ** 3))
    write(child, "memory.max", str(2 * 1024 ** 3))
    write(child, "memory.current", str(1 * 1024 ** 3))
    write(child, "memory.stat", "inactive_file 0\n")

    use_dirs(monkeypatch, child, parent)

    assert system_memory.cgroup_memory_limit() == 2 * 1024 ** 3
    assert system_memory.cgroup_memory_usage() == 1 * 1024 ** 3


def test_ancestors_walk_stops_at_the_mount_root(tmp_path):
    root = tmp_path / "sys" / "fs" / "cgroup"
    deep = root / "docker" / "abc"
    assert system_memory._ancestors(str(root), str(deep)) == [
        str(deep), str(root / "docker"), str(root)
    ]


def test_limit_is_reread_after_cgroup_resize(monkeypatch, tmp_path):
    cgroup_limit = tmp_path / "cgroup" / "container"
    cgroup_limit.mkdir(parents=True)
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: _fake_vmem(total=64 * 1024 ** 3, available=32 * 1024 ** 3))
    use_dirs(monkeypatch, cgroup_limit)

    write(cgroup_limit, "memory.max", str(8 * 1024 ** 3))
    assert system_memory.cgroup_memory_limit() == 8 * 1024 ** 3

    write(cgroup_limit, "memory.max", str(4 * 1024 ** 3))
    assert system_memory.cgroup_memory_limit() == 4 * 1024 ** 3


def test_limit_is_logged_only_when_it_changes(monkeypatch, tmp_path, caplog):
    cgroup_limit = tmp_path / "cgroup"
    cgroup_limit.mkdir()
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: _fake_vmem(total=64 * 1024 ** 3, available=32 * 1024 ** 3))
    use_dirs(monkeypatch, cgroup_limit)

    with caplog.at_level(logging.INFO):
        write(cgroup_limit, "memory.max", str(8 * 1024 ** 3))
        assert system_memory.cgroup_memory_limit() == 8 * 1024 ** 3
        assert system_memory.cgroup_memory_limit() == 8 * 1024 ** 3

        write(cgroup_limit, "memory.max", "max")
        assert system_memory.cgroup_memory_limit() is None

        # Returning to the previous finite value is a real change because the
        # intervening no-limit state was recorded.
        write(cgroup_limit, "memory.max", str(8 * 1024 ** 3))
        assert system_memory.cgroup_memory_limit() == 8 * 1024 ** 3

    assert [record.message for record in caplog.records
            if "Detected cgroup memory limit" in record.message] == [
                "Detected cgroup memory limit 8192 MB",
                "Detected cgroup memory limit 8192 MB",
            ]


def test_unrelated_root_is_not_used_when_process_scope_exists(monkeypatch, tmp_path):
    v2_root = tmp_path / "cgroup"
    process_scope = v2_root / "container"
    unrelated_root = tmp_path / "memory"
    process_scope.mkdir(parents=True)
    unrelated_root.mkdir()

    write(process_scope, "memory.max", str(8 * 1024 ** 3))
    write(unrelated_root, "memory.max", str(1 * 1024 ** 3))
    monkeypatch.setattr(system_memory, "CGROUP_V2_ROOT", str(v2_root))
    monkeypatch.setattr(system_memory, "CGROUP_V1_MEMORY_ROOT", str(unrelated_root))
    monkeypatch.setattr(system_memory, "_own_cgroup_scopes",
                        lambda: [(str(v2_root), str(process_scope))])

    assert system_memory.cgroup_memory_limit() == 8 * 1024 ** 3


# --- malformed data ----------------------------------------------------------

def test_negative_v2_limit_is_rejected(monkeypatch, tmp_path):
    """A malformed memory.max of -1 must not become a negative total."""
    write(tmp_path, "memory.max", "-1")
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_limit() is None
    assert system_memory.virtual_memory_total() == psutil.virtual_memory().total


def test_negative_v1_limit_is_rejected(monkeypatch, tmp_path):
    write(tmp_path, "memory.limit_in_bytes", "-4096")
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_limit() is None
    assert system_memory.virtual_memory_total() == psutil.virtual_memory().total


def test_negative_v2_limit_falls_through_to_v1(monkeypatch, tmp_path):
    """A bad v2 value should not mask a usable v1 limit in the same directory."""
    write(tmp_path, "memory.max", "-1")
    write(tmp_path, "memory.limit_in_bytes", str(8 * 1024 ** 3))
    use_dirs(monkeypatch, tmp_path)

    assert system_memory.cgroup_memory_limit() == 8 * 1024 ** 3

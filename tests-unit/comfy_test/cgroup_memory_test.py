import pytest
import psutil
from unittest.mock import mock_open, patch

import comfy.model_management as model_management


HOST_TOTAL = 754 * 1024 ** 3
HOST_AVAILABLE = 683 * 1024 ** 3
HOST_USED = HOST_TOTAL - HOST_AVAILABLE
CONTAINER_LIMIT = 90 * 1024 ** 3


# Built once, from the real reading, so that patching psutil below cannot recurse.
HOST_MEMORY = psutil.virtual_memory()._replace(
    total=HOST_TOTAL,
    available=HOST_AVAILABLE,
    free=HOST_AVAILABLE,
    used=HOST_USED,
    percent=round(HOST_USED * 100.0 / HOST_TOTAL, 1),
)


@pytest.fixture
def host_psutil():
    with patch("psutil.virtual_memory", lambda: HOST_MEMORY):
        yield


def write_cgroup(directory, limit, usage):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "memory.max").write_text(f"{limit}\n")
    (directory / "memory.current").write_text(f"{usage}\n")


def self_paths(root, path):
    return lambda: [(str(root), "memory.max", "memory.current", path)]


def test_read_cgroup_int(tmp_path):
    path = tmp_path / "memory.max"

    path.write_text("96636764160\n")
    assert model_management._read_cgroup_int(str(path)) == 96636764160

    path.write_text("max\n")
    assert model_management._read_cgroup_int(str(path)) is None

    path.write_text("not a number\n")
    assert model_management._read_cgroup_int(str(path)) is None

    assert model_management._read_cgroup_int(str(tmp_path / "missing")) is None


def test_self_paths_reads_v2_and_v1_lines():
    with patch("builtins.open", mock_open(read_data="0::/docker/abc\n")):
        assert model_management._cgroup_self_paths() == [
            model_management.CGROUP_V2 + ("/docker/abc",)
        ]

    with patch("builtins.open", mock_open(read_data="8:memory:/kubepods/pod1\n4:cpu:/\n")):
        assert model_management._cgroup_self_paths() == [
            model_management.CGROUP_V1 + ("/kubepods/pod1",)
        ]


def test_self_paths_ignores_controllers_without_memory():
    with patch("builtins.open", mock_open(read_data="4:cpu,cpuacct:/some/path\n")):
        assert model_management._cgroup_self_paths() == []


def test_constraint_takes_the_smallest_limit_in_the_hierarchy(tmp_path):
    # An ancestor with a lower limit binds this process, and its usage is the one that counts.
    write_cgroup(tmp_path, 4 * 1024 ** 3, 1 * 1024 ** 3)
    write_cgroup(tmp_path / "outer", 2 * 1024 ** 3, 900 * 1024 ** 2)
    write_cgroup(tmp_path / "outer" / "inner", 3 * 1024 ** 3, 100 * 1024 ** 2)

    with patch.object(model_management, "_cgroup_self_paths",
                      self_paths(tmp_path, "/outer/inner")):
        limit, usage_file = model_management._cgroup_memory_constraint()

    assert limit == 2 * 1024 ** 3
    assert usage_file == str(tmp_path / "outer" / "memory.current")


def test_constraint_reads_a_cgroup_the_mount_root_does_not_expose(tmp_path):
    # Under a host cgroup namespace the mount root has no memory files at all.
    write_cgroup(tmp_path / "docker" / "abc", CONTAINER_LIMIT, 0)

    with patch.object(model_management, "_cgroup_self_paths",
                      self_paths(tmp_path, "/docker/abc")):
        limit, _ = model_management._cgroup_memory_constraint()

    assert limit == CONTAINER_LIMIT


def test_constraint_is_none_without_a_limit(tmp_path):
    (tmp_path / "memory.max").write_text("max\n")

    with patch.object(model_management, "_cgroup_self_paths", self_paths(tmp_path, "/")):
        assert model_management._cgroup_memory_constraint() is None


def test_no_constraint_passes_psutil_through(host_psutil):
    with patch.object(model_management, "CGROUP_MEMORY", None):
        mem = model_management.virtual_memory()

    assert mem.total == HOST_TOTAL
    assert mem.available == HOST_AVAILABLE


def test_limit_larger_than_host_is_ignored(host_psutil, tmp_path):
    with patch.object(model_management, "CGROUP_MEMORY", (HOST_TOTAL * 2, str(tmp_path))):
        mem = model_management.virtual_memory()

    assert mem.total == HOST_TOTAL
    assert mem.available == HOST_AVAILABLE


def test_limit_clamps_total_and_available(host_psutil, tmp_path):
    used = 47 * 1024 ** 3
    usage_file = tmp_path / "memory.current"
    usage_file.write_text(f"{used}\n")

    with patch.object(model_management, "CGROUP_MEMORY", (CONTAINER_LIMIT, str(usage_file))):
        mem = model_management.virtual_memory()

    assert mem.total == CONTAINER_LIMIT
    assert mem.used == used
    assert mem.available == CONTAINER_LIMIT - used
    assert mem.free <= mem.available
    assert mem.percent == pytest.approx(used * 100.0 / CONTAINER_LIMIT, abs=0.1)


def test_usage_above_limit_does_not_go_negative(host_psutil, tmp_path):
    usage_file = tmp_path / "memory.current"
    usage_file.write_text(f"{CONTAINER_LIMIT * 2}\n")

    with patch.object(model_management, "CGROUP_MEMORY", (CONTAINER_LIMIT, str(usage_file))):
        mem = model_management.virtual_memory()

    assert mem.used == CONTAINER_LIMIT
    assert mem.available == 0


def test_unreadable_usage_falls_back_to_host_figures(host_psutil, tmp_path):
    missing = tmp_path / "memory.current"

    with patch.object(model_management, "CGROUP_MEMORY", (CONTAINER_LIMIT, str(missing))):
        mem = model_management.virtual_memory()

    assert mem.total == CONTAINER_LIMIT
    assert mem.used == HOST_USED
    assert mem.available == CONTAINER_LIMIT - HOST_USED

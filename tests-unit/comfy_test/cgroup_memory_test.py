import pytest
import psutil
import torch
from unittest.mock import mock_open, patch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

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


def write_cgroup(directory, limit, usage=0):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "memory.max").write_text(f"{limit}\n")
    (directory / "memory.current").write_text(f"{usage}\n")


def hierarchies(root, path):
    return [(str(root), "memory.max", "memory.current", path)]


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
                      lambda: hierarchies(tmp_path, "/outer/inner")):
        limit, usage_file = model_management._cgroup_memory_constraint()

    assert limit == 2 * 1024 ** 3
    assert usage_file == str(tmp_path / "outer" / "memory.current")


def test_constraint_reads_a_cgroup_the_mount_root_does_not_expose(tmp_path):
    # Under a host cgroup namespace the mount root has no memory files at all.
    write_cgroup(tmp_path / "docker" / "abc", CONTAINER_LIMIT)

    with patch.object(model_management, "_cgroup_self_paths",
                      lambda: hierarchies(tmp_path, "/docker/abc")):
        limit, _ = model_management._cgroup_memory_constraint()

    assert limit == CONTAINER_LIMIT


def test_constraint_is_none_without_a_limit(tmp_path):
    (tmp_path / "memory.max").write_text("max\n")

    with patch.object(model_management, "_cgroup_self_paths", lambda: hierarchies(tmp_path, "/")):
        assert model_management._cgroup_memory_constraint() is None


def test_no_hierarchy_passes_psutil_through(host_psutil):
    with patch.object(model_management, "_cgroup_self_paths", lambda: []):
        mem = model_management.virtual_memory()

    assert mem.total == HOST_TOTAL
    assert mem.available == HOST_AVAILABLE


def test_limit_larger_than_host_is_ignored(host_psutil, tmp_path):
    write_cgroup(tmp_path, HOST_TOTAL * 2)

    with patch.object(model_management, "_cgroup_self_paths", lambda: hierarchies(tmp_path, "/")):
        mem = model_management.virtual_memory()

    assert mem.total == HOST_TOTAL
    assert mem.available == HOST_AVAILABLE


def test_limit_clamps_total_and_available(host_psutil, tmp_path):
    used = 47 * 1024 ** 3
    write_cgroup(tmp_path, CONTAINER_LIMIT, used)

    with patch.object(model_management, "_cgroup_self_paths", lambda: hierarchies(tmp_path, "/")):
        mem = model_management.virtual_memory()

    assert mem.total == CONTAINER_LIMIT
    assert mem.used == used
    assert mem.available == CONTAINER_LIMIT - used
    assert mem.free <= mem.available
    assert mem.percent == pytest.approx(used * 100.0 / CONTAINER_LIMIT, abs=0.1)


def test_limit_lowered_while_running_is_picked_up(host_psutil, tmp_path):
    # memory.max is writable: docker update -m, or an in-place pod resize.
    write_cgroup(tmp_path, CONTAINER_LIMIT, 1 * 1024 ** 3)

    with patch.object(model_management, "_cgroup_self_paths", lambda: hierarchies(tmp_path, "/")):
        assert model_management.virtual_memory().total == CONTAINER_LIMIT
        (tmp_path / "memory.max").write_text(f"{CONTAINER_LIMIT // 2}\n")
        assert model_management.virtual_memory().total == CONTAINER_LIMIT // 2


def test_moving_to_another_cgroup_is_picked_up(host_psutil, tmp_path):
    # A process can be moved by a write to cgroup.procs, so where it sits is resolved on
    # every call rather than once at import.
    write_cgroup(tmp_path / "first", CONTAINER_LIMIT)
    write_cgroup(tmp_path / "second", CONTAINER_LIMIT // 3)
    cgroup = "/first"

    with patch.object(model_management, "_cgroup_self_paths",
                      lambda: hierarchies(tmp_path, cgroup)):
        assert model_management.virtual_memory().total == CONTAINER_LIMIT
        cgroup = "/second"
        assert model_management.virtual_memory().total == CONTAINER_LIMIT // 3


def test_usage_above_limit_does_not_go_negative(host_psutil, tmp_path):
    write_cgroup(tmp_path, CONTAINER_LIMIT, CONTAINER_LIMIT * 2)

    with patch.object(model_management, "_cgroup_self_paths", lambda: hierarchies(tmp_path, "/")):
        mem = model_management.virtual_memory()

    assert mem.used == CONTAINER_LIMIT
    assert mem.available == 0


def test_missing_usage_falls_back_to_host_figures(host_psutil, tmp_path):
    (tmp_path / "memory.max").write_text(f"{CONTAINER_LIMIT}\n")  # no memory.current

    with patch.object(model_management, "_cgroup_self_paths", lambda: hierarchies(tmp_path, "/")):
        mem = model_management.virtual_memory()

    assert mem.total == CONTAINER_LIMIT
    assert mem.used == HOST_USED
    assert mem.available == CONTAINER_LIMIT - HOST_USED

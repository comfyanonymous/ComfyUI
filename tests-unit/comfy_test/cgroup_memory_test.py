import pytest
import psutil
from unittest.mock import patch

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


def test_read_cgroup_int(tmp_path):
    path = tmp_path / "memory.max"

    path.write_text("96636764160\n")
    assert model_management._read_cgroup_int(str(path)) == 96636764160

    path.write_text("max\n")
    assert model_management._read_cgroup_int(str(path)) is None

    path.write_text("not a number\n")
    assert model_management._read_cgroup_int(str(path)) is None

    assert model_management._read_cgroup_int(str(tmp_path / "missing")) is None


def test_no_cgroup_limit_passes_psutil_through(host_psutil):
    with patch.object(model_management, "CGROUP_MEMORY_LIMIT", None):
        mem = model_management.virtual_memory()

    assert mem.total == HOST_TOTAL
    assert mem.available == HOST_AVAILABLE


def test_limit_larger_than_host_is_ignored(host_psutil):
    with patch.object(model_management, "CGROUP_MEMORY_LIMIT", HOST_TOTAL * 2):
        mem = model_management.virtual_memory()

    assert mem.total == HOST_TOTAL
    assert mem.available == HOST_AVAILABLE


def test_limit_clamps_total_and_available(host_psutil):
    used = 47 * 1024 ** 3
    with patch.object(model_management, "CGROUP_MEMORY_LIMIT", CONTAINER_LIMIT), \
         patch.object(model_management, "_cgroup_memory_usage", lambda: used):
        mem = model_management.virtual_memory()

    assert mem.total == CONTAINER_LIMIT
    assert mem.used == used
    assert mem.available == CONTAINER_LIMIT - used
    assert mem.free <= mem.available
    assert mem.percent == pytest.approx(used * 100.0 / CONTAINER_LIMIT, abs=0.1)


def test_usage_above_limit_does_not_go_negative(host_psutil):
    with patch.object(model_management, "CGROUP_MEMORY_LIMIT", CONTAINER_LIMIT), \
         patch.object(model_management, "_cgroup_memory_usage",
                      lambda: CONTAINER_LIMIT * 2):
        mem = model_management.virtual_memory()

    assert mem.used == CONTAINER_LIMIT
    assert mem.available == 0


def test_unreadable_usage_falls_back_to_host_figures(host_psutil):
    with patch.object(model_management, "CGROUP_MEMORY_LIMIT", CONTAINER_LIMIT), \
         patch.object(model_management, "_cgroup_memory_usage", lambda: None):
        mem = model_management.virtual_memory()

    assert mem.total == CONTAINER_LIMIT
    assert mem.used == HOST_USED
    assert mem.available == CONTAINER_LIMIT - HOST_USED

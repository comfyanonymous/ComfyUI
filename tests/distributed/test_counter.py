import os
import shutil
import subprocess
import sys
from threading import Thread, Barrier
from pathlib import Path
import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from comfy.component_model.file_counter import FileCounter
from comfy.component_model.folder_path_types import FolderNames
from comfy.execution_context import context_folder_names_and_paths
from comfy.cmd.folder_paths import init_default_paths


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None


def run_command(command, check=True):
    """Helper to run a shell command."""
    try:
        return subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        print(f"--- STDOUT ---\n{e.stdout}")
        print(f"--- STDERR ---\n{e.stderr}")
        print("--------------")
        raise


@pytest.fixture(
    params=[
        pytest.param("local", id="local_filesystem"),
        pytest.param(
            "nfs",
            id="nfs_share",
            marks=pytest.mark.skipif(
                not sys.platform.startswith("linux")
                or not is_tool("mount.nfs")
                or not is_tool("sudo")
                or not os.path.exists("/sys/module/nfsd"),
                reason="NFS tests require sudo, nfs-common, and the 'nfsd' kernel module to be loaded.",
            ),
        ),
        pytest.param(
            "samba",
            id="samba_share",
            marks=pytest.mark.skipif(
                not sys.platform.startswith("linux")
                or not is_tool("mount.cifs")
                or not is_tool("sudo"),
                reason="Samba tests require sudo on Linux with cifs-utils installed.",
            ),
        ),
    ]
)
def counter_path_factory(request, tmp_path_factory):
    """A parameterized fixture to provide paths on local, NFS, and Samba filesystems."""
    if request.param == "local":
        yield lambda name: str(tmp_path_factory.mktemp("local_test") / name)
        return

    mount_point = tmp_path_factory.mktemp(f"mount_point_{request.param}")

    if request.param == "nfs":
        # 1. Create the host directory that will be mounted into the container.
        nfs_source = tmp_path_factory.mktemp("nfs_source")
        # 2. FIX: Set permissions on the *host* directory.
        os.chmod(str(nfs_source), 0o777)

        # 3. FIX: Use the new container's required path: /mnt/data
        container_path = "/mnt/data"

        # 4. FIX: Change to the new container image and configuration
        nfs_container = DockerContainer("ghcr.io/normal-computing/nfs-server:latest").with_env(
            "NFS_SERVER_ALLOWED_CLIENTS", "*"
        ).with_kwargs(privileged=True).with_exposed_ports(2049).with_volume_mapping(
            str(nfs_source), container_path, mode="rw"  # Mount to /mnt/data
        )

        nfs_container.start()

        # 5. FIX: Wait for the new container's export log message
        # (and remove the timeout as requested)
        nfs_container.waiting_for(
            LogMessageWaitStrategy(r"exporting /mnt/data")
        )

        request.addfinalizer(lambda: nfs_container.stop())

        ip_address = nfs_container.get_container_host_ip()
        nfs_port = nfs_container.get_exposed_port(2049)
        try:
            # 6. FIX: Mount using the new container's command format
            # (mounts root ":" and uses "-t nfs4")
            run_command(f"sleep 1 && sudo mount -t nfs4 -o proto=tcp,port={nfs_port} {ip_address}:/ {mount_point}")
            yield lambda name: str(mount_point / name)
        finally:
            run_command(f"sudo umount {mount_point}", check=False)

    elif request.param == "samba":
        # 1. Create the host directory.
        samba_source = tmp_path_factory.mktemp("samba_source")
        # 2. Set permissions on the *host* directory.
        os.chmod(str(samba_source), 0o777)

        share_name = "storage"

        # 3. FIX: Add the NAME environment variable to tell the container
        # to create a share named "storage".
        samba_container = DockerContainer("dockurr/samba:latest").with_env(
            "RW", "yes"
        ).with_env(
            "NAME", share_name  # <-- This is the crucial fix
        ).with_exposed_ports(445).with_volume_mapping(
            str(samba_source), "/storage", mode="rw"  # This maps the host dir to the internal /storage path
        )

        samba_container.start()

        # 4. Wait for the correct log message
        # (and remove the timeout as requested)
        samba_container.waiting_for(
            LogMessageWaitStrategy(r"smbd version .* started")
        )

        request.addfinalizer(lambda: samba_container.stop())

        ip_address = samba_container.get_container_host_ip()
        samba_port = samba_container.get_exposed_port(445)
        try:
            # 5. FIX: Mount with the default username/password, not as guest.
            run_command(f"sleep 1 && sudo mount -t cifs -o username=samba,password=secret,vers=3.0,port={samba_port},uid=$(id -u),gid=$(id -g) //{ip_address}/{share_name} {mount_point}", check=True)
            yield lambda name: str(mount_point / name)
        finally:
            run_command(f"sudo umount {mount_point}", check=False)


def test_initial_state(counter_path_factory):
    """Verify initial state and file creation."""
    counter_file = counter_path_factory("counter.txt")
    lock_file = counter_path_factory("counter.txt.lock")

    assert not os.path.exists(counter_file)
    assert not os.path.exists(lock_file)

    counter = FileCounter(str(counter_file))
    assert counter.get_and_increment() == 0
    assert os.path.exists(counter_file)
    with open(counter_file, "r") as f:
        assert f.read() == "1"


def test_mkdirs(counter_path_factory):
    counter_file = counter_path_factory("new_dir/counter.txt")
    assert not os.path.exists(os.path.dirname(counter_file))

    counter = FileCounter(str(counter_file))
    assert counter.get_and_increment() == 0
    assert counter.get_and_increment() == 1


def test_increment_and_decrement(counter_path_factory):
    """Test the increment and decrement logic."""
    counter_file = counter_path_factory("counter.txt")
    counter = FileCounter(str(counter_file))

    assert counter.get_and_increment() == 0  # val: 0, new_val: 1
    assert counter.get_and_increment() == 1  # val: 1, new_val: 2
    assert counter.get_and_increment() == 2  # val: 2, new_val: 3

    assert counter.decrement_and_get() == 2  # val: 3, new_val: 2
    assert counter.decrement_and_get() == 1  # val: 2, new_val: 1

    assert counter.get_and_increment() == 1  # val: 1, new_val: 2


def test_multiple_instances_same_path(counter_path_factory):
    """Verify that multiple FileCounter instances on the same path work correctly."""
    counter_file = counter_path_factory("counter.txt")
    counter1 = FileCounter(str(counter_file))
    counter2 = FileCounter(str(counter_file))

    assert counter1.get_and_increment() == 0
    assert counter2.get_and_increment() == 1
    assert counter1.decrement_and_get() == 1
    assert counter2.get_and_increment() == 1

    with open(counter_file, "r") as f:
        assert f.read() == "2"


def test_multithreaded_access(counter_path_factory):
    """Ensure atomicity with multiple threads."""
    counter_file = counter_path_factory("counter.txt")
    counter = FileCounter(str(counter_file))
    num_threads = 10
    increments_per_thread = 100

    def worker():
        for _ in range(increments_per_thread):
            counter.get_and_increment()

    threads = [Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open(counter_file, "r") as f:
        final_value = int(f.read())
        assert final_value == num_threads * increments_per_thread


def test_context_manager(counter_path_factory):
    """Test that the counter can be used as a context manager."""
    counter_file = counter_path_factory("counter.txt")
    counter = FileCounter(str(counter_file))

    # Initial state should be 0
    assert counter.get_and_increment() == 0
    with open(counter_file) as f:
        assert f.read() == "1"

    with counter as wrapper:
        # The wrapper's value is the original count before increment.
        assert wrapper.value == 1
        # It can also be used as an integer directly.
        assert int(wrapper) == 1
        with open(counter_file) as f:
            assert f.read() == "2"  # Inside context, value is 2

    with open(counter_file) as f:
        assert f.read() == "1"  # Exited context, decremented back to 1
    # After exit, the wrapper's 'ctr' attribute holds the new value.
    assert wrapper.ctr == 1


async def test_cleanup_temp_multithreaded(tmp_path):
    """
    Test that cleanup_temp correctly deletes the temp directory only
    after the last thread has exited the context.
    """
    # 1. Use the application's context to define the temp directory for this test.
    # This is a cleaner approach than mocking.
    base_dir = tmp_path / "base"
    temp_dir_override = base_dir / "temp"
    fn = FolderNames(base_paths=[base_dir])
    init_default_paths(fn, base_paths_from_configuration=False)
    # Override the default temp path
    fn.temp_directory = temp_dir_override

    from comfy.component_model.file_counter import cleanup_temp

    num_threads = 5
    # Barrier to make threads wait for each other before exiting.
    barrier = Barrier(num_threads)
    loop = asyncio.get_running_loop()

    def worker():
        """The task for each thread. It enters the cleanup context and waits."""
        with cleanup_temp():
            # The temp directory and counter file should exist inside the context.
            assert temp_dir_override.exists()
            assert (temp_dir_override / "counter.txt").exists()
        # After exiting, the directory should still exist until the last thread is done.

    # Use the context manager to set the folder paths for the current async context.
    with context_folder_names_and_paths(fn):
        # Capture the current context, which now includes the folder_paths settings.

        # Run the worker function in a thread pool, applying the captured context to each task.
        with ThreadPoolExecutor(max_workers=num_threads) as executor:

            tasks = [loop.run_in_executor(executor, contextvars.copy_context().run, worker) for _ in range(num_threads)]
            # Wait for all threads to complete their work.
            await asyncio.gather(*tasks)

    # After all threads have joined, the counter should be 0, and the directory deleted.
    assert not temp_dir_override.exists()
    # the base dir is not going to be deleted
    assert base_dir.exists()

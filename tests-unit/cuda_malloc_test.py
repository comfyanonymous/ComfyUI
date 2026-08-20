import subprocess
from unittest.mock import call, patch

import cuda_malloc


@patch("cuda_malloc.os.path.isfile")
def test_get_nvidia_smi_path_uses_standard_windows_driver_location(isfile):
    system_root = r"C:\Windows"
    program_files = r"C:\Program Files"
    system_path = cuda_malloc.os.path.join(system_root, "System32", "nvidia-smi.exe")
    standard_path = cuda_malloc.os.path.join(program_files, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
    isfile.side_effect = lambda path: path == standard_path

    with (
        patch.object(cuda_malloc.os, "name", "nt"),
        patch.dict(cuda_malloc.os.environ, {"SystemRoot": system_root, "ProgramW6432": program_files}, clear=True),
    ):
        assert cuda_malloc._get_nvidia_smi_path() == standard_path

    assert isfile.call_args_list == [call(system_path), call(standard_path)]


@patch("cuda_malloc._get_nvidia_smi_path", return_value="nvidia-smi")
@patch("cuda_malloc.subprocess.check_output")
def test_get_nvidia_gpu_names_uses_physical_gpu_records(check_output, get_nvidia_smi_path):
    check_output.return_value = b"""GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-a)
  MIG 1g.5gb Device 0: (UUID: MIG-a)
GPU 1: NVIDIA A100-SXM4-40GB (UUID: GPU-b)
"""

    assert cuda_malloc.get_nvidia_gpu_names() == [
        "NVIDIA A100-SXM4-40GB",
        "NVIDIA A100-SXM4-40GB",
    ]
    check_output.assert_called_once_with(["nvidia-smi", "-L"], timeout=5)
    get_nvidia_smi_path.assert_called_once_with()


@patch("cuda_malloc.subprocess.check_output")
def test_get_nvidia_gpu_names_reports_one_physical_gpu(check_output):
    check_output.return_value = b"GPU 0: NVIDIA GeForce RTX 4060 Ti (UUID: GPU-a)\n"

    assert cuda_malloc.get_nvidia_gpu_names() == ["NVIDIA GeForce RTX 4060 Ti"]


@patch("cuda_malloc.subprocess.check_output")
def test_get_nvidia_gpu_names_handles_nvidia_smi_failure(check_output):
    check_output.side_effect = subprocess.CalledProcessError(1, ["nvidia-smi", "-L"])

    assert cuda_malloc.get_nvidia_gpu_names() is None


@patch("cuda_malloc.subprocess.check_output")
def test_get_nvidia_gpu_names_handles_nvidia_smi_timeout(check_output):
    check_output.side_effect = subprocess.TimeoutExpired(["nvidia-smi", "-L"], 5)

    assert cuda_malloc.get_nvidia_gpu_names() is None


@patch("cuda_malloc.subprocess.check_output")
def test_get_nvidia_gpu_names_handles_unexpected_output(check_output):
    for output in (
        b"No devices were found\n",
        b"GPU 0 NVIDIA GeForce RTX 4070\n",
        b"GPU 0: NVIDIA GeForce RTX \xff (UUID: GPU-a)\n",
    ):
        check_output.return_value = output
        assert cuda_malloc.get_nvidia_gpu_names() is None


@patch("cuda_malloc.get_gpu_names")
@patch("cuda_malloc.get_nvidia_gpu_names", return_value=None)
def test_get_nvidia_gpu_count_falls_back_to_windows_display_devices(get_nvidia_gpu_names, get_gpu_names):
    get_gpu_names.return_value = [
        "NVIDIA GeForce RTX 5080",
        "NVIDIA GeForce RTX 5060",
        "AMD Radeon Graphics",
    ]

    assert cuda_malloc.get_nvidia_gpu_count() == 2


@patch("cuda_malloc.get_gpu_names")
@patch(
    "cuda_malloc.get_nvidia_gpu_names",
    return_value=["NVIDIA GeForce RTX 5080", "NVIDIA GeForce RTX 5060"],
)
def test_get_nvidia_gpu_count_prefers_physical_gpu_records(get_nvidia_gpu_names, get_gpu_names):
    assert cuda_malloc.get_nvidia_gpu_count() == 2
    get_gpu_names.assert_not_called()

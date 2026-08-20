import subprocess
from unittest.mock import patch

import cuda_malloc


@patch("cuda_malloc.subprocess.check_output")
def test_get_nvidia_gpu_names_uses_physical_gpu_records(check_output):
    check_output.return_value = b"""GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-a)
  MIG 1g.5gb Device 0: (UUID: MIG-a)
GPU 1: NVIDIA A100-SXM4-40GB (UUID: GPU-b)
"""

    assert cuda_malloc.get_nvidia_gpu_names() == [
        "NVIDIA A100-SXM4-40GB",
        "NVIDIA A100-SXM4-40GB",
    ]
    check_output.assert_called_once_with(["nvidia-smi", "-L"])


@patch("cuda_malloc.subprocess.check_output")
def test_get_nvidia_gpu_names_reports_one_physical_gpu(check_output):
    check_output.return_value = b"GPU 0: NVIDIA GeForce RTX 4060 Ti (UUID: GPU-a)\n"

    assert cuda_malloc.get_nvidia_gpu_names() == ["NVIDIA GeForce RTX 4060 Ti"]


@patch("cuda_malloc.subprocess.check_output")
def test_get_nvidia_gpu_names_handles_nvidia_smi_failure(check_output):
    check_output.side_effect = subprocess.CalledProcessError(1, ["nvidia-smi", "-L"])

    assert cuda_malloc.get_nvidia_gpu_names() == []

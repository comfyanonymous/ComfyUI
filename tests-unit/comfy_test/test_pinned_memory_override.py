import pytest
import torch

from comfy.cli_args import args as cli_args, parser as cli_parser

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management as model_management  # noqa: E402


def test_pinned_memory_override_bypasses_ram_calculation():
    ram = 64 * 1024 ** 3

    assert model_management.calculate_max_pinned_memory(None, ram) != 8 * 1024 ** 3
    assert model_management.calculate_max_pinned_memory(8, ram) == 8 * 1024 ** 3


def test_pinned_memory_default_uses_ram_based_calculation(monkeypatch):
    ram = 64 * 1024 ** 3

    monkeypatch.setattr(model_management, "WINDOWS", False)
    monkeypatch.setattr(model_management, "get_disk_swap_total", lambda: 0)

    expected = max(ram * 0.40, min(ram * 0.90, ram - 4 * 1024 ** 3, ram - 16 * 1024 ** 3))
    assert model_management.calculate_max_pinned_memory(None, ram) == expected


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf", "1e308"])
def test_pinned_memory_rejects_negative_and_non_finite_values(value):
    with pytest.raises(SystemExit):
        cli_parser.parse_args(["--pinned-memory", value])

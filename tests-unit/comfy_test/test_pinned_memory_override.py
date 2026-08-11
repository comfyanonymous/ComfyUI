import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management as model_management  # noqa: E402


def test_pinned_memory_override_bypasses_ram_calculation():
    ram = 64 * 1024 ** 3

    assert model_management.calculate_max_pinned_memory(None, ram) != 8 * 1024 ** 3
    assert model_management.calculate_max_pinned_memory(8, ram) == 8 * 1024 ** 3


def test_pinned_memory_default_uses_ram_based_calculation():
    ram = 64 * 1024 ** 3

    default = model_management.calculate_max_pinned_memory(None, ram)

    assert default > 0
    assert default <= ram

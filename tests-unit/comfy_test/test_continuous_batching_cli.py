import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPORT_CLI = "import comfy.options; comfy.options.enable_args_parsing(); import comfy.cli_args"
PRINT_CONTINUOUS_BATCHING = IMPORT_CLI + "; print(comfy.cli_args.args.continuous_batching)"
PRINT_DYNAMIC_VRAM = IMPORT_CLI + "; print(comfy.cli_args.enables_dynamic_vram())"


def run_cli(*arguments, script=IMPORT_CLI):
    return subprocess.run(
        [sys.executable, "-c", script, *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_continuous_batching_defaults_to_disabled():
    result = run_cli(script=PRINT_CONTINUOUS_BATCHING)
    assert result.returncode == 0
    assert result.stdout.strip() == "0"


def test_continuous_batching_stores_the_prompt_limit():
    result = run_cli("--continuous-batching", "4", script=PRINT_CONTINUOUS_BATCHING)
    assert result.returncode == 0
    assert result.stdout.strip() == "4"


def test_continuous_batching_rejects_negative_limit():
    result = run_cli("--continuous-batching", "-1")
    assert result.returncode == 2
    assert "must be zero or greater" in result.stderr


def test_multi_prompt_continuous_batching_rejects_cache_none():
    result = run_cli("--continuous-batching", "2", "--cache-none")
    assert result.returncode == 2
    assert "incompatible with --cache-none" in result.stderr


def test_single_prompt_continuous_batching_allows_cache_none():
    result = run_cli("--continuous-batching", "1", "--cache-none")
    assert result.returncode == 0


def test_continuous_batching_disables_automatic_dynamic_vram():
    result = run_cli("--continuous-batching", "1", script=PRINT_DYNAMIC_VRAM)
    assert result.returncode == 0
    assert result.stdout.strip() == "False"


def test_continuous_batching_rejects_explicit_dynamic_vram():
    result = run_cli("--continuous-batching", "2", "--enable-dynamic-vram")
    assert result.returncode == 2
    assert "incompatible with --enable-dynamic-vram" in result.stderr

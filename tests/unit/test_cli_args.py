import pytest
from unittest.mock import patch
from comfy import cli_args
from comfy.cli_args_types import LatentPreviewMethod

# Helper function to parse args and return the Configuration object
def _parse_test_args(args_list):
    parser = cli_args._create_parser()
    # The `args_parsing=True` makes it use the provided list.
    with patch.object(parser, 'parse_known_args_with_config_files', return_value=(parser.parse_known_args(args_list)[0], [], [])):
        return cli_args._parse_args(parser, args_parsing=True)

@pytest.mark.parametrize("args, expected", [
    ([], []),
    (['--extra-model-paths-config', 'a'], ['a']),
    (['--extra-model-paths-config', 'a', '--extra-model-paths-config', 'b'], ['a', 'b']),
    (['--extra-model-paths-config', 'a,b'], ['a', 'b']),
    (['--extra-model-paths-config', 'a,b', '--extra-model-paths-config', 'c'], ['a', 'b', 'c']),
    (['--extra-model-paths-config', ' a , b ', '--extra-model-paths-config', 'c'], ['a', 'b', 'c']),
    (['--extra-model-paths-config', 'a,b', 'c'], ['a', 'b', 'c']),
])
def test_extra_model_paths_config(args, expected):
    """Test that extra_model_paths_config is parsed correctly."""
    config = _parse_test_args(args)
    assert config.extra_model_paths_config == expected

def test_default_values():
    """Test that default values are set correctly when no args are provided."""
    config = _parse_test_args([])
    assert config.listen == "127.0.0.1"
    assert config.port == 8188
    assert config.auto_launch is False
    assert config.extra_model_paths_config == []
    assert config.preview_method == LatentPreviewMethod.Auto
    assert config.logging_level == 'INFO'
    assert config.multi_user is False
    assert config.disable_xformers is False
    assert config.gpu_only is False
    assert config.highvram is False
    assert config.lowvram is False
    assert config.normalvram is False
    assert config.novram is False
    assert config.cpu is False

def test_listen_and_port():
    """Test --listen and --port arguments."""
    config = _parse_test_args(['--listen', '0.0.0.0', '--port', '8000'])
    assert config.listen == '0.0.0.0'
    assert config.port == 8000

def test_listen_no_arg():
    """Test --listen without an argument."""
    config = _parse_test_args(['--listen'])
    assert config.listen == '0.0.0.0,::'

def test_auto_launch_flags():
    """Test --auto-launch and --disable-auto-launch flags."""
    config_auto = _parse_test_args(['--auto-launch'])
    assert config_auto.auto_launch is True

    config_disable = _parse_test_args(['--disable-auto-launch'])
    assert config_disable.auto_launch is False

    # Test that --disable-auto-launch overrides --auto-launch if both are present
    # The order matters, argparse behavior. Last one wins for store_true/false.
    config_both_1 = _parse_test_args(['--auto-launch', '--disable-auto-launch'])
    assert config_both_1.auto_launch is False

    config_both_2 = _parse_test_args(['--disable-auto-launch', '--auto-launch'])
    assert config_both_2.auto_launch is False

def test_windows_standalone_build_enables_auto_launch():
    """Test that --windows-standalone-build enables auto-launch."""
    config = _parse_test_args(['--windows-standalone-build'])
    assert config.windows_standalone_build is True
    assert config.auto_launch is True

def test_windows_standalone_build_with_disable_auto_launch():
    """Test that --disable-auto-launch overrides --windows-standalone-build's auto-launch."""
    config = _parse_test_args(['--windows-standalone-build', '--disable-auto-launch'])
    assert config.windows_standalone_build is True
    assert config.auto_launch is False

def test_force_fp16_enables_fp16_unet():
    """Test that --force-fp16 enables --fp16-unet."""
    config = _parse_test_args(['--force-fp16'])
    assert config.force_fp16 is True
    assert config.fp16_unet is True

@pytest.mark.parametrize("vram_arg, expected_true_field", [
    ('--gpu-only', 'gpu_only'),
    ('--highvram', 'highvram'),
    ('--normalvram', 'normalvram'),
    ('--lowvram', 'lowvram'),
    ('--novram', 'novram'),
    ('--cpu', 'cpu'),
])
def test_vram_modes(vram_arg, expected_true_field):
    """Test mutually exclusive VRAM mode arguments."""
    config = _parse_test_args([vram_arg])
    all_vram_fields = ['gpu_only', 'highvram', 'normalvram', 'lowvram', 'novram', 'cpu']
    for field in all_vram_fields:
        if field == expected_true_field:
            assert getattr(config, field) is True
        else:
            assert getattr(config, field) is False

def test_preview_method():
    """Test the --preview-method argument."""
    config = _parse_test_args(['--preview-method', 'TAESD'])
    assert config.preview_method == LatentPreviewMethod.TAESD

def test_logging_level():
    """Test the --logging-level argument."""
    config = _parse_test_args(['--logging-level', 'debug'])
    assert config.logging_level == 'DEBUG'

def test_multi_user():
    """Test the --multi-user flag."""
    config = _parse_test_args(['--multi-user'])
    assert config.multi_user is True

def test_disable_xformers():
    """Test the --disable-xformers flag."""
    config = _parse_test_args(['--disable-xformers'])
    assert config.disable_xformers is True

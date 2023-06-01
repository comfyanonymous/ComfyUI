from comfy.cli_args import ComfyConfigLoader


def test_defaults():
    config = "{}"
    argv = []

    args = ComfyConfigLoader().parse_args_with_string(config, argv)

    assert args.listen == "127.0.0.1"
    assert args.novram == False


def test_config():
    config = """
config:
  network:
    listen: 0.0.0.0
"""
    argv = []

    args = ComfyConfigLoader().parse_args_with_string(config, argv)

    assert args.listen == "0.0.0.0"


def test_cli_args_overrides_config():
    config = """
config:
  network:
    listen: 0.0.0.0
"""
    argv = ["--listen", "192.168.1.100"]

    args = ComfyConfigLoader().parse_args_with_string(config, argv)

    assert args.listen == "192.168.1.100"


def test_config_enum():
    config = """
config:
  pytorch:
    cross_attention: split
"""
    argv = []

    args = ComfyConfigLoader().parse_args_with_string(config, argv)

    assert args.use_split_cross_attention is True
    assert args.use_pytorch_cross_attention is False


def test_config_enum_default():
    config = """
config:
  pytorch:
    cross_attention:
"""
    argv = []

    args = ComfyConfigLoader().parse_args_with_string(config, argv)

    assert args.use_split_cross_attention is False
    assert args.use_pytorch_cross_attention is False


def test_config_enum_exclusive():
    config = """
config:
  pytorch:
    cross_attention: split
"""
    argv = ["--use-pytorch-cross-attention"]

    args = ComfyConfigLoader().parse_args_with_string(config, argv)

    assert args.use_split_cross_attention is False
    assert args.use_pytorch_cross_attention is True

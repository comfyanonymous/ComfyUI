import os
import pytest
from pathlib import Path
import argparse
import comfy.cli_args as cli_args
import importlib
import comfy.options

# No additional imports required since all necessary modules
# (pytest, comfy.cli_args, etc.) are already imported in the test file.
def test_is_valid_directory(tmp_path):
    """
    Test the is_valid_directory function from comfy.cli_args.
    Verifies that:
    - Passing None returns None.
    - A valid directory returns the same path string.
    - An invalid directory path raises an argparse.ArgumentTypeError.
    """
    assert cli_args.is_valid_directory(None) is None
    valid_dir = str(tmp_path)
    returned = cli_args.is_valid_directory(valid_dir)
    assert returned == valid_dir
    invalid_dir = os.path.join(valid_dir, "non_existing_dir")
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        cli_args.is_valid_directory(invalid_dir)
    assert invalid_dir in str(excinfo.value)
def test_listen_argument_no_value():
    """
    Test that when the '--listen' argument is provided without a following value,
    the parser uses the const value "0.0.0.0,::" instead of the default.
    """
    test_args = ["--listen"]
    args = cli_args.parser.parse_args(test_args)
    assert args.listen == "0.0.0.0,::"
def test_preview_method_argument():
    """
    Test that the '--preview-method' argument:
    - Correctly converts a valid value (e.g. "latent2rgb") to a LatentPreviewMethod enum instance.
    - Causes the parser to exit with an error (SystemExit) when provided an invalid value.
    """
    valid_value = "latent2rgb"
    args = cli_args.parser.parse_args(["--preview-method", valid_value])
    assert args.preview_method == cli_args.LatentPreviewMethod.Latent2RGB
    with pytest.raises(SystemExit):
        cli_args.parser.parse_args(["--preview-method", "invalid_value"])
def test_directml_argument():
    """
    Test the '--directml' argument to ensure:
    - When provided without a value, the default const value (-1) is used.
    - When provided with an argument, the argument is correctly parsed as an integer.
    """
    args = cli_args.parser.parse_args(["--directml"])
    assert args.directml == -1
    args = cli_args.parser.parse_args(["--directml", "5"])
    assert args.directml == 5
def test_extra_model_paths_config_argument():
    """
    Test that the '--extra-model-paths-config' argument is parsed correctly.
    Verifies that:
    - When not provided, the default value is None.
    - When provided once with multiple values, the result is a nested list containing one list.
    - When provided multiple times, each occurrence is stored as a separate sublist.
    """
    args = cli_args.parser.parse_args([])
    assert args.extra_model_paths_config is None
    args = cli_args.parser.parse_args(["--extra-model-paths-config", "a.yaml", "b.yaml"])
    assert args.extra_model_paths_config == [["a.yaml", "b.yaml"]]
    args = cli_args.parser.parse_args([
        "--extra-model-paths-config", "a.yaml", "b.yaml",
        "--extra-model-paths-config", "c.yaml"
    ])
    assert args.extra_model_paths_config == [["a.yaml", "b.yaml"], ["c.yaml"]]
def test_windows_standalone_build_flag():
    """
    Test that the '--windows-standalone-build' flag correctly sets auto_launch to True,
    and that when both '--windows-standalone-build' and '--disable-auto-launch' are provided,
    auto_launch becomes False. This test manually applies the module-level post-processing logic.
    """
    def post_process_args(ns):
        if ns.windows_standalone_build:
            ns.auto_launch = True
        if ns.disable_auto_launch:
            ns.auto_launch = False
        return ns
    args = cli_args.parser.parse_args(["--windows-standalone-build"])
    args = post_process_args(args)
    assert args.auto_launch is True
    args = cli_args.parser.parse_args(["--windows-standalone-build", "--disable-auto-launch"])
    args = post_process_args(args)
    assert args.auto_launch is False
def test_verbose_argument():
    """
    Test that the '--verbose' argument works correctly:
    - When provided without a value, it should default to 'DEBUG' (using its const value).
    - When provided with an explicit value, the given value should be used.
    """
    args = cli_args.parser.parse_args(["--verbose"])
    assert args.verbose == "DEBUG"
    args = cli_args.parser.parse_args(["--verbose", "WARNING"])
    assert args.verbose == "WARNING"
def test_mutually_exclusive_cuda_malloc():
    """
    Test that providing both mutually exclusive options '--cuda-malloc' and 
    '--disable-cuda-malloc' raises a SystemExit due to the conflict.
    """
    with pytest.raises(SystemExit):
        cli_args.parser.parse_args(["--cuda-malloc", "--disable-cuda-malloc"])
def test_front_end_version_argument():
    """
    Test that the '--front-end-version' argument:
    - Defaults to "comfyanonymous/ComfyUI@latest" when not provided.
    - Accepts and correctly parses a custom version string when provided.
    """
    args = cli_args.parser.parse_args([])
    assert args.front_end_version == "comfyanonymous/ComfyUI@latest"
    custom_version = "user/custom@1.2.3"
    args = cli_args.parser.parse_args(["--front-end-version", custom_version])
    assert args.front_end_version == custom_version
def test_mutually_exclusive_fpvae_group():
    """
    Test that providing both mutually exclusive '--fp16-vae' and '--bf16-vae'
    arguments causes the parser to exit with an error.
    """
    with pytest.raises(SystemExit):
        cli_args.parser.parse_args(["--fp16-vae", "--bf16-vae"])
def test_default_values():
    """
    Test that default values for all arguments are correctly set when no arguments are provided.
    This verifies the defaults for network settings, directory paths, various flags, and numeric options.
    """
    args = cli_args.parser.parse_args([])
    assert args.listen == "127.0.0.1"
    assert args.port == 8188
    assert args.tls_keyfile is None
    assert args.tls_certfile is None
    assert args.enable_cors_header is None
    assert args.max_upload_size == 100
    assert args.base_directory is None
    assert args.output_directory is None
    assert args.temp_directory is None
    assert args.input_directory is None
    assert args.auto_launch is False
    assert args.disable_auto_launch is False
    assert args.cuda_device is None
    assert not getattr(args, 'cuda_malloc', False)
    assert not getattr(args, 'disable_cuda_malloc', False)
    assert not getattr(args, 'fp32_unet', False)
    assert not getattr(args, 'fp64_unet', False)
    assert not getattr(args, 'bf16_unet', False)
    assert not getattr(args, 'fp16_unet', False)
    assert not getattr(args, 'fp8_e4m3fn_unet', False)
    assert not getattr(args, 'fp8_e5m2_unet', False)
    assert not getattr(args, 'fp16_vae', False)
    assert not getattr(args, 'fp32_vae', False)
    assert not getattr(args, 'bf16_vae', False)
    assert not args.cpu_vae
    assert not getattr(args, 'fp8_e4m3fn_text_enc', False)
    assert not getattr(args, 'fp8_e5m2_text_enc', False)
    assert not getattr(args, 'fp16_text_enc', False)
    assert not getattr(args, 'fp32_text_enc', False)
    assert not getattr(args, 'force_upcast_attention', False)
    assert not getattr(args, 'dont_upcast_attention', False)
    assert not getattr(args, 'gpu_only', False)
    assert not getattr(args, 'highvram', False)
    assert not getattr(args, 'normalvram', False)
    assert not getattr(args, 'lowvram', False)
    assert not getattr(args, 'novram', False)
    assert not getattr(args, 'cpu', False)
    assert args.reserve_vram is None
    assert args.default_hashing_function == 'sha256'
    assert not args.disable_smart_memory
    assert not args.deterministic
    assert not args.fast
    assert args.verbose == 'INFO'
    assert not args.log_stdout
    assert args.front_end_version == "comfyanonymous/ComfyUI@latest"
    assert args.front_end_root is None
    assert args.user_directory is None
    assert not args.enable_compress_response_body
def test_oneapi_device_selector_argument():
    """
    Test that the '--oneapi-device-selector' argument is correctly parsed.
    Verifies that:
    - When not provided, the default value is None.
    - When provided with a specific string, the argument returns that string.
    """
    args = cli_args.parser.parse_args([])
    assert args.oneapi_device_selector is None, "Default for oneapi-device-selector should be None"
    test_value = "GPU0,GPU1"
    args = cli_args.parser.parse_args(["--oneapi-device-selector", test_value])
    assert args.oneapi_device_selector == test_value, f"Expected oneapi-device-selector to be {test_value}"
def test_tls_arguments():
    """
    Test that TLS related arguments are correctly parsed.
    Verifies that:
    - When provided, the '--tls-keyfile' and '--tls-certfile' arguments are correctly stored.
    """
    test_args = ["--tls-keyfile", "keyfile.pem", "--tls-certfile", "certfile.pem"]
    args = cli_args.parser.parse_args(test_args)
    assert args.tls_keyfile == "keyfile.pem"
    assert args.tls_certfile == "certfile.pem"
def test_invalid_directml_argument():
    """
    Test that providing a non-integer value for the '--directml' argument
    raises a SystemExit error due to the argparse type conversion failure.
    """
    with pytest.raises(SystemExit):
        cli_args.parser.parse_args(["--directml", "not_a_number"])
def test_mutually_exclusive_fpte_group():
    """
    Test that providing mutually exclusive text encoder precision options 
    (e.g. '--fp8_e4m3fn-text-enc' and '--fp16-text-enc') raises a SystemExit error.
    """
    with pytest.raises(SystemExit):
        cli_args.parser.parse_args(["--fp8_e4m3fn-text-enc", "--fp16-text-enc"])
def test_miscellaneous_flags():
    """
    Test that miscellaneous boolean flags (disable-metadata, disable-all-custom-nodes, multi-user, and log-stdout)
    are correctly set when provided on the command line.
    """
    args = cli_args.parser.parse_args([
        "--disable-metadata",
        "--disable-all-custom-nodes",
        "--multi-user",
        "--log-stdout"
    ])
    assert args.disable_metadata is True, "Expected --disable-metadata to set disable_metadata to True"
    assert args.disable_all_custom_nodes is True, "Expected --disable-all-custom-nodes to set disable_all_custom_nodes to True"
    assert args.multi_user is True, "Expected --multi-user to set multi_user to True"
    assert args.log_stdout is True, "Expected --log-stdout to set log_stdout to True"
def test_invalid_hashing_function_argument():
    """
    Test that providing an invalid value for the '--default-hashing-function' argument
    raises a SystemExit error due to the invalid choice.
    """
    with pytest.raises(SystemExit):
        cli_args.parser.parse_args(["--default-hashing-function", "invalid_hash"])
def test_front_end_root_argument(tmp_path):
    """
    Test that the '--front-end-root' argument correctly validates and returns 
    the provided directory path as a string.
    """
    valid_dir = str(tmp_path)
    args = cli_args.parser.parse_args(["--front-end-root", valid_dir])
    assert args.front_end_root == valid_dir
def test_enable_cors_header_argument():
    """
    Test that the '--enable-cors-header' argument is parsed correctly:
    - When provided without a value, it should use the const value "*" 
      (indicating allow all origins).
    - When provided with an explicit value, it returns that value.
    """
    args = cli_args.parser.parse_args(["--enable-cors-header"])
    assert args.enable_cors_header == "*", "Expected --enable-cors-header with no value to default to '*'"
    test_origin = "http://example.com"
    args = cli_args.parser.parse_args(["--enable-cors-header", test_origin])
    assert args.enable_cors_header == test_origin, "Expected --enable-cors-header to use the provided origin"
def test_listen_argument_with_explicit_value():
    """
    Test that providing an explicit IP address with '--listen' sets the value correctly.
    """
    test_ip = "192.168.1.100"
    args = cli_args.parser.parse_args(["--listen", test_ip])
    assert args.listen == test_ip
def test_cache_arguments():
    """
    Test that the caching arguments are correctly parsed.
    Verifies that:
    - When provided with '--cache-lru' and an integer value, the cache_lru attribute is set accordingly and cache_classic is False.
    - When provided with '--cache-classic', the cache_classic flag is True and cache_lru remains at its default value (0).
    - Providing both options simultaneously raises a SystemExit error due to mutual exclusivity.
    """
    args = cli_args.parser.parse_args(["--cache-lru", "15"])
    assert args.cache_lru == 15
    assert args.cache_classic is False
    args = cli_args.parser.parse_args(["--cache-classic"])
    assert args.cache_classic is True
    assert args.cache_lru == 0
    with pytest.raises(SystemExit):
        cli_args.parser.parse_args(["--cache-lru", "15", "--cache-classic"])
def test_invalid_port_argument():
    """
    Test that passing a non-integer value to the '--port' argument
    raises a SystemExit error due to type conversion failure.
    """
    with pytest.raises(SystemExit):
        cli_args.parser.parse_args(["--port", "not_an_integer"])
def test_mutually_exclusive_fpunet_group():
    """
    Test that providing mutually exclusive unet precision options (e.g. '--fp32-unet' and '--fp16-unet')
    raises a SystemExit error due to the conflict in the fpunet group.
    """
    with pytest.raises(SystemExit):
        cli_args.parser.parse_args(["--fp32-unet", "--fp16-unet"])
def test_mutually_exclusive_upcast_arguments():
    """
    Test the mutual exclusivity of the upcast arguments:
    --force-upcast-attention and --dont-upcast-attention.
    
    This test verifies that:
    - Providing only --force-upcast-attention sets its flag to True while --dont-upcast-attention remains False.
    - Providing only --dont-upcast-attention sets its flag to True while --force-upcast-attention remains False.
    - Providing both flags simultaneously raises a SystemExit error.
    """
    args = cli_args.parser.parse_args(["--force-upcast-attention"])
    assert getattr(args, "force_upcast_attention", False) is True
    assert getattr(args, "dont_upcast_attention", False) is False
    args = cli_args.parser.parse_args(["--dont-upcast-attention"])
    assert getattr(args, "dont_upcast_attention", False) is True
    assert getattr(args, "force_upcast_attention", False) is False
    with pytest.raises(SystemExit):
        cli_args.parser.parse_args(["--force-upcast-attention", "--dont-upcast-attention"])
def test_disable_xformers_argument():
    """
    Test that the '--disable-xformers' argument correctly sets the disable_xformers flag to True.
    """
    args = cli_args.parser.parse_args(["--disable-xformers"])
    assert args.disable_xformers is True
def test_module_args_parsing_behavior():
    """
    Test the module-level args parsing behavior.
    This test temporarily sets comfy.options.args_parsing to False and reloads the cli_args module.
    When args_parsing is False, the parser should be forced to parse an empty list (using defaults).
    We verify that the globals in cli_args.args have the expected default values.
    """
    # Save the original value
    original_args_parsing = comfy.options.args_parsing
    try:
        # Set args_parsing to False so that parser.parse_args([]) is used at the module level
        comfy.options.args_parsing = False
        reloaded_cli_args = importlib.reload(cli_args)
        # Since no arguments are provided by default, the defaults should be assigned.
        # For instance, 'listen' should be "127.0.0.1" and 'port' should be 8188.
        assert reloaded_cli_args.args.listen == "127.0.0.1", "Expected default listen to be 127.0.0.1"
        assert reloaded_cli_args.args.port == 8188, "Expected default port to be 8188"
        # Additionally, we want to check that a flag (like auto_launch) remains at its default (False)
        assert reloaded_cli_args.args.auto_launch is False, "Expected auto_launch to be False when no args provided"
    finally:
        # Restore the original args_parsing value and reload the module again to reset state.
        comfy.options.args_parsing = original_args_parsing
        importlib.reload(cli_args)  # reset to original state
def test_fpunet_valid_option():
    """
    Test that providing the '--fp16-unet' argument correctly sets the fp16_unet flag,
    and does not set any other mutually exclusive fpunet options.
    """
    args = cli_args.parser.parse_args(["--fp16-unet"])
    assert args.fp16_unet is True, "Expected --fp16-unet to set fp16_unet to True"
    assert not getattr(args, "fp32_unet", False), "Unexpected --fp32-unet flag when --fp16-unet is provided"
    assert not getattr(args, "fp64_unet", False), "Unexpected --fp64-unet flag when --fp16-unet is provided"
    assert not getattr(args, "bf16_unet", False), "Unexpected --bf16-unet flag when --fp16-unet is provided"
    assert not getattr(args, "fp8_e4m3fn_unet", False), "Unexpected --fp8_e4m3fn-unet flag when --fp16-unet is provided"
    assert not getattr(args, "fp8_e5m2_unet", False), "Unexpected --fp8_e5m2-unet flag when --fp16-unet is provided"
def test_dont_print_server_and_quick_test_for_ci():
    """
    Test that the '--dont-print-server' and '--quick-test-for-ci' flags are correctly parsed.
    Verifies that when not provided, their values default to False, and when provided,
    they are set to True.
    """
    # Test with no flags provided (should use defaults)
    args = cli_args.parser.parse_args([])
    assert args.dont_print_server is False, "Expected default dont_print_server to be False"
    assert args.quick_test_for_ci is False, "Expected default quick_test_for_ci to be False"
    
    # Test with both flags provided on the command line
    args = cli_args.parser.parse_args(["--dont-print-server", "--quick-test-for-ci"])
    assert args.dont_print_server is True, "Expected dont_print_server to be True when flag is provided"
    assert args.quick_test_for_ci is True, "Expected quick_test_for_ci to be True when flag is provided"
def test_vram_group_lowvram_flag():
    """
    Test that providing the '--lowvram' flag correctly sets the lowvram flag
    and ensures that all other mutually exclusive vram flags remain False.
    """
    args = cli_args.parser.parse_args(["--lowvram"])
    assert args.lowvram is True
    # Ensure that no other mutually exclusive vram flags are set
    assert not getattr(args, "gpu_only", False)
    assert not getattr(args, "highvram", False)
    assert not getattr(args, "normalvram", False)
    assert not getattr(args, "novram", False)
    assert not getattr(args, "cpu", False)
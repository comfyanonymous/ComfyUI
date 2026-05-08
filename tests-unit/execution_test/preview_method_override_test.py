"""
Unit tests for Queue-specific Preview Method Override feature.

Tests the preview method override functionality:
- LatentPreviewMethod.from_string() method
- set_preview_method() function in latent_preview.py
- default_preview_method variable
- Integration with args.preview_method
"""
import pytest
from comfy.cli_args import args, LatentPreviewMethod
from latent_preview import set_preview_method, default_preview_method


class TestLatentPreviewMethodFromString:
    """Test LatentPreviewMethod.from_string() classmethod."""

    @pytest.mark.parametrize("value,expected", [
        ("auto", LatentPreviewMethod.Auto),
        ("latent2rgb", LatentPreviewMethod.Latent2RGB),
        ("taesd", LatentPreviewMethod.TAESD),
        ("none", LatentPreviewMethod.NoPreviews),
    ])
    def test_valid_values_return_enum(self, value, expected):
        """Valid string values should return corresponding enum."""
        assert LatentPreviewMethod.from_string(value) == expected

    @pytest.mark.parametrize("invalid", [
        "invalid",
        "TAESD",      # Case sensitive
        "AUTO",       # Case sensitive
        "Latent2RGB", # Case sensitive
        "latent",
        "",
        "default",    # default is special, not a method
    ])
    def test_invalid_values_return_none(self, invalid):
        """Invalid string values should return None."""
        assert LatentPreviewMethod.from_string(invalid) is None


class TestLatentPreviewMethodEnumValues:
    """Test LatentPreviewMethod enum has expected values."""

    def test_enum_values(self):
        """Verify enum values match expected strings."""
        assert LatentPreviewMethod.NoPreviews.value == "none"
        assert LatentPreviewMethod.Auto.value == "auto"
        assert LatentPreviewMethod.Latent2RGB.value == "latent2rgb"
        assert LatentPreviewMethod.TAESD.value == "taesd"

    def test_enum_count(self):
        """Verify exactly 4 preview methods exist."""
        assert len(LatentPreviewMethod) == 4


class TestSetPreviewMethod:
    """Test set_preview_method() function from latent_preview.py."""

    def setup_method(self):
        """Store original value before each test."""
        self.original = args.preview_method

    def teardown_method(self):
        """Restore original value after each test."""
        args.preview_method = self.original

    def test_override_with_taesd(self):
        """'taesd' should set args.preview_method to TAESD."""
        set_preview_method("taesd")
        assert args.preview_method == LatentPreviewMethod.TAESD

    def test_override_with_latent2rgb(self):
        """'latent2rgb' should set args.preview_method to Latent2RGB."""
        set_preview_method("latent2rgb")
        assert args.preview_method == LatentPreviewMethod.Latent2RGB

    def test_override_with_auto(self):
        """'auto' should set args.preview_method to Auto."""
        set_preview_method("auto")
        assert args.preview_method == LatentPreviewMethod.Auto

    def test_override_with_none_value(self):
        """'none' should set args.preview_method to NoPreviews."""
        set_preview_method("none")
        assert args.preview_method == LatentPreviewMethod.NoPreviews

    def test_default_restores_original(self):
        """'default' should restore to default_preview_method."""
        # First override to something else
        set_preview_method("taesd")
        assert args.preview_method == LatentPreviewMethod.TAESD

        # Then use 'default' to restore
        set_preview_method("default")
        assert args.preview_method == default_preview_method

    def test_none_param_restores_original(self):
        """None parameter should restore to default_preview_method."""
        # First override to something else
        set_preview_method("taesd")
        assert args.preview_method == LatentPreviewMethod.TAESD

        # Then use None to restore
        set_preview_method(None)
        assert args.preview_method == default_preview_method

    def test_empty_string_restores_original(self):
        """Empty string should restore to default_preview_method."""
        set_preview_method("taesd")
        set_preview_method("")
        assert args.preview_method == default_preview_method

    def test_invalid_value_restores_original(self):
        """Invalid value should restore to default_preview_method."""
        set_preview_method("taesd")
        set_preview_method("invalid_method")
        assert args.preview_method == default_preview_method

    def test_case_sensitive_invalid_restores(self):
        """Case-mismatched values should restore to default."""
        set_preview_method("taesd")
        set_preview_method("TAESD")  # Wrong case
        assert args.preview_method == default_preview_method


class TestDefaultPreviewMethod:
    """Test default_preview_method module variable."""

    def test_default_is_not_none(self):
        """default_preview_method should not be None."""
        assert default_preview_method is not None

    def test_default_is_enum_member(self):
        """default_preview_method should be a LatentPreviewMethod enum."""
        assert isinstance(default_preview_method, LatentPreviewMethod)

    def test_default_matches_args_initial(self):
        """default_preview_method should match CLI default or user setting."""
        # This tests that default_preview_method was captured at module load
        # After set_preview_method(None), args should equal default
        original = args.preview_method
        set_preview_method("taesd")
        set_preview_method(None)
        assert args.preview_method == default_preview_method
        args.preview_method = original


class TestArgsPreviewMethodModification:
    """Test args.preview_method can be modified correctly."""

    def setup_method(self):
        """Store original value before each test."""
        self.original = args.preview_method

    def teardown_method(self):
        """Restore original value after each test."""
        args.preview_method = self.original

    def test_args_accepts_all_enum_values(self):
        """args.preview_method should accept all LatentPreviewMethod values."""
        for method in LatentPreviewMethod:
            args.preview_method = method
            assert args.preview_method == method

    def test_args_modification_and_restoration(self):
        """args.preview_method should be modifiable and restorable."""
        original = args.preview_method

        args.preview_method = LatentPreviewMethod.TAESD
        assert args.preview_method == LatentPreviewMethod.TAESD

        args.preview_method = original
        assert args.preview_method == original


class TestExecutionFlow:
    """Test the execution flow pattern used in execution.py."""

    def setup_method(self):
        """Store original value before each test."""
        self.original = args.preview_method

    def teardown_method(self):
        """Restore original value after each test."""
        args.preview_method = self.original

    def test_sequential_executions_with_different_methods(self):
        """Simulate multiple queue executions with different preview methods."""
        # Execution 1: taesd
        set_preview_method("taesd")
        assert args.preview_method == LatentPreviewMethod.TAESD

        # Execution 2: none
        set_preview_method("none")
        assert args.preview_method == LatentPreviewMethod.NoPreviews

        # Execution 3: default (restore)
        set_preview_method("default")
        assert args.preview_method == default_preview_method

        # Execution 4: auto
        set_preview_method("auto")
        assert args.preview_method == LatentPreviewMethod.Auto

        # Execution 5: no override (None)
        set_preview_method(None)
        assert args.preview_method == default_preview_method

    def test_override_then_default_pattern(self):
        """Test the pattern: override -> execute -> next call restores."""
        # First execution with override
        set_preview_method("latent2rgb")
        assert args.preview_method == LatentPreviewMethod.Latent2RGB

        # Second execution without override restores default
        set_preview_method(None)
        assert args.preview_method == default_preview_method

    def test_extra_data_simulation(self):
        """Simulate extra_data.get('preview_method') patterns."""
        # Simulate: extra_data = {"preview_method": "taesd"}
        extra_data = {"preview_method": "taesd"}
        set_preview_method(extra_data.get("preview_method"))
        assert args.preview_method == LatentPreviewMethod.TAESD

        # Simulate: extra_data = {}
        extra_data = {}
        set_preview_method(extra_data.get("preview_method"))
        assert args.preview_method == default_preview_method

        # Simulate: extra_data = {"preview_method": "default"}
        extra_data = {"preview_method": "default"}
        set_preview_method(extra_data.get("preview_method"))
        assert args.preview_method == default_preview_method


class TestRealWorldScenarios:
    """Tests using real-world prompt data patterns."""

    def setup_method(self):
        """Store original value before each test."""
        self.original = args.preview_method

    def teardown_method(self):
        """Restore original value after each test."""
        args.preview_method = self.original

    def test_captured_prompt_without_preview_method(self):
        """
        Test with captured prompt that has no preview_method.
        Based on: tests-unit/execution_test/fixtures/default_prompt.json
        """
        # Real captured extra_data structure (preview_method absent)
        extra_data = {
            "extra_pnginfo": {"workflow": {}},
            "client_id": "271314f0dabd48e5aaa488ed7a4ceb0d",
            "create_time": 1765416558179
        }

        set_preview_method(extra_data.get("preview_method"))
        assert args.preview_method == default_preview_method

    def test_captured_prompt_with_preview_method_taesd(self):
        """Test captured prompt with preview_method: taesd."""
        extra_data = {
            "extra_pnginfo": {"workflow": {}},
            "client_id": "271314f0dabd48e5aaa488ed7a4ceb0d",
            "preview_method": "taesd"
        }

        set_preview_method(extra_data.get("preview_method"))
        assert args.preview_method == LatentPreviewMethod.TAESD

    def test_captured_prompt_with_preview_method_none(self):
        """Test captured prompt with preview_method: none (disable preview)."""
        extra_data = {
            "extra_pnginfo": {"workflow": {}},
            "client_id": "test-client",
            "preview_method": "none"
        }

        set_preview_method(extra_data.get("preview_method"))
        assert args.preview_method == LatentPreviewMethod.NoPreviews

    def test_captured_prompt_with_preview_method_latent2rgb(self):
        """Test captured prompt with preview_method: latent2rgb."""
        extra_data = {
            "extra_pnginfo": {"workflow": {}},
            "client_id": "test-client",
            "preview_method": "latent2rgb"
        }

        set_preview_method(extra_data.get("preview_method"))
        assert args.preview_method == LatentPreviewMethod.Latent2RGB

    def test_captured_prompt_with_preview_method_auto(self):
        """Test captured prompt with preview_method: auto."""
        extra_data = {
            "extra_pnginfo": {"workflow": {}},
            "client_id": "test-client",
            "preview_method": "auto"
        }

        set_preview_method(extra_data.get("preview_method"))
        assert args.preview_method == LatentPreviewMethod.Auto

    def test_captured_prompt_with_preview_method_default(self):
        """Test captured prompt with preview_method: default (use CLI setting)."""
        # First set to something else
        set_preview_method("taesd")
        assert args.preview_method == LatentPreviewMethod.TAESD

        # Then simulate a prompt with "default"
        extra_data = {
            "extra_pnginfo": {"workflow": {}},
            "client_id": "test-client",
            "preview_method": "default"
        }

        set_preview_method(extra_data.get("preview_method"))
        assert args.preview_method == default_preview_method

    def test_sequential_queue_with_different_preview_methods(self):
        """
        Simulate real queue scenario: multiple prompts with different settings.
        This tests the actual usage pattern in ComfyUI.
        """
        # Queue 1: User wants TAESD preview
        extra_data_1 = {"client_id": "client-1", "preview_method": "taesd"}
        set_preview_method(extra_data_1.get("preview_method"))
        assert args.preview_method == LatentPreviewMethod.TAESD

        # Queue 2: User wants no preview (faster execution)
        extra_data_2 = {"client_id": "client-2", "preview_method": "none"}
        set_preview_method(extra_data_2.get("preview_method"))
        assert args.preview_method == LatentPreviewMethod.NoPreviews

        # Queue 3: User doesn't specify (use server default)
        extra_data_3 = {"client_id": "client-3"}
        set_preview_method(extra_data_3.get("preview_method"))
        assert args.preview_method == default_preview_method

        # Queue 4: User explicitly wants default
        extra_data_4 = {"client_id": "client-4", "preview_method": "default"}
        set_preview_method(extra_data_4.get("preview_method"))
        assert args.preview_method == default_preview_method

        # Queue 5: User wants latent2rgb
        extra_data_5 = {"client_id": "client-5", "preview_method": "latent2rgb"}
        set_preview_method(extra_data_5.get("preview_method"))
        assert args.preview_method == LatentPreviewMethod.Latent2RGB

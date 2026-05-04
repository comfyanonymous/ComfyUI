"""Tests for feature flags functionality."""

from comfy_api.feature_flags import (
    get_connection_feature,
    supports_feature,
    get_server_features,
    SERVER_FEATURE_FLAGS,
)


class TestFeatureFlags:
    """Test suite for feature flags functions."""

    def test_get_server_features_returns_copy(self):
        """Test that get_server_features returns a copy of the server flags."""
        features = get_server_features()
        # Verify it's a copy by modifying it
        features["test_flag"] = True
        # Original should be unchanged
        assert "test_flag" not in SERVER_FEATURE_FLAGS

    def test_get_server_features_contains_expected_flags(self):
        """Test that server features contain expected flags."""
        features = get_server_features()
        assert "supports_preview_metadata" in features
        assert features["supports_preview_metadata"] is True
        assert "max_upload_size" in features
        assert isinstance(features["max_upload_size"], (int, float))

    def test_get_connection_feature_with_missing_sid(self):
        """Test getting feature for non-existent session ID."""
        sockets_metadata = {}
        result = get_connection_feature(sockets_metadata, "missing_sid", "some_feature")
        assert result is False  # Default value

    def test_get_connection_feature_with_custom_default(self):
        """Test getting feature with custom default value."""
        sockets_metadata = {}
        result = get_connection_feature(
            sockets_metadata, "missing_sid", "some_feature", default="custom_default"
        )
        assert result == "custom_default"

    def test_get_connection_feature_with_feature_flags(self):
        """Test getting feature from connection with feature flags."""
        sockets_metadata = {
            "sid1": {
                "feature_flags": {
                    "supports_preview_metadata": True,
                    "custom_feature": "value",
                },
            }
        }
        result = get_connection_feature(sockets_metadata, "sid1", "supports_preview_metadata")
        assert result is True

        result = get_connection_feature(sockets_metadata, "sid1", "custom_feature")
        assert result == "value"

    def test_get_connection_feature_missing_feature(self):
        """Test getting non-existent feature from connection."""
        sockets_metadata = {
            "sid1": {"feature_flags": {"existing_feature": True}}
        }
        result = get_connection_feature(sockets_metadata, "sid1", "missing_feature")
        assert result is False

    def test_supports_feature_returns_boolean(self):
        """Test that supports_feature always returns boolean."""
        sockets_metadata = {
            "sid1": {
                "feature_flags": {
                    "bool_feature": True,
                    "string_feature": "value",
                    "none_feature": None,
                },
            }
        }

        # True boolean feature
        assert supports_feature(sockets_metadata, "sid1", "bool_feature") is True

        # Non-boolean values should return False
        assert supports_feature(sockets_metadata, "sid1", "string_feature") is False
        assert supports_feature(sockets_metadata, "sid1", "none_feature") is False
        assert supports_feature(sockets_metadata, "sid1", "missing_feature") is False

    def test_supports_feature_with_missing_connection(self):
        """Test supports_feature with missing connection."""
        sockets_metadata = {}
        assert supports_feature(sockets_metadata, "missing_sid", "any_feature") is False

    def test_empty_feature_flags_dict(self):
        """Test connection with empty feature flags dictionary."""
        sockets_metadata = {"sid1": {"feature_flags": {}}}
        result = get_connection_feature(sockets_metadata, "sid1", "any_feature")
        assert result is False
        assert supports_feature(sockets_metadata, "sid1", "any_feature") is False

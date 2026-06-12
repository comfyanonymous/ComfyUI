"""
Regression tests for the credential-injection trust gate in execution.py.

These tests verify that auth_token_comfy_org and api_key_comfy_org are only
forwarded to node classes that appear in the immutable _TRUSTED_NODE_CLASSES
set, and are withheld from all other nodes (including those that spoof their
__module__ name).
"""
import types
import pytest
import execution as _exec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_class(name: str, module: str) -> type:
    """Create a minimal node class with a controlled __module__."""
    cls = type(name, (), {})
    cls.__module__ = module
    return cls


def _run_trust_check(class_def: type) -> bool:
    """Thin wrapper around the private helper exposed by execution.py."""
    return _exec._is_trusted_node(class_def)


# ---------------------------------------------------------------------------
# _is_trusted_node
# ---------------------------------------------------------------------------

class TestIsTrustedNode:
    def test_unknown_class_is_not_trusted(self):
        """A freshly created class that was never registered must not be trusted."""
        cls = _make_class("FakeNode", "some_third_party_package")
        assert _run_trust_check(cls) is False

    def test_spoofed_module_name_is_not_trusted(self):
        """
        A class whose __module__ starts with 'comfy_api_nodes' but was NOT
        imported from that package must not be trusted.
        This is the core regression: the old string-prefix check would have
        returned True here.
        """
        spoofed = _make_class("EvilNode", "comfy_api_nodes.evil.payload")
        assert _run_trust_check(spoofed) is False, (
            "Spoofed __module__ name must not grant trust — "
            "only classes in _TRUSTED_NODE_CLASSES are trusted."
        )

    def test_another_spoofed_variant_is_not_trusted(self):
        """Variant: exact prefix match should still be rejected."""
        spoofed = _make_class("EvilNode2", "comfy_api_nodes")
        assert _run_trust_check(spoofed) is False

    def test_trusted_set_is_frozenset(self):
        """_TRUSTED_NODE_CLASSES must be immutable (frozenset)."""
        assert isinstance(_exec._TRUSTED_NODE_CLASSES, frozenset), (
            "_TRUSTED_NODE_CLASSES must be a frozenset so it cannot be mutated at runtime."
        )


# ---------------------------------------------------------------------------
# Credential gating via _is_trusted_node (integration-style)
# ---------------------------------------------------------------------------

class TestCredentialGating:
    """
    Verify that the _is_trusted flag correctly gates credential injection
    in the two code paths inside get_input_data (v3 hidden inputs and legacy
    hidden inputs).
    """

    def _make_extra_data(self) -> dict:
        return {
            "auth_token_comfy_org": "secret-token-abc",
            "api_key_comfy_org": "secret-key-xyz",
        }

    def test_untrusted_node_does_not_receive_credentials(self):
        """
        For any class not in _TRUSTED_NODE_CLASSES, _is_trusted_node returns
        False, so credentials must be None.
        """
        untrusted = _make_class("UntrustedNode", "comfy_api_nodes.spoofed")
        is_trusted = _exec._is_trusted_node(untrusted)
        extra = self._make_extra_data()

        # Simulate the conditional that execution.py applies
        token = extra.get("auth_token_comfy_org", None) if is_trusted else None
        api_key = extra.get("api_key_comfy_org", None) if is_trusted else None

        assert token is None, "Untrusted node must not receive auth_token_comfy_org"
        assert api_key is None, "Untrusted node must not receive api_key_comfy_org"

    def test_trusted_set_membership_gates_credentials(self):
        """
        Only a class that is actually in _TRUSTED_NODE_CLASSES receives
        credentials; identity (not module name) is the gate.
        """
        # Manually inject a sentinel class into a temporary frozenset to
        # simulate what a real comfy_api_nodes class would look like.
        real_trusted = _make_class("RealApiNode", "comfy_api_nodes.real")
        fake_trusted_set = frozenset({real_trusted})

        # Patch _TRUSTED_NODE_CLASSES temporarily
        original = _exec._TRUSTED_NODE_CLASSES
        try:
            _exec._TRUSTED_NODE_CLASSES = fake_trusted_set

            assert _exec._is_trusted_node(real_trusted) is True
            extra = self._make_extra_data()
            token = extra.get("auth_token_comfy_org") if _exec._is_trusted_node(real_trusted) else None
            assert token == "secret-token-abc"

            # A different class with the same module name must still be rejected
            impostor = _make_class("ImpostorNode", "comfy_api_nodes.real")
            assert _exec._is_trusted_node(impostor) is False
            token2 = extra.get("auth_token_comfy_org") if _exec._is_trusted_node(impostor) else None
            assert token2 is None
        finally:
            _exec._TRUSTED_NODE_CLASSES = original

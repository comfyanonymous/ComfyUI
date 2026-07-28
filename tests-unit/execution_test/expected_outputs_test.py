"""Unit tests for the expected_outputs feature.

This feature allows nodes to know at runtime which outputs are connected downstream,
enabling them to skip computing outputs that aren't needed.
"""

from comfy_api.latest import IO
from comfy_execution.graph import DynamicPrompt, get_expected_outputs_for_node
from comfy_execution.utils import (
    CurrentNodeContext,
    ExecutionContext,
    get_executing_context,
    is_output_needed,
)


class TestGetExpectedOutputsForNode:
    """Tests for get_expected_outputs_for_node() function."""

    def test_single_output_connected(self):
        """Test node with single output connected to one downstream node."""
        prompt = {
            "1": {"class_type": "SourceNode", "inputs": {}},
            "2": {"class_type": "ConsumerNode", "inputs": {"image": ["1", 0]}},
        }
        dynprompt = DynamicPrompt(prompt)
        expected = get_expected_outputs_for_node(dynprompt, "1")
        assert expected == frozenset({0})

    def test_multiple_outputs_partial_connected(self):
        """Test node with multiple outputs, only some connected."""
        prompt = {
            "1": {"class_type": "MultiOutputNode", "inputs": {}},
            "2": {"class_type": "ConsumerA", "inputs": {"input": ["1", 0]}},
            # Output 1 is not connected
            "3": {"class_type": "ConsumerC", "inputs": {"input": ["1", 2]}},
        }
        dynprompt = DynamicPrompt(prompt)
        expected = get_expected_outputs_for_node(dynprompt, "1")
        assert expected == frozenset({0, 2})
        assert 1 not in expected  # Output 1 is definitely unused

    def test_no_outputs_connected(self):
        """Test node with no outputs connected."""
        prompt = {
            "1": {"class_type": "SourceNode", "inputs": {}},
            "2": {"class_type": "OtherNode", "inputs": {}},
        }
        dynprompt = DynamicPrompt(prompt)
        expected = get_expected_outputs_for_node(dynprompt, "1")
        assert expected == frozenset()

    def test_same_output_connected_multiple_times(self):
        """Test same output connected to multiple downstream nodes."""
        prompt = {
            "1": {"class_type": "SourceNode", "inputs": {}},
            "2": {"class_type": "ConsumerA", "inputs": {"input": ["1", 0]}},
            "3": {"class_type": "ConsumerB", "inputs": {"input": ["1", 0]}},
            "4": {"class_type": "ConsumerC", "inputs": {"input": ["1", 0]}},
        }
        dynprompt = DynamicPrompt(prompt)
        expected = get_expected_outputs_for_node(dynprompt, "1")
        assert expected == frozenset({0})

    def test_node_not_in_prompt(self):
        """Test getting expected outputs for a node not in the prompt."""
        prompt = {
            "1": {"class_type": "SourceNode", "inputs": {}},
        }
        dynprompt = DynamicPrompt(prompt)
        expected = get_expected_outputs_for_node(dynprompt, "999")
        assert expected == frozenset()

    def test_chained_nodes(self):
        """Test expected outputs in a chain of nodes."""
        prompt = {
            "1": {"class_type": "SourceNode", "inputs": {}},
            "2": {"class_type": "MiddleNode", "inputs": {"input": ["1", 0]}},
            "3": {"class_type": "EndNode", "inputs": {"input": ["2", 0]}},
        }
        dynprompt = DynamicPrompt(prompt)

        # Node 1's output 0 is connected to node 2
        expected_1 = get_expected_outputs_for_node(dynprompt, "1")
        assert expected_1 == frozenset({0})

        # Node 2's output 0 is connected to node 3
        expected_2 = get_expected_outputs_for_node(dynprompt, "2")
        assert expected_2 == frozenset({0})

        # Node 3 has no downstream connections
        expected_3 = get_expected_outputs_for_node(dynprompt, "3")
        assert expected_3 == frozenset()

    def test_complex_graph(self):
        """Test expected outputs in a complex graph with multiple connections."""
        prompt = {
            "1": {"class_type": "MultiOutputNode", "inputs": {}},
            "2": {"class_type": "ProcessorA", "inputs": {"image": ["1", 0], "mask": ["1", 1]}},
            "3": {"class_type": "ProcessorB", "inputs": {"data": ["1", 2]}},
            "4": {"class_type": "Combiner", "inputs": {"a": ["2", 0], "b": ["3", 0]}},
        }
        dynprompt = DynamicPrompt(prompt)

        # Node 1 has outputs 0, 1, 2 all connected
        expected = get_expected_outputs_for_node(dynprompt, "1")
        assert expected == frozenset({0, 1, 2})

    def test_constant_inputs_ignored(self):
        """Test that constant (non-link) inputs don't affect expected outputs."""
        prompt = {
            "1": {"class_type": "SourceNode", "inputs": {}},
            "2": {
                "class_type": "ConsumerNode",
                "inputs": {
                    "image": ["1", 0],
                    "value": 42,
                    "name": "test",
                },
            },
        }
        dynprompt = DynamicPrompt(prompt)
        expected = get_expected_outputs_for_node(dynprompt, "1")
        assert expected == frozenset({0})

    def test_ephemeral_node_invalidates_cache(self):
        """Test that adding ephemeral nodes updates expected outputs."""
        prompt = {
            "1": {"class_type": "SourceNode", "inputs": {}},
            "2": {"class_type": "ConsumerNode", "inputs": {"image": ["1", 0]}},
        }
        dynprompt = DynamicPrompt(prompt)

        # Initially only output 0 is connected
        expected = get_expected_outputs_for_node(dynprompt, "1")
        assert expected == frozenset({0})

        # Add an ephemeral node that connects to output 1
        dynprompt.add_ephemeral_node(
            "eph_1",
            {"class_type": "EphemeralNode", "inputs": {"data": ["1", 1]}},
            parent_id="2",
            display_id="2",
        )

        # Now both outputs 0 and 1 should be expected
        expected = get_expected_outputs_for_node(dynprompt, "1")
        assert expected == frozenset({0, 1})


class TestExternalOutputConsumers:
    """Tests for DynamicPrompt.add_output_consumer() — out-of-band consumers
    (subgraph expansion output mappings) that have no input link in the prompt."""

    def test_external_consumer_only(self):
        """A socket consumed only externally must appear in expected outputs."""
        prompt = {
            "1": {"class_type": "SourceNode", "inputs": {}},
        }
        dynprompt = DynamicPrompt(prompt)
        assert get_expected_outputs_for_node(dynprompt, "1") == frozenset()

        dynprompt.add_output_consumer("1", 1)
        assert get_expected_outputs_for_node(dynprompt, "1") == frozenset({1})

    def test_external_consumer_merges_with_links(self):
        """External consumers merge with input-link consumers."""
        prompt = {
            "1": {"class_type": "SourceNode", "inputs": {}},
            "2": {"class_type": "ConsumerNode", "inputs": {"image": ["1", 0]}},
        }
        dynprompt = DynamicPrompt(prompt)
        dynprompt.add_output_consumer("1", 2)
        assert get_expected_outputs_for_node(dynprompt, "1") == frozenset({0, 2})

    def test_external_consumer_invalidates_cached_map(self):
        """Registering after the map was built must invalidate the cache."""
        prompt = {
            "1": {"class_type": "SourceNode", "inputs": {}},
            "2": {"class_type": "ConsumerNode", "inputs": {"image": ["1", 0]}},
        }
        dynprompt = DynamicPrompt(prompt)
        # Build (and cache) the map first
        assert get_expected_outputs_for_node(dynprompt, "1") == frozenset({0})

        dynprompt.add_output_consumer("1", 1)
        assert get_expected_outputs_for_node(dynprompt, "1") == frozenset({0, 1})


class TestExecutionContext:
    """Tests for ExecutionContext with expected_outputs field."""

    def test_context_with_expected_outputs(self):
        """Test creating ExecutionContext with expected_outputs."""
        ctx = ExecutionContext(
            prompt_id="prompt-123", node_id="node-456", list_index=0, expected_outputs=frozenset({0, 2})
        )
        assert ctx.prompt_id == "prompt-123"
        assert ctx.node_id == "node-456"
        assert ctx.list_index == 0
        assert ctx.expected_outputs == frozenset({0, 2})

    def test_context_without_expected_outputs(self):
        """Test ExecutionContext defaults to None for expected_outputs."""
        ctx = ExecutionContext(prompt_id="prompt-123", node_id="node-456", list_index=0)
        assert ctx.expected_outputs is None

    def test_context_empty_expected_outputs(self):
        """Test ExecutionContext with empty expected_outputs set."""
        ctx = ExecutionContext(
            prompt_id="prompt-123", node_id="node-456", list_index=None, expected_outputs=frozenset()
        )
        assert ctx.expected_outputs == frozenset()
        assert len(ctx.expected_outputs) == 0


class TestCurrentNodeContext:
    """Tests for CurrentNodeContext context manager with expected_outputs."""

    def test_context_manager_with_expected_outputs(self):
        """Test CurrentNodeContext sets and resets context correctly."""
        assert get_executing_context() is None

        with CurrentNodeContext("prompt-1", "node-1", 0, frozenset({0, 1})):
            ctx = get_executing_context()
            assert ctx is not None
            assert ctx.prompt_id == "prompt-1"
            assert ctx.node_id == "node-1"
            assert ctx.list_index == 0
            assert ctx.expected_outputs == frozenset({0, 1})

        assert get_executing_context() is None

    def test_context_manager_without_expected_outputs(self):
        """Test CurrentNodeContext works without expected_outputs (backwards compatible)."""
        with CurrentNodeContext("prompt-1", "node-1"):
            ctx = get_executing_context()
            assert ctx is not None
            assert ctx.expected_outputs is None

    def test_nested_context_managers(self):
        """Test nested CurrentNodeContext managers."""
        with CurrentNodeContext("prompt-1", "node-1", 0, frozenset({0})):
            ctx1 = get_executing_context()
            assert ctx1.expected_outputs == frozenset({0})

            with CurrentNodeContext("prompt-1", "node-2", 0, frozenset({1, 2})):
                ctx2 = get_executing_context()
                assert ctx2.expected_outputs == frozenset({1, 2})
                assert ctx2.node_id == "node-2"

            # After inner context exits, should be back to outer context
            ctx1_again = get_executing_context()
            assert ctx1_again.expected_outputs == frozenset({0})
            assert ctx1_again.node_id == "node-1"

    def test_output_check_pattern(self):
        """Test the typical pattern nodes will use to check expected outputs."""
        with CurrentNodeContext("prompt-1", "node-1", 0, frozenset({0, 2})):
            ctx = get_executing_context()

            # Typical usage pattern
            if ctx and ctx.expected_outputs is not None:
                should_compute_0 = 0 in ctx.expected_outputs
                should_compute_1 = 1 in ctx.expected_outputs
                should_compute_2 = 2 in ctx.expected_outputs
            else:
                # Fallback when info not available
                should_compute_0 = should_compute_1 = should_compute_2 = True

            assert should_compute_0 is True
            assert should_compute_1 is False  # Not in expected_outputs
            assert should_compute_2 is True


class TestSchemaLazyOutputs:
    """Tests for lazy_outputs in V3 Schema."""

    def test_schema_lazy_outputs_default(self):
        """Test that lazy_outputs defaults to False."""
        schema = IO.Schema(
            node_id="TestNode",
            inputs=[],
            outputs=[IO.Float.Output()],
        )
        assert schema.lazy_outputs is False

    def test_schema_lazy_outputs_true(self):
        """Test setting lazy_outputs to True."""
        schema = IO.Schema(
            node_id="TestNode",
            lazy_outputs=True,
            inputs=[],
            outputs=[IO.Float.Output()],
        )
        assert schema.lazy_outputs is True

    def test_v3_node_lazy_outputs_property(self):
        """Test that LAZY_OUTPUTS property works on V3 nodes."""

        class TestNodeWithLazyOutputs(IO.ComfyNode):
            @classmethod
            def define_schema(cls):
                return IO.Schema(
                    node_id="TestNodeWithLazyOutputs",
                    lazy_outputs=True,
                    inputs=[],
                    outputs=[IO.Float.Output()],
                )

            @classmethod
            def execute(cls):
                return IO.NodeOutput(1.0)

        assert TestNodeWithLazyOutputs.LAZY_OUTPUTS is True

    def test_v3_node_lazy_outputs_default(self):
        """Test that LAZY_OUTPUTS defaults to False on V3 nodes."""

        class TestNodeWithoutLazyOutputs(IO.ComfyNode):
            @classmethod
            def define_schema(cls):
                return IO.Schema(
                    node_id="TestNodeWithoutLazyOutputs",
                    inputs=[],
                    outputs=[IO.Float.Output()],
                )

            @classmethod
            def execute(cls):
                return IO.NodeOutput(1.0)

        assert TestNodeWithoutLazyOutputs.LAZY_OUTPUTS is False


class TestIsOutputNeeded:
    """Tests for is_output_needed() helper function."""

    def test_output_needed_when_in_expected(self):
        """Test that output is needed when in expected_outputs."""
        with CurrentNodeContext("prompt-1", "node-1", 0, frozenset({0, 2})):
            assert is_output_needed(0) is True
            assert is_output_needed(2) is True

    def test_output_not_needed_when_not_in_expected(self):
        """Test that output is not needed when not in expected_outputs."""
        with CurrentNodeContext("prompt-1", "node-1", 0, frozenset({0, 2})):
            assert is_output_needed(1) is False
            assert is_output_needed(3) is False

    def test_output_needed_when_no_context(self):
        """Test that output is needed when no context."""
        assert get_executing_context() is None
        assert is_output_needed(0) is True
        assert is_output_needed(1) is True

    def test_output_needed_when_expected_outputs_is_none(self):
        """Test that output is needed when expected_outputs is None."""
        with CurrentNodeContext("prompt-1", "node-1", 0, None):
            assert is_output_needed(0) is True
            assert is_output_needed(1) is True

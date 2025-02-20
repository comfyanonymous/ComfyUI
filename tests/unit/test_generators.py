import pytest

from comfy_extras.nodes.nodes_generators import IntRange, FloatRange1, FloatRange2, FloatRange3, StringSplit


def test_int_range():
    node = IntRange()

    # Basic range
    result = node.execute(0, 5, 1)
    assert result == ([0, 1, 2, 3, 4],)

    # Negative step
    result = node.execute(5, 0, -1)
    assert result == ([5, 4, 3, 2, 1],)

    # Empty range (step in wrong direction)
    result = node.execute(0, 5, -1)
    assert result == ([],)

    # Single item range
    result = node.execute(0, 1, 1)
    assert result == ([0],)

    # Step size > 1
    result = node.execute(0, 10, 2)
    assert result == ([0, 2, 4, 6, 8],)


def test_float_range1():
    node = FloatRange1()

    # Basic range
    result = node.execute(0.0, 1.0, 0.25)
    assert result == ([0.0, 0.25, 0.5, 0.75],)

    # Step size of 1
    result = node.execute(0.0, 3.0, 1.0)
    assert result == ([0.0, 1.0, 2.0],)

    # Zero step
    result = node.execute(0.0, 1.0, 0.0)
    assert result == ([],)

    # Negative step
    result = node.execute(1.0, 0.0, -0.25)
    assert result == ([1.0, 0.75, 0.5, 0.25],)

    # Test floating point precision
    result = node.execute(0.0, 0.3, 0.1)
    assert len(result[0]) == 3
    assert all(abs(a - b) < 1e-10 for a, b in zip(result[0], [0.0, 0.1, 0.2]))


def test_float_range2():
    node = FloatRange2()

    # Basic range with 5 points
    result = node.execute(0.0, 1.0, 5)
    assert result == ([0.0, 0.25, 0.5, 0.75, 1.0],)

    # Zero points
    result = node.execute(0.0, 1.0, 0)
    assert result == ([],)

    # One point
    result = node.execute(0.0, 1.0, 1)
    assert result == ([0.0],)

    # Two points
    result = node.execute(0.0, 1.0, 2)
    assert result == ([0.0, 1.0],)

    # Test negative range
    result = node.execute(1.0, -1.0, 3)
    assert result == ([1.0, 0.0, -1.0],)

    # Test floating point precision
    result = node.execute(0.0, 0.2, 3)
    assert len(result[0]) == 3
    assert all(abs(a - b) < 1e-10 for a, b in zip(result[0], [0.0, 0.1, 0.2]))


def test_float_range3():
    node = FloatRange3()

    # Basic range with 4 spans
    result = node.execute(0.0, 1.0, 4)
    expected = [0.0, 0.25, 0.5, 0.75]  # Note: doesn't include end point
    assert result == (expected,)

    # Zero spans
    result = node.execute(0.0, 1.0, 0)
    assert result == ([],)

    # One span
    result = node.execute(0.0, 1.0, 1)
    assert result == ([0.0],)

    # Test negative range
    result = node.execute(1.0, -1.0, 2)
    assert result == ([1.0, 0.0],)

    # Test floating point precision
    result = node.execute(0.0, 0.3, 3)
    assert len(result[0]) == 3
    assert all(abs(a - b) < 1e-10 for a, b in zip(result[0], [0.0, 0.1, 0.2]))


def test_output_types():
    """Test that all nodes return correct types and list outputs"""
    # IntRange
    result = IntRange().execute(0, 5, 1)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, int) for x in result[0])

    # FloatRange1
    result = FloatRange1().execute(0.0, 1.0, 0.5)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, float) for x in result[0])

    # FloatRange2
    result = FloatRange2().execute(0.0, 1.0, 3)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, float) for x in result[0])

    # FloatRange3
    result = FloatRange3().execute(0.0, 1.0, 2)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, float) for x in result[0])


@pytest.mark.parametrize("node_class", [IntRange, FloatRange1, FloatRange2, FloatRange3])
def test_node_metadata(node_class):
    """Test that all nodes have correct metadata"""
    node = node_class()

    # Check required class attributes
    assert hasattr(node, 'CATEGORY')
    assert hasattr(node, 'OUTPUT_IS_LIST')
    assert hasattr(node, 'RETURN_TYPES')
    assert hasattr(node, 'FUNCTION')

    # Verify OUTPUT_IS_LIST matches number of outputs
    assert len(node.OUTPUT_IS_LIST) == len(node.RETURN_TYPES)
    assert node.OUTPUT_IS_LIST == (True,)

    # Verify input types exist
    input_types = node.INPUT_TYPES()
    assert 'required' in input_types

    # All nodes should have appropriate numeric types
    if node_class == IntRange:
        assert all(v[0] == 'INT' for v in input_types['required'].values())
    else:
        assert input_types['required']['start'][0] == 'FLOAT'
        assert input_types['required']['end'][0] == 'FLOAT'


def test_string_split_basic():
    node = StringSplit()

    # Basic comma split
    result = node.execute("a,b,c")
    assert result == (["a", "b", "c"],)

    # Custom delimiter
    result = node.execute("a|b|c", delimiter="|")
    assert result == (["a", "b", "c"],)

    # Empty string
    result = node.execute("")
    assert result == ([""],)

    # Single value (no delimiters)
    result = node.execute("abc")
    assert result == (["abc"],)


def test_string_split_edge_cases():
    node = StringSplit()

    # Multiple consecutive delimiters
    result = node.execute("a,,b,,c")
    assert result == (["a", "", "b", "", "c"],)

    # Leading/trailing delimiters
    result = node.execute(",a,b,c,")
    assert result == (["", "a", "b", "c", ""],)

    # Multi-character delimiter
    result = node.execute("a<->b<->c", delimiter="<->")
    assert result == (["a", "b", "c"],)

    # Whitespace handling
    result = node.execute("  a  ,  b  ,  c  ")
    assert result == (["  a  ", "  b  ", "  c  "],)

    # Split on whitespace
    result = node.execute("a b c", delimiter=" ")
    assert result == (["a", "b", "c"],)


def test_string_split_special_chars():
    node = StringSplit()

    # Split on newline
    result = node.execute("a\nb\nc", delimiter="\n")
    assert result == (["a", "b", "c"],)

    # Split on tab
    result = node.execute("a\tb\tc", delimiter="\t")
    assert result == (["a", "b", "c"],)

    # Regex special characters as delimiters
    result = node.execute("a.b.c", delimiter=".")
    assert result == (["a", "b", "c"],)

    # Unicode delimiters
    result = node.execute("a→b→c", delimiter="→")
    assert result == (["a", "b", "c"],)


def test_string_split_metadata():
    node = StringSplit()

    # Check required class attributes
    assert hasattr(node, 'CATEGORY')
    assert hasattr(node, 'OUTPUT_IS_LIST')
    assert hasattr(node, 'RETURN_TYPES')
    assert hasattr(node, 'FUNCTION')

    # Verify OUTPUT_IS_LIST matches RETURN_TYPES
    assert len(node.OUTPUT_IS_LIST) == len(node.RETURN_TYPES)
    assert node.OUTPUT_IS_LIST == (True,)
    assert node.RETURN_TYPES == ("STRING",)

    # Check input types
    input_types = node.INPUT_TYPES()
    assert 'required' in input_types
    assert 'value' in input_types['required']
    assert 'delimiter' in input_types['required']
    assert input_types['required']['value'][0] == 'STRING'
    assert input_types['required']['delimiter'][0] == 'STRING'

    # Check delimiter default
    assert input_types['required']['delimiter'][1]['default'] == ','


def test_string_split_return_type():
    node = StringSplit()

    # Verify return structure
    result = node.execute("a,b,c")
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, str) for x in result[0])

import torch

from comfy_extras.nodes.nodes_strings import StringFormat


def test_string_format_basic():
    n = StringFormat()

    # Test basic string formatting
    result, = n.execute(format="Hello, {}!", value0="World")
    assert result == "Hello, World!"

    # Test multiple values
    result, = n.execute(format="{} plus {} equals {}", value0=2, value1=2, value2=4)
    assert result == "2 plus 2 equals 4"


def test_string_format_types():
    n = StringFormat()

    # Test with different types
    result, = n.execute(format="Float: {:.2f}, Int: {}, Bool: {}",
                        value0=3.14159, value1=42, value2=True)
    assert result == "Float: 3.14, Int: 42, Bool: True"

    # Test None values
    result, = n.execute(format="{}, {}, {}", value0=None, value1="test", value2=None)
    assert result == "None, test, None"


def test_string_format_tensors():
    n = StringFormat()

    # Test small tensor
    small_tensor = torch.tensor([1, 2, 3])
    result, = n.execute(format="Tensor: {}", value0=small_tensor)
    assert result == "Tensor: [1, 2, 3]"

    # Test large tensor
    large_tensor = torch.randn(100, 100)
    result, = n.execute(format="Large tensor: {}", value0=large_tensor)
    assert result == "Large tensor: <Tensor shape=100x100>"

    # Test mixed tensor sizes
    small_tensor = torch.tensor([1, 2])
    large_tensor = torch.randn(50, 50)
    result, = n.execute(format="{} and {}", value0=small_tensor, value1=large_tensor)
    assert "and <Tensor shape=50x50>" in result
    assert "[1, 2]" in result


def test_string_format_edge_cases():
    n = StringFormat()

    # Test with missing values
    result, = n.execute(format="{} {} {}", value0="a", value1="b")
    assert result.startswith("Format error: ")

    # Test with empty format string
    result, = n.execute(format="", value0="ignored")
    assert result == ""

    # Test with no placeholders
    result, = n.execute(format="Hello World", value0="ignored")
    assert result == "Hello World"

    # Test with named placeholders
    result, = n.execute(format="X: {value0}, Y: {value1}", value0=10, value1=20)
    assert result == "X: 10, Y: 20"

    # Test mixing None, tensors and regular values
    tensor = torch.tensor([1, 2, 3])
    result, = n.execute(format="{}, {}, {}", value0=None, value1=tensor, value2="test")
    assert result == "None, [1, 2, 3], test"


def test_string_format_tensor_edge_cases():
    n = StringFormat()

    # Test empty tensor
    empty_tensor = torch.tensor([])
    result, = n.execute(format="Empty tensor: {}", value0=empty_tensor)
    assert result == "Empty tensor: []"

    # Test scalar tensor
    scalar_tensor = torch.tensor(5)
    result, = n.execute(format="Scalar tensor: {}", value0=scalar_tensor)
    assert result == "Scalar tensor: 5"

    # Test multi-dimensional small tensor
    small_2d_tensor = torch.tensor([[1, 2], [3, 4]])
    result, = n.execute(format="2D tensor: {}", value0=small_2d_tensor)
    assert result == "2D tensor: [[1, 2], [3, 4]]"

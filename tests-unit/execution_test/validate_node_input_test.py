import pytest
from comfy_execution.validation import validate_node_input


def test_exact_match():
    """Test cases where types match exactly"""
    assert validate_node_input("STRING", "STRING")
    assert validate_node_input("STRING,INT", "STRING,INT")
    assert (
        validate_node_input("INT,STRING", "STRING,INT")
    )  # Order shouldn't matter


def test_strict_mode():
    """Test strict mode validation"""
    # Should pass - received type is subset of input type
    assert validate_node_input("STRING", "STRING,INT", strict=True)
    assert validate_node_input("INT", "STRING,INT", strict=True)
    assert validate_node_input("STRING,INT", "STRING,INT,BOOLEAN", strict=True)

    # Should fail - received type is not subset of input type
    assert not validate_node_input("STRING,INT", "STRING", strict=True)
    assert not validate_node_input("STRING,BOOLEAN", "STRING", strict=True)
    assert not validate_node_input("INT,BOOLEAN", "STRING,INT", strict=True)


def test_non_strict_mode():
    """Test non-strict mode validation (default behavior)"""
    # Should pass - types have overlap
    assert validate_node_input("STRING,BOOLEAN", "STRING,INT")
    assert validate_node_input("STRING,INT", "INT,BOOLEAN")
    assert validate_node_input("STRING", "STRING,INT")

    # Should fail - no overlap in types
    assert not validate_node_input("BOOLEAN", "STRING,INT")
    assert not validate_node_input("FLOAT", "STRING,INT")
    assert not validate_node_input("FLOAT,BOOLEAN", "STRING,INT")


def test_whitespace_handling():
    """Test that whitespace is handled correctly"""
    assert validate_node_input("STRING, INT", "STRING,INT")
    assert validate_node_input("STRING,INT", "STRING, INT")
    assert validate_node_input(" STRING , INT ", "STRING,INT")
    assert validate_node_input("STRING,INT", " STRING , INT ")


def test_empty_strings():
    """Test behavior with empty strings"""
    assert validate_node_input("", "")
    assert not validate_node_input("STRING", "")
    assert not validate_node_input("", "STRING")


def test_single_vs_multiple():
    """Test single type against multiple types"""
    assert validate_node_input("STRING", "STRING,INT,BOOLEAN")
    assert validate_node_input("STRING,INT,BOOLEAN", "STRING", strict=False)
    assert not validate_node_input("STRING,INT,BOOLEAN", "STRING", strict=True)


@pytest.mark.parametrize(
    "received,input_type,strict,expected",
    [
        ("STRING", "STRING", False, True),
        ("STRING,INT", "STRING,INT", False, True),
        ("STRING", "STRING,INT", True, True),
        ("STRING,INT", "STRING", True, False),
        ("BOOLEAN", "STRING,INT", False, False),
        ("STRING,BOOLEAN", "STRING,INT", False, True),
    ],
)
def test_parametrized_cases(received, input_type, strict, expected):
    """Parametrized test cases for various scenarios"""
    assert validate_node_input(received, input_type, strict) == expected


# https://github.com/FredBill1/comfyui-fb-utils/blob/main/core/types.py
class AnyType(str):
    """A special class that is always equal in not equal comparisons."""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


def test_wildcard():
    """Test behavior with wildcard"""
    assert validate_node_input("*", "STRING,INT")
    assert validate_node_input("STRING,INT", "*")
    assert validate_node_input("*", "*")

    # AnyType was previous wildcard hack used by many custom nodes.
    # AnyType should be treated as a wildcard.
    assert validate_node_input(AnyType(), "STRING,INT")
    assert validate_node_input("STRING,INT", AnyType())
    assert validate_node_input(AnyType(), AnyType())

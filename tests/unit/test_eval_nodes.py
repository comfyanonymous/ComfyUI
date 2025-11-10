import pytest
from unittest.mock import Mock, patch

from comfy.cli_args import default_configuration
from comfy.execution_context import context_configuration
from comfy_extras.nodes.nodes_eval import (
    eval_python,
    EvalPython_5_5,
    EvalPython_List_1,
    EvalPython_1_List,
    EvalPython_List_List,
)


@pytest.fixture
def eval_context():
    """Fixture that sets up execution context with eval enabled"""
    config = default_configuration()
    config.enable_eval = True
    with context_configuration(config):
        yield


def test_eval_python_basic_return(eval_context):
    """Test basic return statement with single value"""
    node = EvalPython_5_5()
    result = node.exec_py(pycode="return 42", value0=0, value1=1, value2=2, value3=3, value4=4)
    assert result == (42, None, None, None, None)


def test_eval_python_multiple_returns(eval_context):
    """Test return statement with tuple of values"""
    node = EvalPython_5_5()
    result = node.exec_py(
        pycode="return 1, 2, 3",
        value0=0, value1=0, value2=0, value3=0, value4=0
    )
    assert result == (1, 2, 3, None, None)


def test_eval_python_all_five_returns(eval_context):
    """Test return statement with all five values"""
    node = EvalPython_5_5()
    result = node.exec_py(
        pycode="return 'a', 'b', 'c', 'd', 'e'",
        value0=0, value1=0, value2=0, value3=0, value4=0
    )
    assert result == ('a', 'b', 'c', 'd', 'e')


def test_eval_python_excess_returns(eval_context):
    """Test that excess return values are truncated to 5"""
    node = EvalPython_5_5()
    result = node.exec_py(
        pycode="return 1, 2, 3, 4, 5, 6, 7",
        value0=0, value1=0, value2=0, value3=0, value4=0
    )
    assert result == (1, 2, 3, 4, 5)


def test_eval_python_use_value_args(eval_context):
    """Test that value arguments are accessible in pycode"""
    node = EvalPython_5_5()
    result = node.exec_py(
        pycode="return value0 + value1 + value2",
        value0=10, value1=20, value2=30, value3=0, value4=0
    )
    assert result == (60, None, None, None, None)


def test_eval_python_all_value_args(eval_context):
    """Test all value arguments are accessible"""
    node = EvalPython_5_5()
    result = node.exec_py(
        pycode="return value0, value1, value2, value3, value4",
        value0=1, value1=2, value2=3, value3=4, value4=5
    )
    assert result == (1, 2, 3, 4, 5)


def test_eval_python_computation(eval_context):
    """Test computation with value arguments"""
    node = EvalPython_5_5()
    code = """
x = value0 * 2
y = value1 * 3
z = x + y
return z
"""
    result = node.exec_py(
        pycode=code,
        value0=5, value1=10, value2=0, value3=0, value4=0
    )
    assert result == (40, None, None, None, None)


def test_eval_python_multiline(eval_context):
    """Test multiline code with conditionals"""
    node = EvalPython_5_5()
    code = """
if value0 > 10:
    result = "large"
else:
    result = "small"
return result, value0
"""
    result = node.exec_py(
        pycode=code,
        value0=15, value1=0, value2=0, value3=0, value4=0
    )
    assert result == ("large", 15, None, None, None)


def test_eval_python_list_comprehension(eval_context):
    """Test list comprehension and iteration"""
    node = EvalPython_5_5()
    code = """
numbers = [value0, value1, value2]
doubled = [x * 2 for x in numbers]
return sum(doubled)
"""
    result = node.exec_py(
        pycode=code,
        value0=1, value1=2, value2=3, value3=0, value4=0
    )
    assert result == (12, None, None, None, None)


def test_eval_python_string_operations(eval_context):
    """Test string operations"""
    node = EvalPython_5_5()
    code = """
s1 = str(value0)
s2 = str(value1)
return s1 + s2, len(s1 + s2)
"""
    result = node.exec_py(
        pycode=code,
        value0=123, value1=456, value2=0, value3=0, value4=0
    )
    assert result == ("123456", 6, None, None, None)


def test_eval_python_type_mixing(eval_context):
    """Test mixing different types"""
    node = EvalPython_5_5()
    code = """
return value0, str(value1), float(value2), bool(value3)
"""
    result = node.exec_py(
        pycode=code,
        value0=42, value1=100, value2=3, value3=1, value4=0
    )
    assert result == (42, "100", 3.0, True, None)


def test_eval_python_logger_available(eval_context):
    """Test that logger is available in eval context"""
    node = EvalPython_5_5()
    code = """
logger.info("test log")
return "success"
"""
    result = node.exec_py(
        pycode=code,
        value0=0, value1=0, value2=0, value3=0, value4=0
    )
    assert result == ("success", None, None, None, None)


def test_eval_python_print_available(eval_context):
    """Test that print function is available"""
    node = EvalPython_5_5()
    code = """
print("Hello World!")
return "printed"
"""
    result = node.exec_py(
        pycode=code,
        value0=0, value1=0, value2=0, value3=0, value4=0
    )
    assert result == ("printed", None, None, None, None)

def test_eval_python_print_is_called(eval_context):
    """Test that print function is called and receives correct arguments"""
    node = EvalPython_5_5()

    # Track print calls
    print_calls = []

    code = """
print("Hello", "World")
print("Line 2")
return "done"
"""

    # Mock exec to capture the globals dict and verify print is there
    original_exec = exec
    captured_globals = {}

    def mock_exec(code_str, globals_dict, *args, **kwargs):
        # Capture the globals dict
        captured_globals.update(globals_dict)

        # Wrap the print function to track calls
        original_print = globals_dict.get('print')
        if original_print:
            def tracked_print(*args):
                print_calls.append(args)
                return original_print(*args)
            globals_dict['print'] = tracked_print

        # Run the original exec
        return original_exec(code_str, globals_dict, *args, **kwargs)

    with patch('builtins.exec', side_effect=mock_exec):
        result = node.exec_py(
            pycode=code,
            value0=0, value1=0, value2=0, value3=0, value4=0
        )

    # Verify the result
    assert result == ("done", None, None, None, None)

    # Verify print was in the globals
    assert 'print' in captured_globals

    # Verify print was called twice with correct arguments
    assert len(print_calls) == 2
    assert print_calls[0] == ("Hello", "World")
    assert print_calls[1] == ("Line 2",)


def test_eval_python_print_sends_to_server(eval_context):
    """Test that print sends messages to PromptServer via context"""
    from comfy.execution_context import current_execution_context

    node = EvalPython_5_5()
    ctx = current_execution_context()

    # Mock the server's send_progress_text method
    original_send = ctx.server.send_progress_text if hasattr(ctx.server, 'send_progress_text') else None
    mock_send = Mock()
    ctx.server.send_progress_text = mock_send

    code = """
print("Hello", "World")
print("Value:", value0)
return "done"
"""

    try:
        result = node.exec_py(
            pycode=code,
            value0=42, value1=0, value2=0, value3=0, value4=0
        )

        # Verify the result
        assert result == ("done", None, None, None, None)

        # Verify print messages were sent to server
        assert mock_send.call_count == 2

        # Verify the messages sent
        calls = mock_send.call_args_list
        assert calls[0][0][0] == "Hello World"
        assert calls[0][0][1] == ctx.node_id
        assert calls[1][0][0] == "Value: 42"
        assert calls[1][0][1] == ctx.node_id
    finally:
        # Restore original
        if original_send:
            ctx.server.send_progress_text = original_send


def test_eval_python_config_disabled_raises():
    """Test that enable_eval=False raises an error"""
    node = EvalPython_5_5()
    config = default_configuration()
    config.enable_eval = False
    with context_configuration(config):
        with pytest.raises(ValueError, match="Python eval is disabled"):
            node.exec_py(
                pycode="return 42",
                value0=0, value1=0, value2=0, value3=0, value4=0
            )


def test_eval_python_config_enabled_works(eval_context):
    """Test that enable_eval=True allows execution"""
    node = EvalPython_5_5()
    result = node.exec_py(
        pycode="return 42",
        value0=0, value1=0, value2=0, value3=0, value4=0
    )
    assert result == (42, None, None, None, None)


def test_eval_python_default_code(eval_context):
    """Test the default code example works"""
    node = EvalPython_5_5()
    # Get the default code from INPUT_TYPES
    default_code = EvalPython_5_5.INPUT_TYPES()["required"]["pycode"][1]["default"]

    result = node.exec_py(
        pycode=default_code,
        value0=1, value1=2, value2=3, value3=4, value4=5
    )
    # Default code prints and returns the values
    assert result == (1, 2, 3, 4, 5)


def test_eval_python_function_definition(eval_context):
    """Test defining and using functions"""
    node = EvalPython_5_5()
    code = """
def multiply(a, b):
    return a * b

result = multiply(value0, value1)
return result
"""
    result = node.exec_py(
        pycode=code,
        value0=7, value1=8, value2=0, value3=0, value4=0
    )
    assert result == (56, None, None, None, None)


def test_eval_python_nested_functions(eval_context):
    """Test nested function definitions"""
    node = EvalPython_5_5()
    code = """
def outer(x):
    def inner(y):
        return y * 2
    return inner(x) + 10

result = outer(value0)
return result
"""
    result = node.exec_py(
        pycode=code,
        value0=5, value1=0, value2=0, value3=0, value4=0
    )
    assert result == (20, None, None, None, None)


def test_eval_python_dict_operations(eval_context):
    """Test dictionary creation and operations"""
    node = EvalPython_5_5()
    code = """
data = {
    'a': value0,
    'b': value1,
    'c': value2
}
return sum(data.values()), len(data)
"""
    result = node.exec_py(
        pycode=code,
        value0=10, value1=20, value2=30, value3=0, value4=0
    )
    assert result == (60, 3, None, None, None)


def test_eval_python_list_operations(eval_context):
    """Test list creation and operations"""
    node = EvalPython_5_5()
    code = """
items = [value0, value1, value2, value3, value4]
filtered = [x for x in items if x > 5]
return len(filtered), sum(filtered)
"""
    result = node.exec_py(
        pycode=code,
        value0=1, value1=10, value2=3, value3=15, value4=2
    )
    assert result == (2, 25, None, None, None)


def test_eval_python_early_return(eval_context):
    """Test early return in conditional"""
    node = EvalPython_5_5()
    code = """
if value0 > 100:
    return "large"
return "small"
"""
    result = node.exec_py(
        pycode=code,
        value0=150, value1=0, value2=0, value3=0, value4=0
    )
    assert result == ("large", None, None, None, None)


def test_eval_python_loop_with_return(eval_context):
    """Test loop with return statement"""
    node = EvalPython_5_5()
    code = """
total = 0
for i in range(value0):
    total += i
return total
"""
    result = node.exec_py(
        pycode=code,
        value0=10, value1=0, value2=0, value3=0, value4=0
    )
    assert result == (45, None, None, None, None)


def test_eval_python_exception_handling(eval_context):
    """Test try/except blocks"""
    node = EvalPython_5_5()
    code = """
try:
    result = value0 / value1
except ZeroDivisionError:
    result = float('inf')
return result
"""
    result = node.exec_py(
        pycode=code,
        value0=10, value1=0, value2=0, value3=0, value4=0
    )
    assert result == (float('inf'), None, None, None, None)


def test_eval_python_none_values(eval_context):
    """Test handling None values in inputs"""
    node = EvalPython_5_5()
    code = """
return value0, value1 is None, value2 is None
"""
    result = node.exec_py(
        pycode=code,
        value0=42, value1=None, value2=None, value3=0, value4=0
    )
    assert result == (42, True, True, None, None)


def test_eval_python_input_types():
    """Test that INPUT_TYPES returns correct structure"""
    input_types = EvalPython_5_5.INPUT_TYPES()
    assert "required" in input_types
    assert "optional" in input_types
    assert "pycode" in input_types["required"]
    assert input_types["required"]["pycode"][0] == "CODE_BLOCK_PYTHON"

    # Check optional inputs
    for i in range(5):
        assert f"value{i}" in input_types["optional"]


def test_eval_python_metadata():
    """Test node metadata"""
    assert EvalPython_5_5.FUNCTION == "exec_py"
    assert EvalPython_5_5.CATEGORY == "eval"
    assert len(EvalPython_5_5.RETURN_TYPES) == 5
    assert len(EvalPython_5_5.RETURN_NAMES) == 5
    assert all(name.startswith("item") for name in EvalPython_5_5.RETURN_NAMES)


def test_eval_python_factory_custom_inputs_outputs(eval_context):
    """Test creating nodes with custom input/output counts"""
    # Create a node with 3 inputs and 2 outputs
    CustomNode = eval_python(inputs=3, outputs=2)

    node = CustomNode()

    # Verify INPUT_TYPES has correct number of inputs
    input_types = CustomNode.INPUT_TYPES()
    assert len(input_types["optional"]) == 3
    assert "value0" in input_types["optional"]
    assert "value1" in input_types["optional"]
    assert "value2" in input_types["optional"]
    assert "value3" not in input_types["optional"]

    # Verify RETURN_TYPES has correct number of outputs
    assert len(CustomNode.RETURN_TYPES) == 2
    assert len(CustomNode.RETURN_NAMES) == 2

    # Test execution
    result = node.exec_py(
        pycode="return value0 + value1 + value2, value0 * 2",
        value0=1, value1=2, value2=3
    )
    assert result == (6, 2)


def test_eval_python_factory_custom_name(eval_context):
    """Test creating nodes with custom names"""
    CustomNode = eval_python(inputs=2, outputs=2, name="MyCustomEval")

    assert CustomNode.__name__ == "MyCustomEval"
    assert CustomNode.__qualname__ == "MyCustomEval"


def test_eval_python_factory_default_name(eval_context):
    """Test that default name follows pattern"""
    CustomNode = eval_python(inputs=3, outputs=4)

    assert CustomNode.__name__ == "EvalPython_3_4"
    assert CustomNode.__qualname__ == "EvalPython_3_4"


def test_eval_python_factory_single_output(eval_context):
    """Test node with single output"""
    SingleOutputNode = eval_python(inputs=2, outputs=1)

    node = SingleOutputNode()
    result = node.exec_py(
        pycode="return value0 + value1",
        value0=10, value1=20
    )
    assert result == (30,)


def test_eval_python_factory_many_outputs(eval_context):
    """Test node with many outputs"""
    ManyOutputNode = eval_python(inputs=1, outputs=10)

    node = ManyOutputNode()
    result = node.exec_py(
        pycode="return tuple(range(10))",
        value0=0
    )
    assert result == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def test_eval_python_factory_fewer_returns_than_outputs(eval_context):
    """Test that fewer returns are padded with None"""
    Node = eval_python(inputs=2, outputs=5)

    node = Node()
    result = node.exec_py(
        pycode="return value0, value1",
        value0=1, value1=2
    )
    assert result == (1, 2, None, None, None)


def test_eval_python_factory_more_returns_than_outputs(eval_context):
    """Test that excess returns are truncated"""
    Node = eval_python(inputs=2, outputs=3)

    node = Node()
    result = node.exec_py(
        pycode="return 1, 2, 3, 4, 5",
        value0=0, value1=0
    )
    assert result == (1, 2, 3)


def test_eval_python_list_1_input_is_list(eval_context):
    """Test EvalPython_List_1 with list input"""
    node = EvalPython_List_1()

    # Verify INPUT_IS_LIST is set
    assert EvalPython_List_1.INPUT_IS_LIST is True

    # Test that value0 receives a list
    result = node.exec_py(
        pycode="return sum(value0)",
        value0=[1, 2, 3, 4, 5]
    )
    assert result == (15,)


def test_eval_python_list_1_iterate_list(eval_context):
    """Test EvalPython_List_1 iterating over list input"""
    node = EvalPython_List_1()

    result = node.exec_py(
        pycode="return [x * 2 for x in value0]",
        value0=[1, 2, 3]
    )
    assert result == ([2, 4, 6],)


def test_eval_python_1_list_output_is_list(eval_context):
    """Test EvalPython_1_List with list output"""
    node = EvalPython_1_List()

    # Verify OUTPUT_IS_LIST is set
    assert EvalPython_1_List.OUTPUT_IS_LIST == (True,)

    # Test that returns a list
    result = node.exec_py(
        pycode="return list(range(value0))",
        value0=5
    )
    assert result == ([0, 1, 2, 3, 4],)


def test_eval_python_1_list_multiple_items(eval_context):
    """Test EvalPython_1_List returning multiple items in list"""
    node = EvalPython_1_List()

    result = node.exec_py(
        pycode="return ['a', 'b', 'c']",
        value0=0
    )
    assert result == (['a', 'b', 'c'],)


def test_eval_python_list_list_both(eval_context):
    """Test EvalPython_List_List with both list input and output"""
    node = EvalPython_List_List()

    # Verify both are set
    assert EvalPython_List_List.INPUT_IS_LIST is True
    assert EvalPython_List_List.OUTPUT_IS_LIST == (True,)

    # Test processing list input and returning list output
    result = node.exec_py(
        pycode="return [x ** 2 for x in value0]",
        value0=[1, 2, 3, 4]
    )
    assert result == ([1, 4, 9, 16],)


def test_eval_python_list_list_filter(eval_context):
    """Test EvalPython_List_List filtering a list"""
    node = EvalPython_List_List()

    result = node.exec_py(
        pycode="return [x for x in value0 if x > 5]",
        value0=[1, 3, 5, 7, 9, 11]
    )
    assert result == ([7, 9, 11],)


def test_eval_python_list_list_transform(eval_context):
    """Test EvalPython_List_List transforming list elements"""
    node = EvalPython_List_List()

    result = node.exec_py(
        pycode="return [str(x).upper() for x in value0]",
        value0=['hello', 'world', 'python']
    )
    assert result == (['HELLO', 'WORLD', 'PYTHON'],)


def test_eval_python_factory_with_list_flags(eval_context):
    """Test factory function with custom list flags"""
    # Create node with input as list but output scalar
    ListInputNode = eval_python(inputs=1, outputs=1, input_is_list=True, output_is_list=None)

    assert ListInputNode.INPUT_IS_LIST is True

    node = ListInputNode()
    result = node.exec_py(
        pycode="return len(value0)",
        value0=[1, 2, 3, 4, 5]
    )
    assert result == (5,)


def test_eval_python_factory_scalar_output_list(eval_context):
    """Test factory function with scalar input and list output"""
    ScalarToListNode = eval_python(inputs=1, outputs=1, input_is_list=None, output_is_list=(True,))

    assert ScalarToListNode.OUTPUT_IS_LIST == (True,)

    node = ScalarToListNode()
    result = node.exec_py(
        pycode="return [value0] * 3",
        value0='x'
    )
    assert result == (['x', 'x', 'x'],)


def test_eval_python_list_empty_list(eval_context):
    """Test list nodes with empty lists"""
    node = EvalPython_List_List()

    result = node.exec_py(
        pycode="return []",
        value0=[]
    )
    assert result == ([],)

import pytest
from comfy_extras.nodes_math import Add, Subtract, Multiply, Divide
from comfy_api.latest import io


class TestAdd:
    """Test suite for Add node."""

    def test_basic_addition_integers(self):
        """Test basic addition of two integers."""
        operands = {"operand0": 5, "operand1": 3}
        result = Add.execute(operands)
        assert result.value == 8
        assert isinstance(result, io.NodeOutput)

    def test_addition_multiple_integers(self):
        """Test addition with multiple integers."""
        operands = {"operand0": 5, "operand1": 3, "operand2": 2}
        result = Add.execute(operands)
        assert result.value == 10

    def test_addition_with_zero(self):
        """Test addition with zero."""
        operands = {"operand0": 10, "operand1": 0}
        result = Add.execute(operands)
        assert result.value == 10

    def test_addition_with_negative(self):
        """Test addition with negative numbers."""
        operands = {"operand0": 5, "operand1": -3}
        result = Add.execute(operands)
        assert result.value == 2

    def test_addition_floats(self):
        """Test addition with floats."""
        operands = {"operand0": 5.5, "operand1": 3.2}
        result = Add.execute(operands)
        assert abs(result.value - 8.7) < 1e-10

    def test_addition_strings(self):
        """Test addition with strings (concatenation)."""
        operands = {"operand0": "Hello", "operand1": " World"}
        result = Add.execute(operands)
        assert result.value == "Hello World"

    def test_addition_lists(self):
        """Test addition with lists (like conditionings)."""
        operands = {"operand0": [1, 2, 3], "operand1": [4, 5]}
        result = Add.execute(operands)
        assert result.value == [1, 2, 3, 4, 5]


class TestSubtract:
    """Test suite for Subtract node."""

    def test_basic_subtraction_integers(self):
        """Test basic subtraction of two integers."""
        operands = {"operand0": 10, "operand1": 3}
        result = Subtract.execute(operands)
        assert result.value == 7
        assert isinstance(result, io.NodeOutput)

    def test_subtraction_multiple_integers(self):
        """Test subtraction with multiple integers."""
        operands = {"operand0": 20, "operand1": 5, "operand2": 3}
        result = Subtract.execute(operands)
        assert result.value == 12

    def test_subtraction_with_zero(self):
        """Test subtraction with zero."""
        operands = {"operand0": 10, "operand1": 0}
        result = Subtract.execute(operands)
        assert result.value == 10

    def test_subtraction_resulting_negative(self):
        """Test subtraction that results in negative number."""
        operands = {"operand0": 3, "operand1": 10}
        result = Subtract.execute(operands)
        assert result.value == -7

    def test_subtraction_with_negative(self):
        """Test subtraction with negative numbers."""
        operands = {"operand0": 5, "operand1": -3}
        result = Subtract.execute(operands)
        assert result.value == 8

    def test_subtraction_floats(self):
        """Test subtraction with floats."""
        operands = {"operand0": 10.5, "operand1": 3.2}
        result = Subtract.execute(operands)
        assert abs(result.value - 7.3) < 1e-10


class TestMultiply:
    """Test suite for Multiply node."""

    def test_basic_multiplication_integers(self):
        """Test basic multiplication of two integers."""
        operands = {"operand0": 5, "operand1": 3}
        result = Multiply.execute(operands)
        assert result.value == 15
        assert isinstance(result, io.NodeOutput)

    def test_multiplication_multiple_integers(self):
        """Test multiplication with multiple integers."""
        operands = {"operand0": 2, "operand1": 3, "operand2": 4}
        result = Multiply.execute(operands)
        assert result.value == 24

    def test_multiplication_with_zero(self):
        """Test multiplication with zero."""
        operands = {"operand0": 10, "operand1": 0}
        result = Multiply.execute(operands)
        assert result.value == 0

    def test_multiplication_with_one(self):
        """Test multiplication with one."""
        operands = {"operand0": 10, "operand1": 1}
        result = Multiply.execute(operands)
        assert result.value == 10

    def test_multiplication_with_negative(self):
        """Test multiplication with negative numbers."""
        operands = {"operand0": 5, "operand1": -3}
        result = Multiply.execute(operands)
        assert result.value == -15

    def test_multiplication_floats(self):
        """Test multiplication with floats."""
        operands = {"operand0": 2.5, "operand1": 4.0}
        result = Multiply.execute(operands)
        assert abs(result.value - 10.0) < 1e-10

    def test_multiplication_string_repetition(self):
        """Test multiplication with string (repetition)."""
        operands = {"operand0": "Hello", "operand1": 3}
        result = Multiply.execute(operands)
        assert result.value == "HelloHelloHello"


class TestDivide:
    """Test suite for Divide node."""

    def test_basic_division_integers(self):
        """Test basic division of two integers."""
        operands = {"operand0": 10, "operand1": 2}
        result = Divide.execute(operands)
        assert result.value == 5.0  # Division returns float
        assert isinstance(result, io.NodeOutput)

    def test_division_multiple_integers(self):
        """Test division with multiple integers."""
        operands = {"operand0": 100, "operand1": 2, "operand2": 5}
        result = Divide.execute(operands)
        assert result.value == 10.0

    def test_division_with_remainder(self):
        """Test division that results in float."""
        operands = {"operand0": 10, "operand1": 3}
        result = Divide.execute(operands)
        assert abs(result.value - 3.3333333333333335) < 1e-10

    def test_division_with_negative(self):
        """Test division with negative numbers."""
        operands = {"operand0": 10, "operand1": -2}
        result = Divide.execute(operands)
        assert result.value == -5.0

    def test_division_by_one(self):
        """Test division by one."""
        operands = {"operand0": 10, "operand1": 1}
        result = Divide.execute(operands)
        assert result.value == 10.0

    def test_division_floats(self):
        """Test division with floats."""
        operands = {"operand0": 10.5, "operand1": 2.5}
        result = Divide.execute(operands)
        assert abs(result.value - 4.2) < 1e-10

    def test_validate_inputs_division_by_zero(self):
        """Test that division by zero is caught in validation."""
        operands = {"operand0": 10, "operand1": 0}
        validation_result = Divide.validate_inputs(operands)
        assert validation_result == "Division by zero is not allowed"

    def test_validate_inputs_division_by_zero_multiple(self):
        """Test that division by zero in later operands is caught."""
        operands = {"operand0": 10, "operand1": 2, "operand2": 0}
        validation_result = Divide.validate_inputs(operands)
        assert validation_result == "Division by zero is not allowed"

    def test_validate_inputs_normal_division(self):
        """Test that normal division passes validation."""
        operands = {"operand0": 10, "operand1": 2}
        validation_result = Divide.validate_inputs(operands)
        assert validation_result is True

    def test_execute_division_by_zero_raises_error(self):
        """Test that division by zero raises ValueError in execute."""
        operands = {"operand0": 10, "operand1": 0}
        with pytest.raises(ValueError, match="Division by zero is not allowed"):
            Divide.execute(operands)

    def test_execute_division_by_zero_multiple_raises_error(self):
        """Test that division by zero in later operands raises error."""
        operands = {"operand0": 100, "operand1": 2, "operand2": 0}
        with pytest.raises(ValueError, match="Division by zero is not allowed"):
            Divide.execute(operands)


class TestMathNodesSchema:
    """Test suite for node schemas."""

    def test_add_schema(self):
        """Test Add schema definition."""
        schema = Add.define_schema()
        assert schema.node_id == "Add"
        assert schema.display_name == "Add"
        assert schema.category == "math"

    def test_subtract_schema(self):
        """Test Subtract schema definition."""
        schema = Subtract.define_schema()
        assert schema.node_id == "Subtract"
        assert schema.display_name == "Subtract"
        assert schema.category == "math"

    def test_multiply_schema(self):
        """Test Multiply schema definition."""
        schema = Multiply.define_schema()
        assert schema.node_id == "Multiply"
        assert schema.display_name == "Multiply"
        assert schema.category == "math"

    def test_divide_schema(self):
        """Test Divide schema definition."""
        schema = Divide.define_schema()
        assert schema.node_id == "Divide"
        assert schema.display_name == "Divide"
        assert schema.category == "math"


class TestMathExtension:
    """Test suite for MathExtension."""

    @pytest.mark.asyncio
    async def test_extension_node_list(self):
        """Test that MathExtension returns all math nodes."""
        from comfy_extras.nodes_math import MathExtension
        
        extension = MathExtension()
        node_list = await extension.get_node_list()
        
        assert len(node_list) == 4
        assert Add in node_list
        assert Subtract in node_list
        assert Multiply in node_list
        assert Divide in node_list

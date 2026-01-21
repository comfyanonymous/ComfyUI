import pytest
from comfy_extras.nodes_math import IntAdd, IntSubtract, IntMultiply, IntDivide
from comfy_api.latest import io


class TestIntAdd:
    """Test suite for IntAdd node."""

    def test_basic_addition(self):
        """Test basic addition of two positive integers."""
        result = IntAdd.execute(5, 3)
        assert result.value == 8
        assert isinstance(result, io.NodeOutput)

    def test_addition_with_zero(self):
        """Test addition with zero."""
        result = IntAdd.execute(10, 0)
        assert result.value == 10

    def test_addition_with_negative(self):
        """Test addition with negative numbers."""
        result = IntAdd.execute(5, -3)
        assert result.value == 2

    def test_addition_both_negative(self):
        """Test addition of two negative numbers."""
        result = IntAdd.execute(-5, -3)
        assert result.value == -8

    def test_addition_large_numbers(self):
        """Test addition with large numbers."""
        result = IntAdd.execute(1000000, 2000000)
        assert result.value == 3000000


class TestIntSubtract:
    """Test suite for IntSubtract node."""

    def test_basic_subtraction(self):
        """Test basic subtraction of two positive integers."""
        result = IntSubtract.execute(10, 3)
        assert result.value == 7
        assert isinstance(result, io.NodeOutput)

    def test_subtraction_with_zero(self):
        """Test subtraction with zero."""
        result = IntSubtract.execute(10, 0)
        assert result.value == 10

    def test_subtraction_resulting_negative(self):
        """Test subtraction that results in negative number."""
        result = IntSubtract.execute(3, 10)
        assert result.value == -7

    def test_subtraction_with_negative(self):
        """Test subtraction with negative numbers."""
        result = IntSubtract.execute(5, -3)
        assert result.value == 8

    def test_subtraction_both_negative(self):
        """Test subtraction of two negative numbers."""
        result = IntSubtract.execute(-5, -3)
        assert result.value == -2


class TestIntMultiply:
    """Test suite for IntMultiply node."""

    def test_basic_multiplication(self):
        """Test basic multiplication of two positive integers."""
        result = IntMultiply.execute(5, 3)
        assert result.value == 15
        assert isinstance(result, io.NodeOutput)

    def test_multiplication_with_zero(self):
        """Test multiplication with zero."""
        result = IntMultiply.execute(10, 0)
        assert result.value == 0

    def test_multiplication_with_one(self):
        """Test multiplication with one."""
        result = IntMultiply.execute(10, 1)
        assert result.value == 10

    def test_multiplication_with_negative(self):
        """Test multiplication with negative numbers."""
        result = IntMultiply.execute(5, -3)
        assert result.value == -15

    def test_multiplication_both_negative(self):
        """Test multiplication of two negative numbers."""
        result = IntMultiply.execute(-5, -3)
        assert result.value == 15

    def test_multiplication_large_numbers(self):
        """Test multiplication with large numbers."""
        result = IntMultiply.execute(1000, 2000)
        assert result.value == 2000000


class TestIntDivide:
    """Test suite for IntDivide node."""

    def test_basic_division(self):
        """Test basic division of two positive integers."""
        result = IntDivide.execute(10, 2)
        assert result.value == 5
        assert isinstance(result, io.NodeOutput)

    def test_division_with_remainder(self):
        """Test division that results in floor division."""
        result = IntDivide.execute(10, 3)
        assert result.value == 3  # Floor division: 10 // 3 = 3

    def test_division_with_negative(self):
        """Test division with negative numbers."""
        result = IntDivide.execute(10, -3)
        assert result.value == -4  # Floor division: 10 // -3 = -4

    def test_division_both_negative(self):
        """Test division of two negative numbers."""
        result = IntDivide.execute(-10, -3)
        assert result.value == 3  # Floor division: -10 // -3 = 3

    def test_division_by_one(self):
        """Test division by one."""
        result = IntDivide.execute(10, 1)
        assert result.value == 10

    def test_validate_inputs_division_by_zero(self):
        """Test that division by zero is caught in validation."""
        validation_result = IntDivide.validate_inputs(10, 0)
        assert validation_result == "Division by zero is not allowed"

    def test_validate_inputs_normal_division(self):
        """Test that normal division passes validation."""
        validation_result = IntDivide.validate_inputs(10, 2)
        assert validation_result is True

    def test_execute_division_by_zero_raises_error(self):
        """Test that division by zero raises ValueError in execute."""
        with pytest.raises(ValueError, match="Division by zero is not allowed"):
            IntDivide.execute(10, 0)

    def test_division_large_numbers(self):
        """Test division with large numbers."""
        result = IntDivide.execute(2000000, 1000)
        assert result.value == 2000


class TestMathNodesSchema:
    """Test suite for node schemas."""

    def test_int_add_schema(self):
        """Test IntAdd schema definition."""
        schema = IntAdd.define_schema()
        assert schema.node_id == "IntAdd"
        assert schema.display_name == "Int Add"
        assert schema.category == "math"
        assert len(schema.inputs) == 2
        assert len(schema.outputs) == 1

    def test_int_subtract_schema(self):
        """Test IntSubtract schema definition."""
        schema = IntSubtract.define_schema()
        assert schema.node_id == "IntSubtract"
        assert schema.display_name == "Int Subtract"
        assert schema.category == "math"

    def test_int_multiply_schema(self):
        """Test IntMultiply schema definition."""
        schema = IntMultiply.define_schema()
        assert schema.node_id == "IntMultiply"
        assert schema.display_name == "Int Multiply"
        assert schema.category == "math"

    def test_int_divide_schema(self):
        """Test IntDivide schema definition."""
        schema = IntDivide.define_schema()
        assert schema.node_id == "IntDivide"
        assert schema.display_name == "Int Divide"
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
        assert IntAdd in node_list
        assert IntSubtract in node_list
        assert IntMultiply in node_list
        assert IntDivide in node_list

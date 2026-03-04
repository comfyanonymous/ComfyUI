import pytest
from collections import OrderedDict
from unittest.mock import patch, MagicMock

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras.nodes_math import _positional_alias, MathExpressionNode


class TestPositionalAlias:
    def test_first_letter(self):
        assert _positional_alias(0) == "a"

    def test_last_letter(self):
        assert _positional_alias(25) == "z"

    def test_all_letters(self):
        expected = list("abcdefghijklmnopqrstuvwxyz")
        result = [_positional_alias(i) for i in range(26)]
        assert result == expected


class TestMathExpressionExecute:
    @staticmethod
    def _exec(expression: str, **kwargs) -> object:
        values = OrderedDict(kwargs)
        return MathExpressionNode.execute(expression, values)

    def test_addition(self):
        result = self._exec("a + b", a=3, b=4)
        assert result[0] == 7.0
        assert result[1] == 7

    def test_subtraction(self):
        result = self._exec("a - b", a=10, b=3)
        assert result[0] == 7.0
        assert result[1] == 7

    def test_multiplication(self):
        result = self._exec("a * b", a=3, b=5)
        assert result[0] == 15.0
        assert result[1] == 15

    def test_division(self):
        result = self._exec("a / b", a=10, b=4)
        assert result[0] == 2.5
        assert result[1] == 2

    def test_single_input(self):
        result = self._exec("a * 2", a=5)
        assert result[0] == 10.0
        assert result[1] == 10

    def test_three_inputs(self):
        result = self._exec("a + b + c", a=1, b=2, c=3)
        assert result[0] == 6.0
        assert result[1] == 6

    def test_float_inputs(self):
        result = self._exec("a + b", a=1.5, b=2.5)
        assert result[0] == 4.0
        assert result[1] == 4

    def test_mixed_int_float_inputs(self):
        result = self._exec("a * b", a=1024, b=1.5)
        assert result[0] == 1536.0
        assert result[1] == 1536

    def test_mixed_resolution_scale(self):
        result = self._exec("a * b", a=512, b=0.75)
        assert result[0] == 384.0
        assert result[1] == 384

    def test_sum_values_array(self):
        result = self._exec("sum(values)", a=1, b=2, c=3)
        assert result[0] == 6.0

    def test_sum_variadic(self):
        result = self._exec("sum(a, b, c)", a=1, b=2, c=3)
        assert result[0] == 6.0

    def test_min_values(self):
        result = self._exec("min(values)", a=5, b=2, c=8)
        assert result[0] == 2.0

    def test_max_values(self):
        result = self._exec("max(values)", a=5, b=2, c=8)
        assert result[0] == 8.0

    def test_abs_function(self):
        result = self._exec("abs(a)", a=-7)
        assert result[0] == 7.0
        assert result[1] == 7

    def test_sqrt(self):
        result = self._exec("sqrt(a)", a=16)
        assert result[0] == 4.0
        assert result[1] == 4

    def test_ceil(self):
        result = self._exec("ceil(a)", a=2.3)
        assert result[0] == 3.0
        assert result[1] == 3

    def test_floor(self):
        result = self._exec("floor(a)", a=2.7)
        assert result[0] == 2.0
        assert result[1] == 2

    def test_sin(self):
        result = self._exec("sin(a)", a=0)
        assert result[0] == 0.0

    def test_log10(self):
        result = self._exec("log10(a)", a=100)
        assert result[0] == 2.0
        assert result[1] == 2

    def test_float_output_type(self):
        result = self._exec("a + b", a=1, b=2)
        assert isinstance(result[0], float)

    def test_int_output_type(self):
        result = self._exec("a + b", a=1, b=2)
        assert isinstance(result[1], int)

    def test_non_numeric_result_raises(self):
        with pytest.raises((ValueError, Exception)):
            self._exec("'hello'", a=42)

    def test_undefined_function_raises(self):
        with pytest.raises(Exception, match="not defined"):
            self._exec("str(a)", a=42)

    def test_boolean_result_raises(self):
        with pytest.raises(ValueError, match="got bool"):
            self._exec("a > b", a=5, b=3)

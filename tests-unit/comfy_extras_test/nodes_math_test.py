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
        assert result[0] == 7

    def test_subtraction(self):
        result = self._exec("a - b", a=10, b=3)
        assert result[0] == 7

    def test_multiplication(self):
        result = self._exec("a * b", a=3, b=5)
        assert result[0] == 15

    def test_division(self):
        result = self._exec("a / b", a=10, b=4)
        assert result[0] == 2.5

    def test_single_input(self):
        result = self._exec("a * 2", a=5)
        assert result[0] == 10

    def test_three_inputs(self):
        result = self._exec("a + b + c", a=1, b=2, c=3)
        assert result[0] == 6

    def test_float_inputs(self):
        result = self._exec("a + b", a=1.5, b=2.5)
        assert result[0] == 4.0

    def test_sum_values_array(self):
        result = self._exec("$sum(values)", a=1, b=2, c=3)
        assert result[0] == 6

    def test_non_numeric_result_raises(self):
        with pytest.raises(ValueError, match="must evaluate to a numeric result"):
            self._exec("$string(a)", a=42)

    def test_error_message_includes_expression(self):
        with pytest.raises(ValueError, match="'\\$string\\(a\\)'"):
            self._exec("$string(a)", a=42)

    def test_boolean_result_raises(self):
        with pytest.raises(ValueError, match="got bool"):
            self._exec("a > b", a=5, b=3)

import pytest
from comfy_extras.nodes.nodes_logic import LazySwitch, UnaryOperation, BooleanUnaryOperation, BinaryOperation, \
    BooleanBinaryOperation


def test_lazy_switch():
    n = LazySwitch()

    # Test True branch
    res_true_check = n.check_lazy_status(switch=True)
    assert res_true_check == ["on_true"]
    res_true, = n.execute(switch=True, on_false="val_false", on_true="val_true")
    assert res_true == "val_true"

    # Test False branch
    res_false_check = n.check_lazy_status(switch=False)
    assert res_false_check == ["on_false"]
    res_false, = n.execute(switch=False, on_false="val_false", on_true="val_true")
    assert res_false == "val_false"

    # Test with None
    res_none, = n.execute(switch=False, on_false=None, on_true="val_true")
    assert res_none is None


def test_unary_operation():
    n = UnaryOperation()

    # Test 'not'
    res_not_true, = n.execute(value=True, op="not")
    assert res_not_true is False
    res_not_false, = n.execute(value=False, op="not")
    assert res_not_false is True
    res_not_str, = n.execute(value="hello", op="not")
    assert res_not_str is False
    res_not_empty_str, = n.execute(value="", op="not")
    assert res_not_empty_str is True

    # Test 'inv' (invert, ~)
    res_inv, = n.execute(value=5, op="inv")
    assert res_inv == ~5  # -6

    # Test 'neg' (-)
    res_neg, = n.execute(value=10, op="neg")
    assert res_neg == -10
    res_neg_neg, = n.execute(value=-10, op="neg")
    assert res_neg_neg == 10

    # Test 'pos' (+)
    res_pos, = n.execute(value=-5, op="pos")
    assert res_pos == -5
    res_pos_pos, = n.execute(value=5, op="pos")
    assert res_pos_pos == 5


def test_boolean_unary_operation():
    n = BooleanUnaryOperation()

    # Test 'not'
    res_not_true, = n.execute(value=True, op="not")
    assert res_not_true is False
    res_not_false, = n.execute(value=False, op="not")
    assert res_not_false is True

    # Test truthiness
    res_not_int_1, = n.execute(value=1, op="not")
    assert res_not_int_1 is False
    res_not_int_0, = n.execute(value=0, op="not")
    assert res_not_int_0 is True
    res_not_str, = n.execute(value="hello", op="not")
    assert res_not_str is False
    res_not_empty_str, = n.execute(value="", op="not")
    assert res_not_empty_str is True


def test_binary_operation():
    n = BinaryOperation()

    # Test ops
    res_eq, = n.execute(lhs=5, op="eq", rhs=5)
    assert res_eq is True
    res_ne, = n.execute(lhs=5, op="ne", rhs=6)
    assert res_ne is True
    res_lt, = n.execute(lhs=5, op="lt", rhs=6)
    assert res_lt is True
    res_gt, = n.execute(lhs=6, op="gt", rhs=5)
    assert res_gt is True
    res_le, = n.execute(lhs=5, op="le", rhs=5)
    assert res_le is True
    res_ge, = n.execute(lhs=5, op="ge", rhs=5)
    assert res_ge is True
    res_add, = n.execute(lhs=5, op="add", rhs=3)
    assert res_add == 8
    res_sub, = n.execute(lhs=5, op="sub", rhs=3)
    assert res_sub == 2
    res_mul, = n.execute(lhs=5, op="mul", rhs=3)
    assert res_mul == 15
    res_div, = n.execute(lhs=10, op="truediv", rhs=4)
    assert res_div == 2.5
    res_floor_div, = n.execute(lhs=10, op="floordiv", rhs=3)
    assert res_floor_div == 3

    # Test logical 'and'
    res_and_tt, = n.execute(lhs=True, op="and", rhs=True)
    assert res_and_tt is True
    res_and_tf, = n.execute(lhs=True, op="and", rhs=False)
    assert res_and_tf is False
    res_and_ff, = n.execute(lhs=False, op="and", rhs=False)
    assert res_and_ff is False

    # Test logical 'or'
    res_or_tf, = n.execute(lhs=True, op="or", rhs=False)
    assert res_or_tf is True
    res_or_ft, = n.execute(lhs=False, op="or", rhs=True)
    assert res_or_ft is True
    res_or_ff, = n.execute(lhs=False, op="or", rhs=False)
    assert res_or_ff is False


def test_binary_operation_lazy_check():
    n = BinaryOperation()

    # Test standard ops
    assert n.check_lazy_status(op="eq") == ["lhs", "rhs"]
    assert n.check_lazy_status(op="add", lhs=1, rhs=2) == []
    assert n.check_lazy_status(op="add", lhs=None, rhs=None) == ["lhs", "rhs"]

    # Test 'and'
    assert n.check_lazy_status(op="and", lhs=None) == ["lhs"]
    assert n.check_lazy_status(op="and", lhs=True, rhs=None) == ["rhs"]
    assert n.check_lazy_status(op="and", lhs=False, rhs=None) == []
    assert n.check_lazy_status(op="and", lhs=False, rhs=True) == []

    # Test 'or'
    assert n.check_lazy_status(op="or", lhs=None) == ["lhs"]
    assert n.check_lazy_status(op="or", lhs=True, rhs=None) == []
    assert n.check_lazy_status(op="or", lhs=False, rhs=None) == ["rhs"]
    assert n.check_lazy_status(op="or", lhs=True, rhs=False) == []


def test_boolean_binary_operation():
    n = BooleanBinaryOperation()

    # Test 'eq'
    res_eq, = n.execute(lhs=5, op="eq", rhs=5)
    assert res_eq is True
    res_ne, = n.execute(lhs=5, op="eq", rhs=6)
    assert res_ne is False

    # Test truthiness
    res_and_truthy, = n.execute(lhs="hello", op="and", rhs=1)
    assert res_and_truthy is True
    res_and_falsy, = n.execute(lhs="hello", op="and", rhs=0)
    assert res_and_falsy is False
    res_or_falsy, = n.execute(lhs="", op="or", rhs=0)
    assert res_or_falsy is False
    res_or_truthy, = n.execute(lhs="", op="or", rhs="test")
    assert res_or_truthy is True


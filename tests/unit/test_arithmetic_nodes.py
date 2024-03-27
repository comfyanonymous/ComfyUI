import pytest

from comfy_extras.nodes.nodes_arithmetic import IntAdd, IntSubtract, IntMultiply, IntDivide, IntMod, IntPower, FloatAdd, FloatSubtract, FloatMultiply, FloatDivide, FloatPower, FloatMin, FloatMax, FloatAbs, FloatAverage, IntMin, IntMax, IntAbs, IntAverage, FloatLerp, IntLerp, IntClamp, IntInverseLerp, FloatClamp, FloatInverseLerp


def test_int_add():
    n = IntAdd()
    res, = n.execute(value0=1, value1=2, value2=3)
    assert res == 6


def test_int_subtract():
    n = IntSubtract()
    res, = n.execute(value0=10, value1=3)
    assert res == 7


def test_int_multiply():
    n = IntMultiply()
    res, = n.execute(value0=2, value1=3, value2=4)
    assert res == 24


def test_int_divide():
    n = IntDivide()
    res, = n.execute(value0=10, value1=3)
    assert res == 3

    res, = n.execute(value0=10, value1=0)
    assert res == 0


def test_int_mod():
    n = IntMod()
    res, = n.execute(value0=10, value1=3)
    assert res == 1

    res, = n.execute(value0=10, value1=0)
    assert res == 0


def test_int_power():
    n = IntPower()
    res, = n.execute(base=2, exponent=3)
    assert res == 8


def test_float_add():
    n = FloatAdd()
    res, = n.execute(value0=1.5, value1=2.3, value2=3.7)
    assert pytest.approx(res) == 7.5


def test_float_subtract():
    n = FloatSubtract()
    res, = n.execute(value0=10.5, value1=3.2)
    assert pytest.approx(res) == 7.3


def test_float_multiply():
    n = FloatMultiply()
    res, = n.execute(value0=2.5, value1=3.0, value2=4.0)
    assert pytest.approx(res) == 30.0


def test_float_divide():
    n = FloatDivide()
    res, = n.execute(value0=10.0, value1=4.0)
    assert pytest.approx(res) == 2.5

    res, = n.execute(value0=10.0, value1=0.0)
    assert res == float("inf")


def test_float_power():
    n = FloatPower()
    res, = n.execute(base=2.5, exponent=3.0)
    assert pytest.approx(res) == 15.625


def test_float_min():
    n = FloatMin()
    res, = n.execute(value0=1.5, value1=2.3, value2=0.7)
    assert res == 0.7


def test_float_max():
    n = FloatMax()
    res, = n.execute(value0=1.5, value1=2.3, value2=0.7)
    assert res == 2.3


def test_float_abs():
    n = FloatAbs()
    res, = n.execute(value=-3.14)
    assert res == 3.14


def test_float_average():
    n = FloatAverage()
    res, = n.execute(value0=1.5, value1=2.5, value2=3.5)
    assert res == 2.5


def test_int_min():
    n = IntMin()
    res, = n.execute(value0=5, value1=2, value2=7)
    assert res == 2


def test_int_max():
    n = IntMax()
    res, = n.execute(value0=5, value1=2, value2=7)
    assert res == 7


def test_int_abs():
    n = IntAbs()
    res, = n.execute(value=-10)
    assert res == 10


def test_int_average():
    n = IntAverage()
    res, = n.execute(value0=2, value1=4, value2=6)
    assert res == 4


def test_float_lerp():
    n = FloatLerp()
    res, = n.execute(a=0.0, b=1.0, t=0.5, clamped=True)
    assert res == 0.5

    res, = n.execute(a=0.0, b=1.0, t=1.5, clamped=True)
    assert res == 1.0

    res, = n.execute(a=0.0, b=1.0, t=1.5, clamped=False)
    assert res == 1.5


def test_int_lerp():
    n = IntLerp()
    res, = n.execute(a=0, b=10, t=0.5, clamped=True)
    assert res == 5

    res, = n.execute(a=0, b=10, t=1.5, clamped=True)
    assert res == 10

    res, = n.execute(a=0, b=10, t=1.5, clamped=False)
    assert res == 15


def test_float_inverse_lerp():
    n = FloatInverseLerp()
    res, = n.execute(a=0.0, b=1.0, value=0.5, clamped=True)
    assert res == 0.5

    res, = n.execute(a=0.0, b=1.0, value=1.5, clamped=True)
    assert res == 1.0

    res, = n.execute(a=0.0, b=1.0, value=1.5, clamped=False)
    assert res == 1.5


def test_float_clamp():
    n = FloatClamp()
    res, = n.execute(value=0.5, min=0.0, max=1.0)
    assert res == 0.5

    res, = n.execute(value=1.5, min=0.0, max=1.0)
    assert res == 1.0

    res, = n.execute(value=-0.5, min=0.0, max=1.0)
    assert res == 0.0


def test_int_inverse_lerp():
    n = IntInverseLerp()
    res, = n.execute(a=0, b=10, value=5, clamped=True)
    assert res == 0.5

    res, = n.execute(a=0, b=10, value=15, clamped=True)
    assert res == 1.0

    res, = n.execute(a=0, b=10, value=15, clamped=False)
    assert res == 1.5


def test_int_clamp():
    n = IntClamp()
    res, = n.execute(value=5, min=0, max=10)
    assert res == 5

    res, = n.execute(value=15, min=0, max=10)
    assert res == 10

    res, = n.execute(value=-5, min=0, max=10)
    assert res == 0

import torch
import math

def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )

    mantissa_scaled += torch.rand(mantissa_scaled.size(), dtype=mantissa_scaled.dtype, layout=mantissa_scaled.layout, device=mantissa_scaled.device, generator=generator)
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)

#Not 100% sure about this
def manual_stochastic_round_to_float8(x, dtype, generator=None):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    # Combine exponent calculation and clamping
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    # Combine mantissa calculation and rounding
    normal_mask = ~(exponent == 0)

    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )
    del abs_x

    return sign.to(dtype=dtype)



def stochastic_rounding(value, dtype, seed=0):
    if dtype == torch.float32:
        return value.to(dtype=torch.float32)
    if dtype == torch.float16:
        return value.to(dtype=torch.float16)
    if dtype == torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        return manual_stochastic_round_to_float8(value, dtype, generator=generator)

    return value.to(dtype=dtype)

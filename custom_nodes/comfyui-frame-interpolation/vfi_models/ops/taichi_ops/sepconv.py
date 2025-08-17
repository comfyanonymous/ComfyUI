import taichi as ti
import taichi.math as tm
from functools import reduce

@ti.kernel
def sepconv_out(tenIn: ti.types.ndarray(), tenVer: ti.types.ndarray(), tenHor: ti.types.ndarray(), tenOut: ti.types.ndarray()):
    N, C, H, W = tenIn.shape
    intIndex = 0
    for i, ch, y, x in ti.ndrange(N, C, H, W):
        fltOut, fltKahanc, fltKahany, fltKahant = 0.0, 0.0, 0.0, 0.0
        for intFy, intFx in ti.ndrange(tenVer.shape[1], tenHor.shape[1]):
            fltKahany = tenIn[i, ch, y + intFy, x + intFx] * tenVer[i, intFy, y, x] * tenHor[i, intFx, y, x]
            fltKahany = fltKahany - fltKahanc
            fltKahant = fltOut + fltKahany
            fltKahanc = (fltKahant - fltOut) - fltKahany
            fltOut = fltKahant
        tenOut[intIndex] = fltOut
        intIndex += 1


def worker_interface(op_name, tensors):
    if op_name == "sepconv_out":
        tenIn, tenVer, tenHor = tensors
        real_tenOut_shape = [
            tenIn.shape[0],
            tenIn.shape[1],
            tenVer.shape[2] and tenHor.shape[2],
            tenVer.shape[3] and tenHor.shape[3],
        ]
        tenOut = tenIn.new_zeros([
            int(reduce(lambda a, b: a * b, real_tenOut_shape))
        ])
        sepconv_out(tenIn, tenVer, tenHor, tenOut)
        tenOut = tenOut.view(*real_tenOut_shape)
        return (tenOut, )
    
    raise NotImplementedError(op_name)

__all__ = ["worker_interface"]
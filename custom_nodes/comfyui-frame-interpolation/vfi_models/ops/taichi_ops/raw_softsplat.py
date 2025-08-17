#Seperate taichi kernels to another file so that comfy.model_management won't be called in the new process

import taichi as ti
import taichi.math as tm

@ti.func
def put_to_tenOut(tenOut: ti.types.ndarray(), fltIn: ti.i32, flt: ti.i32, pos:tm.uvec2, i:ti.i32, ch:ti.i32):
    N, C, H, W = tenOut.shape
    if (pos.x >= 0) and (pos.x < W) and (pos.y >= 0) and (pos.y < H):
        tenOut[i, ch, pos.y, pos.x] += fltIn * flt
@ti.kernel
def softsplat_out(tenIn: ti.types.ndarray(), tenFlow: ti.types.ndarray(), tenOut: ti.types.ndarray()):
    N, C, H, W = tenIn.shape
    for i, ch, y, x in ti.ndrange(N, C, H, W):
        fltX = x + tenFlow[i, 0, y, x]
        fltY = y + tenFlow[i, 1, y, x]
        fltIn = tenIn[i, ch, y, x]

        northWest = tm.ivec2(ti.floor(fltX), ti.floor(fltY))
        northEast = northWest + [1, 0]
        southWest = northWest + [0, 1]
        southEast = northWest + [1, 1]
   
        fltNorthwest = (southEast.x - fltX) * (southEast.y - fltY)
        fltNortheast = (fltX - southWest.x) * (southWest.y - fltY)
        fltSouthwest = (northEast.x - fltX) * (fltY - northEast.y)
        fltSoutheast = (fltX - northWest.x) * (fltY - northWest.y)

        put_to_tenOut(tenOut, fltIn, fltNorthwest, northWest, i, ch)
        put_to_tenOut(tenOut, fltIn, fltNortheast, northEast, i, ch)
        put_to_tenOut(tenOut, fltIn, fltSouthwest, southWest, i, ch)
        put_to_tenOut(tenOut, fltIn, fltSoutheast, southEast, i, ch)

@ti.func
def add_to_fltFlowgrad(fltFlowgrad, tenOutgrad, fltIn, flt, pos, i, ch):
    N, C, H, W = tenOutgrad.shape
    if (pos.x >= 0) and (pos.x < W) and (pos.y >= 0) and (pos.y < H):
        fltFlowgrad += tenOutgrad[i, ch, pos.y, pos.x] * fltIn * flt

@ti.kernel
def softsplat_flowgrad(
    tenIn: ti.types.ndarray(), 
    tenFlow: ti.types.ndarray(), 
    tenOutgrad: ti.types.ndarray(), 
    tenIngrad: ti.types.ndarray(), 
    tenFlowgrad: ti.types.ndarray()
):
    N, C, H, W = tenFlowgrad.shape
    for i, ch, y, x in ti.ndrange(N, C, H, W):
        fltFlowgrad = 0.0
        fltX = x + tenFlow[i, 0, y, x]
        fltY = y + tenFlow[i, 1, y, x]

        northWest = tm.vec2(ti.floor(fltX, dtype=ti.i32), ti.floor(fltY, dtype=ti.i32))
        northEast = tm.vec2(northWest.x + 1, northWest.y)
        southWest = tm.vec2(northWest.x, northWest.y + 1)
        southEast = tm.vec2(northWest.x + 1, northWest.y + 1)

        if ch == 0:
            fltNorthwest = -1.0 * (southEast.y - fltY)
            fltNortheast = +1.0 * (southWest.y - fltY)
            fltSouthwest = -1.0 * (fltY - northEast.y)
            fltSoutheast = +1.0 * (fltY - northWest.y)
        
        elif ch == 1:
            fltNorthwest = -1.0 * (southEast.x - fltX)
            fltNortheast = -1.0 * (fltX - southWest.x)
            fltSouthwest = +1.0 * (northEast.x - fltX)
            fltSoutheast = +1.0 * (fltX - northWest.x)

        for outgrad_ch in ti.ndrange(tenOutgrad.shape[1]):
            fltIn = tenIn[i, outgrad_ch, y, x]
            add_to_fltFlowgrad(fltFlowgrad, tenOutgrad, fltIn, fltNorthwest, northWest, i, outgrad_ch)
            add_to_fltFlowgrad(fltFlowgrad, tenOutgrad, fltIn, fltNortheast, northEast, i, outgrad_ch)
            add_to_fltFlowgrad(fltFlowgrad, tenOutgrad, fltIn, fltSouthwest, southWest, i, outgrad_ch)
            add_to_fltFlowgrad(fltFlowgrad, tenOutgrad, fltIn, fltSoutheast, southEast, i, outgrad_ch)

        tenFlowgrad[i] = fltFlowgrad #Is 'i' the same as intIndex?

@ti.func
def add_to_fltIngrad(fltIngrad, tenOutgrad, flt, pos, i, ch):
    N, C, H, W = tenOutgrad.shape
    if (pos.x >= 0) and (pos.x < W) and (pos.y >= 0) and (pos.y < H):
        fltIngrad += tenOutgrad[i, ch, pos.y, pos.x] * flt
@ti.kernel
def softsplat_ingrad(
    tenIn: ti.types.ndarray(), 
    tenFlow: ti.types.ndarray(), 
    tenOutgrad: ti.types.ndarray(), 
    tenIngrad: ti.types.ndarray(), 
    tenFlowgrad: ti.types.ndarray()
):
    N, C, H, W = tenIngrad.shape
    for i, ch, y, x in ti.ndrange(N, C, H, W):
        fltIngrad = 0.0
        fltX = x + tenFlow[i, 0, y, x]
        fltY = y + tenFlow[i, 1, y, x]

        northWest = tm.vec2(ti.floor(fltX, dtype=ti.i32), ti.floor(fltY, dtype=ti.i32))
        northEast = tm.vec2(northWest.x + 1, northWest.y)
        southWest = tm.vec2(northWest.x, northWest.y + 1)
        southEast = tm.vec2(northWest.x + 1, northWest.y + 1)

        fltNorthwest = (southEast.x - fltX) * (southEast.y - fltY)
        fltNortheast = (fltX - southWest.x) * (southWest.y - fltY)
        fltSouthwest = (northEast.x - fltX) * (fltY - northEast.y)
        fltSoutheast = (fltX - northWest.x) * (fltY - northWest.y)
        
        add_to_fltIngrad(fltIngrad, tenOutgrad, fltNorthwest, northWest, i, ch)
        add_to_fltIngrad(fltIngrad, tenOutgrad, fltNortheast, northEast, i, ch)
        add_to_fltIngrad(fltIngrad, tenOutgrad, fltSouthwest, southWest, i, ch)
        add_to_fltIngrad(fltIngrad, tenOutgrad, fltSoutheast, southEast, i, ch)
        tenIngrad[i] = fltIngrad

# end

def worker_interface(op_name, tensors):
    if op_name == "softsplat_out":
        tenIn, tenFlow = tensors
        tenOut = tenIn.new_zeros(tenIn.shape)
        softsplat_out(tenIn, tenFlow, tenOut)
        return (tenOut, )
    
    raise NotImplementedError(op_name)

__all__ = ["worker_interface"]
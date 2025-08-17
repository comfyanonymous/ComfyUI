import torch.multiprocessing as mp
import torch
from .raw_softsplat import worker_interface as raw_softsplat
from .costvol import worker_interface as costvol
from .sepconv import worker_interface as sepconv
from .utils import to_shared_memory, to_device
import taichi as ti
import traceback

def f(child_conn, device: torch.DeviceObjType):
    ti.init(arch=ti.gpu)
    while True:
        op_name, tensors = child_conn.recv()
        tensors = to_device(tensors, device)
        try:
            if "softsplat" in op_name:
                result = raw_softsplat(op_name, tensors)
            elif "costvol" in op_name:
                result = costvol(op_name, tensors)
            elif "sepconv" in op_name:
                result = sepconv(op_name, tensors)
            else:
                raise NotImplementedError(op_name)
            child_conn.send(to_shared_memory(result))
        except:
            child_conn.send(traceback.format_exc())

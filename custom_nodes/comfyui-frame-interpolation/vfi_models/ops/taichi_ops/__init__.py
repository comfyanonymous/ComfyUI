import comfy.model_management as model_management
import torch
import torch.multiprocessing as mp
from .worker_process import f
from .utils import to_shared_memory

parent_conn, child_conn, process = None, None, None
device = model_management.get_torch_device()

def req_to_taichi_process(op_name, *tensors):
    global parent_conn, child_conn, process
    if parent_conn is None:
        mp.set_start_method('spawn', force=True)
        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(target=f, args=(child_conn, device))
        process.start()
    
    tensors = to_shared_memory(tensors)
    parent_conn.send((op_name, tensors))
    result = parent_conn.recv()
    del tensors
    
    if type(result) not in [tuple, list]:
        raise Exception(result)
        
    return [tensor.to(device) for tensor in result]

def softsplat(
    tenIn: torch.Tensor, tenFlow: torch.Tensor, tenMetric: torch.Tensor, strMode: str
):
    assert strMode.split("-")[0] in ["sum", "avg", "linear", "soft"]

    if strMode == "sum":
        assert tenMetric is None
    if strMode == "avg":
        assert tenMetric is None
    if strMode.split("-")[0] == "linear":
        assert tenMetric is not None
    if strMode.split("-")[0] == "soft":
        assert tenMetric is not None

    if strMode == "avg":
        tenIn = torch.cat(
            [
                tenIn,
                tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]]),
            ],
            1,
        )

    elif strMode.split("-")[0] == "linear":
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)

    elif strMode.split("-")[0] == "soft":
        tenIn = torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)

    # end

    tenOut = req_to_taichi_process("softsplat_out", tenIn, tenFlow)[0]

    if strMode.split("-")[0] in ["avg", "linear", "soft"]:
        tenNormalize = tenOut[:, -1:, :, :]

        if len(strMode.split("-")) == 1:
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split("-")[1] == "addeps":
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split("-")[1] == "zeroeps":
            tenNormalize[tenNormalize == 0.0] = 1.0

        elif strMode.split("-")[1] == "clipeps":
            tenNormalize = tenNormalize.clip(0.0000001, None)

        # end

        tenOut = tenOut[:, :-1, :, :] / tenNormalize
    # end

    return tenOut

def FunctionSoftsplat(tenInput, tenFlow, tenMetric, strType):
    assert tenMetric is None or tenMetric.shape[1] == 1
    assert strType in ["summation", "average", "linear", "softmax"]

    if strType == "average":
        tenInput = torch.cat(
            [
                tenInput,
                tenInput.new_ones(
                    tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3]
                ),
            ],
            1,
        )

    elif strType == "linear":
        tenInput = torch.cat([tenInput * tenMetric, tenMetric], 1)

    elif strType == "softmax":
        tenInput = torch.cat([tenInput * tenMetric.exp(), tenMetric.exp()], 1)

    # end

    tenOutput = req_to_taichi_process("softsplat_out", tenInput, tenFlow)[0]

    if strType != "summation":
        tenNormalize = tenOutput[:, -1:, :, :]

        tenNormalize[tenNormalize == 0.0] = 1.0

        tenOutput = tenOutput[:, :-1, :, :] / tenNormalize
    # end

    return tenOutput


# end


class ModuleSoftsplat(torch.nn.Module):
    def __init__(self, strType):
        super(self).__init__()

        self.strType = strType

    # end

    def forward(self, tenInput, tenFlow, tenMetric):
        return FunctionSoftsplat(tenInput, tenFlow, tenMetric, self.strType)

def softsplat_func(tenIn, tenFlow):
    return req_to_taichi_process("softsplat_out", tenIn, tenFlow)[0]

class costvol_func:
    @staticmethod
    def apply(tenOne, tenTwo):
        return req_to_taichi_process("costvol_out", tenOne, tenTwo)[0]

class sepconv_func:
    @staticmethod
    def apply(tenIn, tenVer, tenHor):
        return req_to_taichi_process("sepconv_out", tenIn, tenVer, tenHor)[0]

def init():
    one_sample = torch.ones(1, 3, 16, 16, dtype=torch.float32, device=device)
    softsplat_func(one_sample, one_sample)
    costvol_func.apply(one_sample, one_sample)
    sepconv_func.apply(one_sample, one_sample, one_sample)

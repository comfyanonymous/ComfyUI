import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter


_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if isinstance(val, dict):
        res_dict = {}
        for k, v in val.items():
            if  k!= 'cos_cis_img' and k != 'sin_cis_img':
                res_dict[k] = conversion_helper(v, conversion)
            else:
                res_dict[k] = v
        return res_dict
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_convertor(val)
        return val
    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_HALF_TYPES,)):
            val = val.float()
        return val
    return conversion_helper(val, float_conversion)


class Float16Module(torch.nn.Module):

    def __init__(self, module, args):
        super(Float16Module, self).__init__()

        self.add_module('module', module.half())

        def float16_convertor(val):
            return val.half()

        self.float16_convertor = float16_convertor

        self.config = self.module.config
        self.dtype = torch.float16

    def forward(self, *inputs, **kwargs):
        inputs = fp32_to_float16(inputs, self.float16_convertor)
        kwargs = fp32_to_float16(kwargs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        outputs = float16_to_fp32(outputs)
        return outputs

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix,
                                                          keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


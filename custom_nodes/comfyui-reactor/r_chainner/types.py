from typing import Union

from r_chainner.archs.face.gfpganv1_clean_arch import GFPGANv1Clean


PyTorchFaceModels = (GFPGANv1Clean,)
PyTorchFaceModel = Union[GFPGANv1Clean]


def is_pytorch_face_model(model: object):
    return isinstance(model, PyTorchFaceModels)

PyTorchModels = (*PyTorchFaceModels, )
PyTorchModel = Union[PyTorchFaceModel]


def is_pytorch_model(model: object):
    return isinstance(model, PyTorchModels)

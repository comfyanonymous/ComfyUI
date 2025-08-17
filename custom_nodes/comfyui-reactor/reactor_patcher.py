import os.path as osp
import glob
import logging
import insightface
from insightface.model_zoo.model_zoo import ModelRouter, PickableInferenceSession
from insightface.model_zoo.retinaface import RetinaFace
from insightface.model_zoo.landmark import Landmark
from insightface.model_zoo.attribute import Attribute
from insightface.model_zoo.inswapper import INSwapper
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.app import FaceAnalysis
from insightface.utils import DEFAULT_MP_NAME, ensure_available
from insightface.model_zoo import model_zoo
import onnxruntime
import onnx
from onnx import numpy_helper
from scripts.reactor_logger import logger


def patched_get_model_log(self, **kwargs):
    session = PickableInferenceSession(self.onnx_file, **kwargs)
    print(f'Applied providers: {session._providers}, with options: {session._provider_options}')
    inputs = session.get_inputs()
    input_cfg = inputs[0]
    input_shape = input_cfg.shape
    outputs = session.get_outputs()

    if len(outputs) >= 5:
        return RetinaFace(model_file=self.onnx_file, session=session)
    elif input_shape[2] == 192 and input_shape[3] == 192:
        return Landmark(model_file=self.onnx_file, session=session)
    elif input_shape[2] == 96 and input_shape[3] == 96:
        return Attribute(model_file=self.onnx_file, session=session)
    elif len(inputs) == 2 and input_shape[2] == 128 and input_shape[3] == 128:
        return INSwapper(model_file=self.onnx_file, session=session)
    elif len(inputs) == 2 and input_shape[2] == 256 and input_shape[3] == 256:
        return INSwapper(model_file=self.onnx_file, session=session)
    elif input_shape[2] == input_shape[3] and input_shape[2] >= 112 and input_shape[2] % 16 == 0:
        return ArcFaceONNX(model_file=self.onnx_file, session=session)
    else:
        return None

def patched_get_model(self, **kwargs):
    session = PickableInferenceSession(self.onnx_file, **kwargs)
    inputs = session.get_inputs()
    input_cfg = inputs[0]
    input_shape = input_cfg.shape
    outputs = session.get_outputs()

    if len(outputs) >= 5:
        return RetinaFace(model_file=self.onnx_file, session=session)
    elif input_shape[2] == 192 and input_shape[3] == 192:
        return Landmark(model_file=self.onnx_file, session=session)
    elif input_shape[2] == 96 and input_shape[3] == 96:
        return Attribute(model_file=self.onnx_file, session=session)
    elif len(inputs) == 2 and input_shape[2] == 128 and input_shape[3] == 128:
        return INSwapper(model_file=self.onnx_file, session=session)
    elif len(inputs) == 2 and input_shape[2] == 256 and input_shape[3] == 256:
        return INSwapper(model_file=self.onnx_file, session=session)
    elif input_shape[2] == input_shape[3] and input_shape[2] >= 112 and input_shape[2] % 16 == 0:
        return ArcFaceONNX(model_file=self.onnx_file, session=session)
    else:
        return None


def patched_faceanalysis_init(self, name=DEFAULT_MP_NAME, root='~/.insightface', allowed_modules=None, **kwargs):
    onnxruntime.set_default_logger_severity(3)
    self.models = {}
    self.model_dir = ensure_available('models', name, root=root)
    onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
    onnx_files = sorted(onnx_files)
    for onnx_file in onnx_files:
        model = model_zoo.get_model(onnx_file, **kwargs)
        if model is None:
            print('model not recognized:', onnx_file)
        elif allowed_modules is not None and model.taskname not in allowed_modules:
            print('model ignore:', onnx_file, model.taskname)
            del model
        elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
            self.models[model.taskname] = model
        else:
            print('duplicated model task type, ignore:', onnx_file, model.taskname)
            del model
    assert 'detection' in self.models
    self.det_model = self.models['detection']


def patched_faceanalysis_prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
    self.det_thresh = det_thresh
    assert det_size is not None
    self.det_size = det_size
    for taskname, model in self.models.items():
        if taskname == 'detection':
            model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
        else:
            model.prepare(ctx_id)


def patched_inswapper_init(self, model_file=None, session=None):
    self.model_file = model_file
    self.session = session
    model = onnx.load(self.model_file)
    graph = model.graph
    self.emap = numpy_helper.to_array(graph.initializer[-1])
    self.input_mean = 0.0
    self.input_std = 255.0
    if self.session is None:
        self.session = onnxruntime.InferenceSession(self.model_file, None)
    inputs = self.session.get_inputs()
    self.input_names = []
    for inp in inputs:
        self.input_names.append(inp.name)
    outputs = self.session.get_outputs()
    output_names = []
    for out in outputs:
        output_names.append(out.name)
    self.output_names = output_names
    assert len(self.output_names) == 1
    input_cfg = inputs[0]
    input_shape = input_cfg.shape
    self.input_shape = input_shape
    self.input_size = tuple(input_shape[2:4][::-1])


def pathced_retinaface_prepare(self, ctx_id, **kwargs):
    if ctx_id<0:
        self.session.set_providers(['CPUExecutionProvider'])
    nms_thresh = kwargs.get('nms_thresh', None)
    if nms_thresh is not None:
        self.nms_thresh = nms_thresh
    det_thresh = kwargs.get('det_thresh', None)
    if det_thresh is not None:
        self.det_thresh = det_thresh
    input_size = kwargs.get('input_size', None)
    if input_size is not None and self.input_size is None:
        self.input_size = input_size


def patch_insightface(get_model, faceanalysis_init, faceanalysis_prepare, inswapper_init, retinaface_prepare):
    insightface.model_zoo.model_zoo.ModelRouter.get_model = get_model
    insightface.app.FaceAnalysis.__init__ = faceanalysis_init
    insightface.app.FaceAnalysis.prepare = faceanalysis_prepare
    insightface.model_zoo.inswapper.INSwapper.__init__ = inswapper_init
    insightface.model_zoo.retinaface.RetinaFace.prepare = retinaface_prepare


# original_functions = [ModelRouter.get_model, FaceAnalysis.__init__, FaceAnalysis.prepare, INSwapper.__init__, RetinaFace.prepare]
original_functions = [patched_get_model_log, FaceAnalysis.__init__, FaceAnalysis.prepare, INSwapper.__init__, RetinaFace.prepare]
patched_functions = [patched_get_model, patched_faceanalysis_init, patched_faceanalysis_prepare, patched_inswapper_init, pathced_retinaface_prepare]


def apply_patch(console_log_level):
    if console_log_level == 0:
        patch_insightface(*patched_functions)
        logger.setLevel(logging.WARNING)
    elif console_log_level == 1:
        patch_insightface(*patched_functions)
        logger.setLevel(logging.STATUS)
    elif console_log_level == 2:
        patch_insightface(*original_functions)
        logger.setLevel(logging.INFO)

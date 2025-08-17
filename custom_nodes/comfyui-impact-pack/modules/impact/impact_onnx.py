import impact.additional_dependencies
import numpy as np
from impact import utils
import logging

impact.additional_dependencies.ensure_onnx_package()

try:
    import onnxruntime

    def onnx_inference(image, onnx_model):
        # prepare image
        pil = utils.tensor2pil(image)
        image = np.ascontiguousarray(pil)
        image = image[:, :, ::-1]  # to BGR image
        image = image.astype(np.float32)
        image -= [103.939, 116.779, 123.68]  # 'caffe' mode image preprocessing

        # do detection
        onnx_model = onnxruntime.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
        outputs = onnx_model.run(
            [s_i.name for s_i in onnx_model.get_outputs()],
            {onnx_model.get_inputs()[0].name: np.expand_dims(image, axis=0)},
        )

        labels = [op for op in outputs if op.dtype == "int32"][0]
        scores = [op for op in outputs if isinstance(op[0][0], np.float32)][0]
        boxes = [op for op in outputs if isinstance(op[0][0], np.ndarray)][0]

        # filter-out useless item
        idx = np.where(labels[0] == -1)[0][0]

        labels = labels[0][:idx]
        scores = scores[0][:idx]
        boxes = boxes[0][:idx].astype(np.uint32)

        return labels, scores, boxes
except Exception:
    logging.error("[Impact Pack] ComfyUI-Impact-Pack: 'onnxruntime' package doesn't support 'python 3.11', yet.\t{e}")

import numpy as np
import torch
from PIL import Image
from .parsing_api import onnx_inference
from ...libs.utils import install_package

class HumanParsing:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None

    def __call__(self, input_image, mask_components):
        if self.session is None:
            install_package('onnxruntime')
            import onnxruntime as ort

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # session_options.add_session_config_entry('gpu_id', str(gpu_id))
            self.session = ort.InferenceSession(self.model_path, sess_options=session_options,
                                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        parsed_image, mask = onnx_inference(self.session, input_image, mask_components)
        return parsed_image, mask


class HumanParts:

    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        # self.classes_dict = {
        #     "background": 0,
        #     "hair": 2,
        #     "glasses": 4,
        #     "top-clothes": 5,
        #     "bottom-clothes": 9,
        #     "torso-skin": 10,
        #     "face": 13,
        #     "left-arm": 14,
        #     "right-arm": 15,
        #     "left-leg": 16,
        #     "right-leg": 17,
        #     "left-foot": 18,
        #     "right-foot": 19,
        # },
        self.classes = [0, 13, 2, 4, 5, 9, 10, 14, 15, 16, 17, 18, 19]


    def __call__(self, input_image, mask_components):
        if self.session is None:
            install_package('onnxruntime')
            import onnxruntime as ort

            self.session = ort.InferenceSession(self.model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

        mask, = self.get_mask(self.session, input_image, 0, mask_components)
        return mask

    def get_mask(self, model, image, rotation, mask_components):
        image = image.squeeze(0)
        image_np = image.numpy() * 255

        pil_image = Image.fromarray(image_np.astype(np.uint8))
        original_size = pil_image.size  # to resize the mask later
        # resize to 512x512 as the model expects
        pil_image = pil_image.resize((512, 512))
        center = (256, 256)

        if rotation != 0:
            pil_image = pil_image.rotate(rotation, center=center)

        # normalize the image
        image_np = np.array(pil_image).astype(np.float32) / 127.5 - 1
        image_np = np.expand_dims(image_np, axis=0)

        # use the onnx model to get the mask
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        result = model.run([output_name], {input_name: image_np})
        result = np.array(result[0]).argmax(axis=3).squeeze(0)

        score: int = 0

        mask = np.zeros_like(result)
        for class_index in mask_components:
            detected = result == self.classes[class_index]
            mask[detected] = 255
            score += mask.sum()

        # back to the original size
        mask_image = Image.fromarray(mask.astype(np.uint8), mode="L")
        if rotation != 0:
            mask_image = mask_image.rotate(-rotation, center=center)

        mask_image = mask_image.resize(original_size)

        # and back to numpy...
        mask = np.array(mask_image).astype(np.float32) / 255

        # add 2 dimensions to match the expected output
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        # ensure to return a "binary mask_image"

        del image_np, result  # free up memory, maybe not necessary

        return (torch.from_numpy(mask.astype(np.uint8)),)
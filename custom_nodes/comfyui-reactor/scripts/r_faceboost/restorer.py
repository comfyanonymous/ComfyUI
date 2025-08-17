import sys
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

try:
    import torch.cuda as cuda
except:
    cuda = None

import comfy.utils
import folder_paths
import comfy.model_management as model_management

from scripts.reactor_logger import logger
from r_basicsr.utils.registry import ARCH_REGISTRY
from r_chainner import model_loading
from reactor_utils import (
    tensor2img,
    img2tensor,
    set_ort_session,
    prepare_cropped_face,
    normalize_cropped_face
)


if cuda is not None:
    if cuda.is_available():
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]


def get_restored_face(cropped_face,
                      face_restore_model,
                      face_restore_visibility,
                      codeformer_weight,
                      interpolation: str = "Bicubic"):

    if interpolation == "Bicubic":
        interpolate = cv2.INTER_CUBIC
    elif interpolation == "Bilinear":
        interpolate = cv2.INTER_LINEAR
    elif interpolation == "Nearest":
        interpolate = cv2.INTER_NEAREST
    elif interpolation == "Lanczos":
        interpolate = cv2.INTER_LANCZOS4
    
    face_size = 512
    if "1024" in face_restore_model.lower():
        face_size = 1024
    elif "2048" in face_restore_model.lower():
        face_size = 2048

    scale = face_size / cropped_face.shape[0]
    
    logger.status(f"Boosting the Face with {face_restore_model} | Face Size is set to {face_size} with Scale Factor = {scale} and '{interpolation}' interpolation")

    cropped_face = cv2.resize(cropped_face, (face_size, face_size), interpolation=interpolate)

    # For upscaling the base 128px face, I found bicubic interpolation to be the best compromise targeting antialiasing
    # and detail preservation. Nearest is predictably unusable, Linear produces too much aliasing, and Lanczos produces
    # too many hallucinations and artifacts/fringing.

    model_path = folder_paths.get_full_path("facerestore_models", face_restore_model)
    device = model_management.get_torch_device()

    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

    try:

        with torch.no_grad():

            if ".onnx" in face_restore_model:  # ONNX models

                ort_session = set_ort_session(model_path, providers=providers)
                ort_session_inputs = {}
                facerestore_model = ort_session

                for ort_session_input in ort_session.get_inputs():
                    if ort_session_input.name == "input":
                        cropped_face_prep = prepare_cropped_face(cropped_face)
                        ort_session_inputs[ort_session_input.name] = cropped_face_prep
                    if ort_session_input.name == "weight":
                        weight = np.array([1], dtype=np.double)
                        ort_session_inputs[ort_session_input.name] = weight

                output = ort_session.run(None, ort_session_inputs)[0][0]
                restored_face = normalize_cropped_face(output)

            else:  # PTH models

                if "codeformer" in face_restore_model.lower():
                    codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                        dim_embd=512,
                        codebook_size=1024,
                        n_head=8,
                        n_layers=9,
                        connect_list=["32", "64", "128", "256"],
                    ).to(device)
                    checkpoint = torch.load(model_path)["params_ema"]
                    codeformer_net.load_state_dict(checkpoint)
                    facerestore_model = codeformer_net.eval()
                else:
                    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
                    facerestore_model = model_loading.load_state_dict(sd).eval()
                    facerestore_model.to(device)

                output = facerestore_model(cropped_face_t, w=codeformer_weight)[
                    0] if "codeformer" in face_restore_model.lower() else facerestore_model(cropped_face_t)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

        del output
        torch.cuda.empty_cache()

    except Exception as error:

        print(f"\tFailed inference: {error}", file=sys.stderr)
        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

    if face_restore_visibility < 1:
        restored_face = cropped_face * (1 - face_restore_visibility) + restored_face * face_restore_visibility

    restored_face = restored_face.astype("uint8")
    return restored_face, scale

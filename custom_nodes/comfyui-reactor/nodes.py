import os, glob, sys
import logging

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import normalize
from torchvision.ops import masks_to_boxes

import numpy as np
import cv2
import math
from typing import List
from PIL import Image
import io
from scipy import stats
from insightface.app.common import Face
from segment_anything import sam_model_registry

from modules.processing import StableDiffusionProcessingImg2Img
from modules.shared import state
# from comfy_extras.chainner_models import model_loading
import comfy.model_management as model_management
import comfy.utils
import folder_paths

import scripts.reactor_version
from r_chainner import model_loading
from scripts.reactor_faceswap import (
    FaceSwapScript,
    get_models,
    get_current_faces_model,
    analyze_faces,
    half_det_size,
    providers
)
from scripts.reactor_swapper import (
    unload_all_models,
)
from scripts.reactor_logger import logger
from reactor_utils import (
    batch_tensor_to_pil,
    batched_pil_to_tensor,
    tensor_to_pil,
    img2tensor,
    tensor2img,
    save_face_model,
    load_face_model,
    download,
    set_ort_session,
    prepare_cropped_face,
    normalize_cropped_face,
    add_folder_path_and_extensions,
    rgba2rgb_tensor,
    progress_bar,
    progress_bar_reset
)
from reactor_patcher import apply_patch
from r_facelib.utils.face_restoration_helper import FaceRestoreHelper
from r_basicsr.utils.registry import ARCH_REGISTRY
import scripts.r_archs.codeformer_arch
import scripts.r_masking.subcore as subcore
import scripts.r_masking.core as core
import scripts.r_masking.segs as masking_segs

import scripts.reactor_sfw as sfw


models_dir = folder_paths.models_dir
REACTOR_MODELS_PATH = os.path.join(models_dir, "reactor")
FACE_MODELS_PATH = os.path.join(REACTOR_MODELS_PATH, "faces")
NSFWDET_MODEL_PATH = os.path.join(models_dir, "nsfw_detector","vit-base-nsfw-detector")

if not os.path.exists(REACTOR_MODELS_PATH):
    os.makedirs(REACTOR_MODELS_PATH)
    if not os.path.exists(FACE_MODELS_PATH):
        os.makedirs(FACE_MODELS_PATH)

dir_facerestore_models = os.path.join(models_dir, "facerestore_models")
os.makedirs(dir_facerestore_models, exist_ok=True)
folder_paths.folder_names_and_paths["facerestore_models"] = ([dir_facerestore_models], folder_paths.supported_pt_extensions)

BLENDED_FACE_MODEL = None
FACE_SIZE: int = 512
FACE_HELPER = None

if "ultralytics" not in folder_paths.folder_names_and_paths:
    add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(models_dir, "ultralytics", "bbox")], folder_paths.supported_pt_extensions)
    add_folder_path_and_extensions("ultralytics_segm", [os.path.join(models_dir, "ultralytics", "segm")], folder_paths.supported_pt_extensions)
    add_folder_path_and_extensions("ultralytics", [os.path.join(models_dir, "ultralytics")], folder_paths.supported_pt_extensions)
if "sams" not in folder_paths.folder_names_and_paths:
    add_folder_path_and_extensions("sams", [os.path.join(models_dir, "sams")], folder_paths.supported_pt_extensions)

def get_facemodels():
    models_path = os.path.join(FACE_MODELS_PATH, "*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".safetensors")]
    return models

def get_restorers():
    models_path = os.path.join(models_dir, "facerestore_models/*")
    models = glob.glob(models_path)
    models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx"))]
    if len(models) == 0:
        fr_urls = [
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.3.pth",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx",
        ]
        for model_url in fr_urls:
            model_name = os.path.basename(model_url)
            model_path = os.path.join(dir_facerestore_models, model_name)
            download(model_url, model_path, model_name)
        models = glob.glob(models_path)
        models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx"))]
    return models

def get_model_names(get_models):
    models = get_models()
    names = []
    for x in models:
        names.append(os.path.basename(x))
    names.sort(key=str.lower)
    names.insert(0, "none")
    return names

def model_names():
    models = get_models()
    return {os.path.basename(x): x for x in models}


class reactor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "input_image": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "face_restore_model": (get_model_names(get_restorers),),
                "face_restore_visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "detect_gender_input": (["no","female","male"], {"default": "no"}),
                "detect_gender_source": (["no","female","male"], {"default": "no"}),
                "input_faces_index": ("STRING", {"default": "0"}),
                "source_faces_index": ("STRING", {"default": "0"}),
                "console_log_level": ([0, 1, 2], {"default": 1}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
                "face_boost": ("FACE_BOOST",),
            },
            "hidden": {"faces_order": "FACES_ORDER"},
        }

    RETURN_TYPES = ("IMAGE","FACE_MODEL","IMAGE")
    RETURN_NAMES = ("SWAPPED_IMAGE","FACE_MODEL","ORIGINAL_IMAGE")
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def __init__(self):
        # self.face_helper = None
        self.faces_order = ["large-small", "large-small"]
        # self.face_size = FACE_SIZE
        self.face_boost_enabled = False
        self.restore = True
        self.boost_model = None
        self.interpolation = "Bicubic"
        self.boost_model_visibility = 1
        self.boost_cf_weight = 0.5

    def restore_face(
            self,
            input_image,
            face_restore_model,
            face_restore_visibility,
            codeformer_weight,
            facedetection,
        ):

        result = input_image

        if face_restore_model != "none" and not model_management.processing_interrupted():

            global FACE_SIZE, FACE_HELPER

            self.face_helper = FACE_HELPER

            faceSize = 512
            if "1024" in face_restore_model.lower():
                faceSize = 1024
            elif "2048" in face_restore_model.lower():
                faceSize = 2048

            logger.status(f"Restoring with {face_restore_model} | Face Size is set to {faceSize}")

            model_path = folder_paths.get_full_path("facerestore_models", face_restore_model)

            device = model_management.get_torch_device()

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

            elif ".onnx" in face_restore_model:

                ort_session = set_ort_session(model_path, providers=providers)
                ort_session_inputs = {}
                facerestore_model = ort_session

            else:

                sd = comfy.utils.load_torch_file(model_path, safe_load=True)
                facerestore_model = model_loading.load_state_dict(sd).eval()
                facerestore_model.to(device)

            if faceSize != FACE_SIZE or self.face_helper is None:
                self.face_helper = FaceRestoreHelper(1, face_size=faceSize, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=device)
                FACE_SIZE = faceSize
                FACE_HELPER = self.face_helper

            # Copying Tensor to CPU (if it isn't) to convert torch.Tensor to np.ndarray
            image_np = 255. * result.cpu().numpy()

            total_images = image_np.shape[0]

            out_images = []
            
            pbar = progress_bar(total_images)

            for i in range(total_images):

                # if total_images > 1:
                #     logger.status(f"Restoring {i}")

                cur_image_np = image_np[i,:, :, ::-1]

                original_resolution = cur_image_np.shape[0:2]

                if facerestore_model is None or self.face_helper is None:
                    return result

                self.face_helper.clean_all()
                self.face_helper.read_image(cur_image_np)
                self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                self.face_helper.align_warp_face()

                restored_face = None

                for idx, cropped_face in enumerate(self.face_helper.cropped_faces):

                    # if ".pth" in face_restore_model:
                    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                    try:

                        with torch.no_grad():

                            if ".onnx" in face_restore_model: # ONNX models

                                for ort_session_input in ort_session.get_inputs():
                                    if ort_session_input.name == "input":
                                        cropped_face_prep = prepare_cropped_face(cropped_face)
                                        ort_session_inputs[ort_session_input.name] = cropped_face_prep
                                    if ort_session_input.name == "weight":
                                        weight = np.array([ 1 ], dtype = np.double)
                                        ort_session_inputs[ort_session_input.name] = weight

                                output = ort_session.run(None, ort_session_inputs)[0][0]
                                restored_face = normalize_cropped_face(output)

                            else: # PTH models

                                output = facerestore_model(cropped_face_t, w=codeformer_weight)[0] if "codeformer" in face_restore_model.lower() else facerestore_model(cropped_face_t)[0]
                                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

                        del output
                        torch.cuda.empty_cache()

                    except Exception as error:

                        print(f"\tFailed inference: {error}", file=sys.stderr)
                        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                    if face_restore_visibility < 1:
                        restored_face = cropped_face * (1 - face_restore_visibility) + restored_face * face_restore_visibility

                    restored_face = restored_face.astype("uint8")
                    self.face_helper.add_restored_face(restored_face)

                self.face_helper.get_inverse_affine(None)

                restored_img = self.face_helper.paste_faces_to_input_image()
                restored_img = restored_img[:, :, ::-1]

                if original_resolution != restored_img.shape[0:2]:
                    restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_AREA)

                self.face_helper.clean_all()

                # out_images[i] = restored_img
                out_images.append(restored_img)

                if state.interrupted or model_management.processing_interrupted():
                    logger.status("Interrupted by User")
                    return input_image
                
                pbar.update(1)

            restored_img_np = np.array(out_images).astype(np.float32) / 255.0
            restored_img_tensor = torch.from_numpy(restored_img_np)

            result = restored_img_tensor

            progress_bar_reset(pbar)

        return result

    def execute(self, enabled, input_image, swap_model, detect_gender_source, detect_gender_input, source_faces_index, input_faces_index, console_log_level, face_restore_model,face_restore_visibility, codeformer_weight, facedetection, source_image=None, face_model=None, faces_order=None, face_boost=None):

        device = model_management.get_torch_device()

        if isinstance(input_image, torch.Tensor) and input_image.device != device:
            input_image = input_image.to(device)

        if face_boost is not None:
            self.face_boost_enabled = face_boost["enabled"]
            self.boost_model = face_boost["boost_model"]
            self.interpolation = face_boost["interpolation"]
            self.boost_model_visibility = face_boost["visibility"]
            self.boost_cf_weight = face_boost["codeformer_weight"]
            self.restore = face_boost["restore_with_main_after"]
        else:
            self.face_boost_enabled = False

        if faces_order is None:
            faces_order = self.faces_order

        apply_patch(console_log_level)

        if not enabled:
            return (input_image,face_model)
        elif source_image is None and face_model is None:
            logger.error("Please provide 'source_image' or `face_model`")
            return (input_image,face_model)

        if face_model == "none":
            face_model = None

        script = FaceSwapScript()
        pil_images = batch_tensor_to_pil(input_image)

        # NSFW checker
        logger.status("Checking for any unsafe content...")
        pbar = progress_bar(len(pil_images))
        pil_images_sfw = []
        for img in pil_images:
            if state.interrupted or model_management.processing_interrupted():
                logger.status("Interrupted by User")
                break
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            if not sfw.nsfw_image(img_byte_arr, NSFWDET_MODEL_PATH):
                pil_images_sfw.append(img)
            pbar.update(1)
        pil_images = pil_images_sfw
        # # #
        progress_bar_reset(pbar)

        if len(pil_images) > 0:

            if source_image is not None:
                source = tensor_to_pil(source_image)
            else:
                source = None
            p = StableDiffusionProcessingImg2Img(pil_images)
            script.process(
                p=p,
                img=source,
                enable=True,
                source_faces_index=source_faces_index,
                faces_index=input_faces_index,
                model=swap_model,
                swap_in_source=True,
                swap_in_generated=True,
                gender_source=detect_gender_source,
                gender_target=detect_gender_input,
                face_model=face_model,
                faces_order=faces_order,
                # face boost:
                face_boost_enabled=self.face_boost_enabled,
                face_restore_model=self.boost_model,
                face_restore_visibility=self.boost_model_visibility,
                codeformer_weight=self.boost_cf_weight,
                interpolation=self.interpolation,
            )
            result = batched_pil_to_tensor(p.init_images)
            original_image = batched_pil_to_tensor(pil_images)

            if face_model is None:
                current_face_model = get_current_faces_model()
                face_model_to_provide = current_face_model[0] if (current_face_model is not None and len(current_face_model) > 0) else face_model
            else:
                face_model_to_provide = face_model

            if self.restore or not self.face_boost_enabled:
                result = reactor.restore_face(self,result,face_restore_model,face_restore_visibility,codeformer_weight,facedetection)

        else:
            image_black = Image.new("RGB", (512, 512))
            result = batched_pil_to_tensor([image_black])
            face_model_to_provide = None
            original_image = result

        return (result,face_model_to_provide,original_image)


class ReActorPlusOpt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "input_image": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "face_restore_model": (get_model_names(get_restorers),),
                "face_restore_visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
                "options": ("OPTIONS",),
                "face_boost": ("FACE_BOOST",),
            }
        }

    RETURN_TYPES = ("IMAGE","FACE_MODEL","IMAGE")
    RETURN_NAMES = ("SWAPPED_IMAGE","FACE_MODEL","ORIGINAL_IMAGE")
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def __init__(self):
        # self.face_helper = None
        self.faces_order = ["large-small", "large-small"]
        self.detect_gender_input = "no"
        self.detect_gender_source = "no"
        self.input_faces_index = "0"
        self.source_faces_index = "0"
        self.console_log_level = 1
        # self.face_size = 512
        self.face_boost_enabled = False
        self.restore = True
        self.boost_model = None
        self.interpolation = "Bicubic"
        self.boost_model_visibility = 1
        self.boost_cf_weight = 0.5

    def execute(self, enabled, input_image, swap_model, facedetection, face_restore_model, face_restore_visibility, codeformer_weight, source_image=None, face_model=None, options=None, face_boost=None):

        if options is not None:
            self.faces_order = [options["input_faces_order"], options["source_faces_order"]]
            self.console_log_level = options["console_log_level"]
            self.detect_gender_input = options["detect_gender_input"]
            self.detect_gender_source = options["detect_gender_source"]
            self.input_faces_index = options["input_faces_index"]
            self.source_faces_index = options["source_faces_index"]

        if face_boost is not None:
            self.face_boost_enabled = face_boost["enabled"]
            self.restore = face_boost["restore_with_main_after"]
        else:
            self.face_boost_enabled = False

        result = reactor.execute(
            self,enabled,input_image,swap_model,self.detect_gender_source,self.detect_gender_input,self.source_faces_index,self.input_faces_index,self.console_log_level,face_restore_model,face_restore_visibility,codeformer_weight,facedetection,source_image,face_model,self.faces_order, face_boost=face_boost
        )

        return result


class LoadFaceModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_model": (get_model_names(get_facemodels),),
            }
        }

    RETURN_TYPES = ("FACE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "🌌 ReActor"

    def load_model(self, face_model):
        self.face_model = face_model
        self.face_models_path = FACE_MODELS_PATH
        if self.face_model != "none":
            face_model_path = os.path.join(self.face_models_path, self.face_model)
            out = load_face_model(face_model_path)
        else:
            out = None
        return (out, )


class ReActorWeight:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "faceswap_weight": (["0%", "12.5%", "25%", "37.5%", "50%", "62.5%", "75%", "87.5%", "100%"], {"default": "50%"}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
            }
        }
    
    RETURN_TYPES = ("IMAGE","FACE_MODEL")
    RETURN_NAMES = ("INPUT_IMAGE","FACE_MODEL")
    FUNCTION = "set_weight"

    OUTPUT_NODE = True

    CATEGORY = "🌌 ReActor"

    def set_weight(self, input_image, faceswap_weight, face_model=None, source_image=None):

        if input_image is None:
            logger.error("Please provide `input_image`")
            return (input_image,None)
        
        if source_image is None and face_model is None:
            logger.error("Please provide `source_image` or `face_model`")
            return (input_image,None)

        weight = float(faceswap_weight.split("%")[0])

        images = []
        faces = [] if face_model is None else [face_model]
        embeddings = [] if face_model is None else [face_model.embedding]

        if weight == 0:
            images = [input_image]
            faces = []
            embeddings = []
        elif weight == 100:
            if face_model is None:
                images = [source_image]
        else:
            if weight > 50:
                images = [input_image]
                count = round(100/(100-weight))
            else:
                if face_model is None:
                    images = [source_image]
                count = round(100/(weight))
            for i in range(count-1):
                if weight > 50:
                    if face_model is None:
                        images.append(source_image)
                    else:
                        faces.append(face_model)
                        embeddings.append(face_model.embedding)
                else:
                    images.append(input_image)
        
        images_list: List[Image.Image] = []

        apply_patch(1)

        if len(images) > 0:

            for image in images:
                img = tensor_to_pil(image)
                images_list.append(img)

            for image in images_list:
                face = BuildFaceModel.build_face_model(self,image)
                if isinstance(face, str):
                    continue
                faces.append(face)
                embeddings.append(face.embedding)
        
        if len(faces) > 0:
            blended_embedding = np.mean(embeddings, axis=0)
            blended_face = Face(
                bbox=faces[0].bbox,
                kps=faces[0].kps,
                det_score=faces[0].det_score,
                landmark_3d_68=faces[0].landmark_3d_68,
                pose=faces[0].pose,
                landmark_2d_106=faces[0].landmark_2d_106,
                embedding=blended_embedding,
                gender=faces[0].gender,
                age=faces[0].age
            )
            if blended_face is None:
                no_face_msg = "Something went wrong, please try another set of images"
                logger.error(no_face_msg)

        return (input_image,blended_face)


class BuildFaceModel:
    def __init__(self):
        self.output_dir = FACE_MODELS_PATH

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_mode": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "send_only": ("BOOLEAN", {"default": False, "label_off": "NO", "label_on": "YES"}),
                "face_model_name": ("STRING", {"default": "default"}),
                "compute_method": (["Mean", "Median", "Mode"], {"default": "Mean"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "face_models": ("FACE_MODEL",),
            }
        }

    RETURN_TYPES = ("FACE_MODEL",)
    FUNCTION = "blend_faces"

    OUTPUT_NODE = True

    CATEGORY = "🌌 ReActor"

    def build_face_model(self, image: Image.Image, det_size=(640, 640)):
        logging.StreamHandler.terminator = "\n"
        if image is None:
            error_msg = "Please load an Image"
            logger.error(error_msg)
            return error_msg
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face_model = analyze_faces(image, det_size)

        if len(face_model) == 0:
            print("")
            det_size_half = half_det_size(det_size)
            face_model = analyze_faces(image, det_size_half)
            if face_model is not None and len(face_model) > 0:
                print("...........................................................", end=" ")

        if face_model is not None and len(face_model) > 0:
            return face_model[0]
        else:
            no_face_msg = "No face found, please try another image"
            # logger.error(no_face_msg)
            return no_face_msg

    def blend_faces(self, save_mode, send_only, face_model_name, compute_method, images=None, face_models=None):
        global BLENDED_FACE_MODEL
        blended_face: Face = BLENDED_FACE_MODEL

        if send_only and blended_face is None:
            send_only = False

        if (images is not None or face_models is not None) and not send_only:

            faces = []
            embeddings = []

            apply_patch(1)

            if images is not None:
                images_list: List[Image.Image] = batch_tensor_to_pil(images)

                n = len(images_list)

                for i,image in enumerate(images_list):
                    logging.StreamHandler.terminator = " "
                    logger.status(f"Building Face Model {i+1} of {n}...")
                    face = self.build_face_model(image)
                    if isinstance(face, str):
                        logger.error(f"No faces found in image {i+1}, skipping")
                        continue
                    else:
                        print(f"{int(((i+1)/n)*100)}%")
                    faces.append(face)
                    embeddings.append(face.embedding)

            elif face_models is not None:

                n = len(face_models)

                for i,face_model in enumerate(face_models):
                    logging.StreamHandler.terminator = " "
                    logger.status(f"Extracting Face Model {i+1} of {n}...")
                    face = face_model
                    if isinstance(face, str):
                        logger.error(f"No faces found for face_model {i+1}, skipping")
                        continue
                    else:
                        print(f"{int(((i+1)/n)*100)}%")
                    faces.append(face)
                    embeddings.append(face.embedding)

            logging.StreamHandler.terminator = "\n"
            if len(faces) > 0:
                # compute_method_name = "Mean" if compute_method == 0 else "Median" if compute_method == 1 else "Mode"
                logger.status(f"Blending with Compute Method '{compute_method}'...")
                blended_embedding = np.mean(embeddings, axis=0) if compute_method == "Mean" else np.median(embeddings, axis=0) if compute_method == "Median" else stats.mode(embeddings, axis=0)[0].astype(np.float32)
                blended_face = Face(
                    bbox=faces[0].bbox,
                    kps=faces[0].kps,
                    det_score=faces[0].det_score,
                    landmark_3d_68=faces[0].landmark_3d_68,
                    pose=faces[0].pose,
                    landmark_2d_106=faces[0].landmark_2d_106,
                    embedding=blended_embedding,
                    gender=faces[0].gender,
                    age=faces[0].age
                )
                if blended_face is not None:
                    BLENDED_FACE_MODEL = blended_face
                    if save_mode:
                        face_model_path = os.path.join(FACE_MODELS_PATH, face_model_name + ".safetensors")
                        save_face_model(blended_face,face_model_path)
                        # done_msg = f"Face model has been saved to '{face_model_path}'"
                        # logger.status(done_msg)
                    logger.status("--Done!--")
                    # return (blended_face,)
                else:
                    no_face_msg = "Something went wrong, please try another set of images"
                    logger.error(no_face_msg)
                    # return (blended_face,)
            # logger.status("--Done!--")
        if images is None and face_models is None:
            logger.error("Please provide `images` or `face_models`")
        return (blended_face,)


class SaveFaceModel:
    def __init__(self):
        self.output_dir = FACE_MODELS_PATH

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_mode": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "face_model_name": ("STRING", {"default": "default"}),
                "select_face_index": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_model"

    OUTPUT_NODE = True

    CATEGORY = "🌌 ReActor"

    def save_model(self, save_mode, face_model_name, select_face_index, image=None, face_model=None, det_size=(640, 640)):
        if save_mode and image is not None:
            source = tensor_to_pil(image)
            source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
            apply_patch(1)
            logger.status("Building Face Model...")
            face_model_raw = analyze_faces(source, det_size)
            if len(face_model_raw) == 0:
                det_size_half = half_det_size(det_size)
                face_model_raw = analyze_faces(source, det_size_half)
            try:
                face_model = face_model_raw[select_face_index]
            except:
                logger.error("No face(s) found")
                return face_model_name
            logger.status("--Done!--")
        if save_mode and (face_model != "none" or face_model is not None):
            face_model_path = os.path.join(self.output_dir, face_model_name + ".safetensors")
            save_face_model(face_model,face_model_path)
        if image is None and face_model is None:
            logger.error("Please provide `face_model` or `image`")
        return face_model_name


class RestoreFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "model": (get_model_names(get_restorers),),
                "visibility": ("FLOAT", {"default": 1, "min": 0.0, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    # def __init__(self):
    #     self.face_helper = None
    #     self.face_size = 512

    def execute(self, image, model, visibility, codeformer_weight, facedetection):
        result = reactor.restore_face(self,image,model,visibility,codeformer_weight,facedetection)
        return (result,)


class MaskHelper:
    def __init__(self):
        self.labels = "all"
        self.detailer_hook = None
        self.device_mode = "AUTO"
        self.detection_hint = "center-1"
        self._sam_cache = {}
        self._bbox_cache = {}

    @classmethod
    def INPUT_TYPES(s):
        bboxs = ["bbox/"+x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/"+x for x in folder_paths.get_filename_list("ultralytics_segm")]
        sam_models = [x for x in folder_paths.get_filename_list("sams") if 'hq' not in x]
        return {
            "required": {
                "image": ("IMAGE",),
                "swapped_image": ("IMAGE",),
                "bbox_model_name": (bboxs + segms, ),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                "bbox_drop_size": ("INT", {"min": 1, "max": 8192, "step": 1, "default": 10}),
                "sam_model_name": (sam_models, ),
                "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_hint_use_negative": (["False", "Small", "Outter"], ),
                "morphology_operation": (["dilate", "erode", "open", "close"],),
                "morphology_distance": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "blur_radius": ("INT", {"default": 9, "min": 0, "max": 48, "step": 1}),
                "sigma_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 3., "step": 0.01}),
            },
            "optional": {
                "mask_optional": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE","MASK","IMAGE","IMAGE")
    RETURN_NAMES = ("IMAGE","MASK","MASK_PREVIEW","SWAPPED_FACE")
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self, image, swapped_image, bbox_model_name, bbox_threshold, bbox_dilation, bbox_crop_factor, bbox_drop_size, sam_model_name, sam_dilation, sam_threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative, morphology_operation, morphology_distance, blur_radius, sigma_factor, mask_optional=None):
        device = model_management.get_torch_device()

        # Оптимально перемещаем тензоры
        if isinstance(image, torch.Tensor) and image.device != device:
            image = image.to(device)

        if isinstance(swapped_image, torch.Tensor) and swapped_image.device != device:
            swapped_image = swapped_image.to(device)

        if mask_optional is not None:
            combined_mask = mask_optional
        else:
            # Load and cache BBox model
            if bbox_model_name not in self._bbox_cache:
                bbox_model_path = folder_paths.get_full_path("ultralytics", bbox_model_name)
                model = subcore.load_yolo(bbox_model_path)
                self._bbox_cache[bbox_model_name] = subcore.UltraBBoxDetector(model)
            bbox_detector = self._bbox_cache[bbox_model_name]

            segs_all, seg_labels = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, bbox_drop_size, self.detailer_hook)

            if self.labels != 'all':
                labels = self.labels.split(',') if isinstance(self.labels, str) else self.labels
                segs_all, _ = masking_segs.filter(segs_all, labels)

            # Load and cache SAM model
            if sam_model_name not in self._sam_cache:
                sam_model_path = folder_paths.get_full_path("sams", sam_model_name)
                if 'vit_h' in sam_model_name:
                    model_kind = 'vit_h'
                elif 'vit_l' in sam_model_name:
                    model_kind = 'vit_l'
                else:
                    model_kind = 'vit_b'
                sam = sam_model_registry[model_kind](checkpoint=sam_model_path)
                size = os.path.getsize(sam_model_path)
                sam.safe_to = core.SafeToGPU(size)
                sam.safe_to.to_device(sam, device)
                sam.is_auto_mode = self.device_mode == "AUTO"
                self._sam_cache[sam_model_name] = sam
            else:
                sam = self._sam_cache[sam_model_name]

            # Handle batched input
            if image.ndim == 4:
                combined_masks = []
                for i in range(image.shape[0]):
                    segs_i = segs_all[i] if i < len(segs_all) else []
                    segs_tuple = ([segs_i], seg_labels) if isinstance(segs_i, dict) else (segs_i, seg_labels)
                    image_device = image.to(device) if image.device != device else image
                    image_i = image_device[i]
                    mask_i, _ = core.make_sam_mask_segmented(
                        sam, segs_tuple, image_i, self.detection_hint,
                        sam_dilation, sam_threshold, bbox_expansion,
                        mask_hint_threshold, mask_hint_use_negative
                    )
                    combined_masks.append(mask_i)
                combined_mask = torch.stack(combined_masks)
            else:
                image_device = image.to(device) if image.device != device else image
                combined_mask, _ = core.make_sam_mask_segmented(
                    sam, (segs_all, seg_labels), image_device, self.detection_hint,
                    sam_dilation, sam_threshold, bbox_expansion,
                    mask_hint_threshold, mask_hint_use_negative
                )

        # Morph operations
        if morphology_operation == "dilate":
            # print(f"max before: {combined_mask.max()}, min: {combined_mask.min()}, sum: {combined_mask.sum()}")
            combined_mask = self.iterative_morphology(combined_mask, morphology_distance, op="dilate")
            # print(f"after morph: {combined_mask.max()}, min: {combined_mask.min()}, sum: {combined_mask.sum()}")
        elif morphology_operation == "erode":
            combined_mask = self.iterative_morphology(combined_mask, morphology_distance, op="erode")
        elif morphology_operation == "open":
            combined_mask = self.iterative_morphology(self.iterative_morphology(combined_mask, morphology_distance, op="erode"), morphology_distance, op="dilate")
        elif morphology_operation == "close":
            combined_mask = self.iterative_morphology(self.iterative_morphology(combined_mask, morphology_distance, op="dilate"), morphology_distance, op="erode")

        # Gaussian blur
        if blur_radius > 0:
            blur = T.GaussianBlur(kernel_size=blur_radius * 2 + 1, sigma=sigma_factor)
            mask_blurred = blur(combined_mask.unsqueeze(1)).squeeze(1)
        else:
            mask_blurred = combined_mask

        # Apply mask to swapped image (basic RGBA composite)
        swapped_image = swapped_image.to(device) if swapped_image.device != device else swapped_image
        swapped_rgba = core.tensor2rgba(swapped_image)

        mask_image_final = mask_blurred
        
        # *** CUT BY MASK ***:
    
        if len(swapped_image.shape) < 4:
            C = 1
        else:
            C = swapped_image.shape[3]

        # We operate on RGBA to keep the code clean and then convert back after
        swapped_image = core.tensor2rgba(swapped_image)
        mask = core.tensor2mask(mask_image_final)

        # Scale the mask to be a matching size if it isn't
        B, H, W, _ = swapped_image.shape
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
        MB, _, _ = mask.shape

        if MB < B:
            assert(B % MB == 0)
            mask = mask.repeat(B // MB, 1, 1)

        # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
        is_empty = ~torch.gt(torch.max(torch.reshape(mask,[MB, H * W]), dim=1).values, 0.)
        mask[is_empty,0,0] = 1.
        boxes = masks_to_boxes(mask)
        mask[is_empty,0,0] = 0.

        min_x = boxes[:,0]
        min_y = boxes[:,1]
        max_x = boxes[:,2]
        max_y = boxes[:,3]

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        use_width = int(torch.max(width).item())
        use_height = int(torch.max(height).item())

        alpha_mask = torch.ones((B, H, W, 4))
        alpha_mask[:,:,:,3] = mask

        alpha_mask = alpha_mask.to(device) if alpha_mask.device != device else alpha_mask

        swapped_image = swapped_image * alpha_mask

        cutted_image = torch.zeros((B, use_height, use_width, 4))
        for i in range(0, B):
            if not is_empty[i]:
                ymin = int(min_y[i].item())
                ymax = int(max_y[i].item())
                xmin = int(min_x[i].item())
                xmax = int(max_x[i].item())
                single = (swapped_image[i, ymin:ymax+1, xmin:xmax+1,:]).unsqueeze(0)
                resized = torch.nn.functional.interpolate(single.permute(0, 3, 1, 2), size=(use_height, use_width), mode='bicubic').permute(0, 2, 3, 1)
                cutted_image[i] = resized[0]
        
        # Preserve our type unless we were previously RGB and added non-opaque alpha due to the mask size
        if C == 1:
            cutted_image = core.tensor2mask(cutted_image)
        elif C == 3 and torch.min(cutted_image[:,:,:,3]) == 1:
            cutted_image = core.tensor2rgb(cutted_image)

        # *** PASTE BY MASK ***:

        image_base = core.tensor2rgba(image)
        image_to_paste = core.tensor2rgba(cutted_image)
        mask = core.tensor2mask(mask_image_final)

        # Scale the mask to be a matching size if it isn't
        B, H, W, C = image_base.shape
        MB = mask.shape[0]
        PB = image_to_paste.shape[0]

        if B < PB:
            assert(PB % B == 0)
            image_base = image_base.repeat(PB // B, 1, 1, 1)
        B, H, W, C = image_base.shape
        if MB < B:
            assert(B % MB == 0)
            mask = mask.repeat(B // MB, 1, 1)
        elif B < MB:
            assert(MB % B == 0)
            image_base = image_base.repeat(MB // B, 1, 1, 1)
        if PB < B:
            assert(B % PB == 0)
            image_to_paste = image_to_paste.repeat(B // PB, 1, 1, 1)

        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
        MB, MH, MW = mask.shape

        # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
        is_empty = ~torch.gt(torch.max(torch.reshape(mask,[MB, MH * MW]), dim=1).values, 0.)
        mask[is_empty,0,0] = 1.
        boxes = masks_to_boxes(mask)
        mask[is_empty,0,0] = 0.

        min_x = boxes[:,0]
        min_y = boxes[:,1]
        max_x = boxes[:,2]
        max_y = boxes[:,3]
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        target_width = max_x - min_x + 1
        target_height = max_y - min_y + 1

        result = image_base.detach().clone()
        face_segment = mask_image_final
        
        for i in range(0, MB):
            if is_empty[i]:
                continue
            else:
                image_index = i
                SB, SH, SW, _ = image_to_paste.shape

                # Figure out the desired size
                width = int(target_width[i].item())
                height = int(target_height[i].item())

                width = SW
                height = SH

                # Resize the image we're pasting if needed
                resized_image = image_to_paste[i].unsqueeze(0)

                pasting = torch.ones([H, W, C])
                ymid = float(mid_y[i].item())
                ymin = int(math.floor(ymid - height / 2)) + 1
                ymax = int(math.floor(ymid + height / 2)) + 1
                xmid = float(mid_x[i].item())
                xmin = int(math.floor(xmid - width / 2)) + 1
                xmax = int(math.floor(xmid + width / 2)) + 1

                _, source_ymax, source_xmax, _ = resized_image.shape
                source_ymin, source_xmin = 0, 0

                if xmin < 0:
                    source_xmin = abs(xmin)
                    xmin = 0
                if ymin < 0:
                    source_ymin = abs(ymin)
                    ymin = 0
                if xmax > W:
                    source_xmax -= (xmax - W)
                    xmax = W
                if ymax > H:
                    source_ymax -= (ymax - H)
                    ymax = H

                pasting[ymin:ymax, xmin:xmax, :] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, :]
                pasting[:, :, 3] = 1.

                pasting_alpha = torch.zeros([H, W])
                pasting_alpha[ymin:ymax, xmin:xmax] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, 3]

                paste_mask = torch.min(pasting_alpha, mask[i]).unsqueeze(2).repeat(1, 1, 4)

                pasting = pasting.to(device) if pasting.device != device else pasting
                paste_mask = paste_mask.to(device) if paste_mask.device != device else paste_mask

                result[image_index] = pasting * paste_mask + result[image_index] * (1. - paste_mask)

                face_segment = result

                face_segment[...,3] = mask[i]

                result = rgba2rgb_tensor(result)

        try:
            torch.cuda.empty_cache()
        except:
            pass

        return (result, combined_mask, mask_blurred, face_segment)

    def iterative_morphology(self, image, distance, op="dilate"):
        if distance <= 0:
            return image
        image = image.unsqueeze(1)  # shape [B, 1, H, W] or [1, 1, H, W]
        for _ in range(distance):
            if op == "dilate":
                image = F.max_pool2d(image, kernel_size=3, stride=1, padding=1)
            elif op == "erode":
                image = -F.max_pool2d(-image, kernel_size=3, stride=1, padding=1)
        return image.squeeze(1)


class ImageDublicator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "count": ("INT", {"default": 1, "min": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGES",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self, image, count):
        images = [image for i in range(count)]
        return (images,)


class ImageRGBA2RGB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self, image):
        out = rgba2rgb_tensor(image)
        return (out,)


class MakeFaceModelBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_model1": ("FACE_MODEL",),
            },
            "optional": {
                "face_model2": ("FACE_MODEL",),
                "face_model3": ("FACE_MODEL",),
                "face_model4": ("FACE_MODEL",),
                "face_model5": ("FACE_MODEL",),
                "face_model6": ("FACE_MODEL",),
                "face_model7": ("FACE_MODEL",),
                "face_model8": ("FACE_MODEL",),
                "face_model9": ("FACE_MODEL",),
                "face_model10": ("FACE_MODEL",),
            },
        }

    RETURN_TYPES = ("FACE_MODEL",)
    RETURN_NAMES = ("FACE_MODELS",)
    FUNCTION = "execute"

    CATEGORY = "🌌 ReActor"

    def execute(self, **kwargs):
        if len(kwargs) > 0:
            face_models = [value for value in kwargs.values()]
            return (face_models,)
        else:
            logger.error("Please provide at least 1 `face_model`")
            return (None,)


class ReActorOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_faces_order": (
                    ["left-right","right-left","top-bottom","bottom-top","small-large","large-small"], {"default": "large-small"}
                ),
                "input_faces_index": ("STRING", {"default": "0"}),
                "detect_gender_input": (["no","female","male"], {"default": "no"}),
                "source_faces_order": (
                    ["left-right","right-left","top-bottom","bottom-top","small-large","large-small"], {"default": "large-small"}
                ),
                "source_faces_index": ("STRING", {"default": "0"}),
                "detect_gender_source": (["no","female","male"], {"default": "no"}),
                "console_log_level": ([0, 1, 2], {"default": 1}),
            }
        }

    RETURN_TYPES = ("OPTIONS",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self,input_faces_order, input_faces_index, detect_gender_input, source_faces_order, source_faces_index, detect_gender_source, console_log_level):
        options: dict = {
            "input_faces_order": input_faces_order,
            "input_faces_index": input_faces_index,
            "detect_gender_input": detect_gender_input,
            "source_faces_order": source_faces_order,
            "source_faces_index": source_faces_index,
            "detect_gender_source": detect_gender_source,
            "console_log_level": console_log_level,
        }
        return (options, )


class ReActorFaceBoost:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "boost_model": (get_model_names(get_restorers),),
                "interpolation": (["Nearest","Bilinear","Bicubic","Lanczos"], {"default": "Bicubic"}),
                "visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "restore_with_main_after": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FACE_BOOST",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self,enabled,boost_model,interpolation,visibility,codeformer_weight,restore_with_main_after):
        face_boost: dict = {
            "enabled": enabled,
            "boost_model": boost_model,
            "interpolation": interpolation,
            "visibility": visibility,
            "codeformer_weight": codeformer_weight,
            "restore_with_main_after": restore_with_main_after,
        }
        return (face_boost, )

class ReActorUnload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trigger": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self, trigger):
        unload_all_models()
        return (trigger,)


NODE_CLASS_MAPPINGS = {
    # --- MAIN NODES ---
    "ReActorFaceSwap": reactor,
    "ReActorFaceSwapOpt": ReActorPlusOpt,
    "ReActorOptions": ReActorOptions,
    "ReActorFaceBoost": ReActorFaceBoost,
    "ReActorMaskHelper": MaskHelper,
    "ReActorSetWeight": ReActorWeight,
    # --- Operations with Face Models ---
    "ReActorSaveFaceModel": SaveFaceModel,
    "ReActorLoadFaceModel": LoadFaceModel,
    "ReActorBuildFaceModel": BuildFaceModel,
    "ReActorMakeFaceModelBatch": MakeFaceModelBatch,
    # --- Additional Nodes ---
    "ReActorRestoreFace": RestoreFace,
    "ReActorImageDublicator": ImageDublicator,
    "ImageRGBA2RGB": ImageRGBA2RGB,
    "ReActorUnload": ReActorUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # --- MAIN NODES ---
    "ReActorFaceSwap": "ReActor 🌌 Fast Face Swap",
    "ReActorFaceSwapOpt": "ReActor 🌌 Fast Face Swap [OPTIONS]",
    "ReActorOptions": "ReActor 🌌 Options",
    "ReActorFaceBoost": "ReActor 🌌 Face Booster",
    "ReActorMaskHelper": "ReActor 🌌 Masking Helper",
    "ReActorSetWeight": "ReActor 🌌 Set Face Swap Weight",
    # --- Operations with Face Models ---
    "ReActorSaveFaceModel": "Save Face Model 🌌 ReActor",
    "ReActorLoadFaceModel": "Load Face Model 🌌 ReActor",
    "ReActorBuildFaceModel": "Build Blended Face Model 🌌 ReActor",
    "ReActorMakeFaceModelBatch": "Make Face Model Batch 🌌 ReActor",
    # --- Additional Nodes ---
    "ReActorRestoreFace": "Restore Face 🌌 ReActor",
    "ReActorImageDublicator": "Image Dublicator (List) 🌌 ReActor",
    "ImageRGBA2RGB": "Convert RGBA to RGB 🌌 ReActor",
    "ReActorUnload": "Unload ReActor Models 🌌 ReActor",
}

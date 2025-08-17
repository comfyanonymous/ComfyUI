import os, glob

from PIL import Image

import modules.scripts as scripts
# from modules.upscaler import Upscaler, UpscalerData
from modules import scripts, scripts_postprocessing
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
)
from modules.shared import state
from scripts.reactor_logger import logger
from scripts.reactor_swapper import (
    swap_face,
    swap_face_many,
    get_current_faces_model,
    analyze_faces,
    half_det_size,
    providers
)
import folder_paths
import comfy.model_management as model_management


def get_models():
    swappers = [
        "insightface",
        "reswapper"
    ]
    models_list = []
    for folder in swappers:
        models_folder = folder + "/*"
        models_path = os.path.join(folder_paths.models_dir,models_folder)
        models = glob.glob(models_path)
        models = [x for x in models if x.endswith(".onnx") or x.endswith(".pth")]
        models_list.extend(models)
    return models_list


class FaceSwapScript(scripts.Script):

    def process(
        self,
        p: StableDiffusionProcessing,
        img,
        enable,
        source_faces_index,
        faces_index,
        model,
        swap_in_source,
        swap_in_generated,
        gender_source,
        gender_target,
        face_model,
        faces_order,
        face_boost_enabled,
        face_restore_model,
        face_restore_visibility,
        codeformer_weight,
        interpolation,
    ):
        self.enable = enable
        if self.enable:

            self.source = img    
            self.swap_in_generated = swap_in_generated
            self.gender_source = gender_source
            self.gender_target = gender_target
            self.model = model
            self.face_model = face_model
            self.faces_order = faces_order
            self.face_boost_enabled = face_boost_enabled
            self.face_restore_model = face_restore_model
            self.face_restore_visibility = face_restore_visibility
            self.codeformer_weight = codeformer_weight
            self.interpolation = interpolation
            self.source_faces_index = [
                int(x) for x in source_faces_index.strip(",").split(",") if x.isnumeric()
            ]
            self.faces_index = [
                int(x) for x in faces_index.strip(",").split(",") if x.isnumeric()
            ]
            if len(self.source_faces_index) == 0:
                self.source_faces_index = [0]
            if len(self.faces_index) == 0:
                self.faces_index = [0]
            
            if self.gender_source is None or self.gender_source == "no":
                self.gender_source = 0
            elif self.gender_source  == "female":
                self.gender_source = 1
            elif self.gender_source  == "male":
                self.gender_source = 2
            
            if self.gender_target is None or self.gender_target == "no":
                self.gender_target = 0
            elif self.gender_target  == "female":
                self.gender_target = 1
            elif self.gender_target  == "male":
                self.gender_target = 2

            # if self.source is not None:
            if isinstance(p, StableDiffusionProcessingImg2Img) and swap_in_source:
                logger.status(f"Working: source face index %s, target face index %s", self.source_faces_index, self.faces_index)

                if len(p.init_images) == 1:

                    result = swap_face(
                        self.source,
                        p.init_images[0],
                        source_faces_index=self.source_faces_index,
                        faces_index=self.faces_index,
                        model=self.model,
                        gender_source=self.gender_source,
                        gender_target=self.gender_target,
                        face_model=self.face_model,
                        faces_order=self.faces_order,
                        face_boost_enabled=self.face_boost_enabled,
                        face_restore_model=self.face_restore_model,
                        face_restore_visibility=self.face_restore_visibility,
                        codeformer_weight=self.codeformer_weight,
                        interpolation=self.interpolation,
                    )
                    p.init_images[0] = result

                    # for i in range(len(p.init_images)):
                    #     if state.interrupted or model_management.processing_interrupted():
                    #         logger.status("Interrupted by User")
                    #         break
                    #     if len(p.init_images) > 1:
                    #         logger.status(f"Swap in %s", i)
                    #     result = swap_face(
                    #         self.source,
                    #         p.init_images[i],
                    #         source_faces_index=self.source_faces_index,
                    #         faces_index=self.faces_index,
                    #         model=self.model,
                    #         gender_source=self.gender_source,
                    #         gender_target=self.gender_target,
                    #         face_model=self.face_model,
                    #     )
                    #     p.init_images[i] = result

                elif len(p.init_images) > 1:
                    result = swap_face_many(
                        self.source,
                        p.init_images,
                        source_faces_index=self.source_faces_index,
                        faces_index=self.faces_index,
                        model=self.model,
                        gender_source=self.gender_source,
                        gender_target=self.gender_target,
                        face_model=self.face_model,
                        faces_order=self.faces_order,
                        face_boost_enabled=self.face_boost_enabled,
                        face_restore_model=self.face_restore_model,
                        face_restore_visibility=self.face_restore_visibility,
                        codeformer_weight=self.codeformer_weight,
                        interpolation=self.interpolation,
                    )
                    p.init_images = result

                logger.status("--Done!--")
            # else:
            #     logger.error(f"Please provide a source face")

    def postprocess_batch(self, p, *args, **kwargs):
        if self.enable:
            images = kwargs["images"]

    def postprocess_image(self, p, script_pp: scripts.PostprocessImageArgs, *args):
        if self.enable and self.swap_in_generated:
            if self.source is not None:
                logger.status(f"Working: source face index %s, target face index %s", self.source_faces_index, self.faces_index)
                image: Image.Image = script_pp.image
                result = swap_face(
                    self.source,
                    image,
                    source_faces_index=self.source_faces_index,
                    faces_index=self.faces_index,
                    model=self.model,
                    upscale_options=self.upscale_options,
                    gender_source=self.gender_source,
                    gender_target=self.gender_target,
                )
                try:
                    pp = scripts_postprocessing.PostprocessedImage(result)
                    pp.info = {}
                    p.extra_generation_params.update(pp.info)
                    script_pp.image = pp.image
                except:
                    logger.error(f"Cannot create a result image")

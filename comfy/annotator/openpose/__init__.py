import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .body import Body
from .hand import Hand
from annotator.util import annotator_ckpts_path


body_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth"
hand_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth"


class OpenposeDetector:
    def __init__(self):
        body_modelpath = os.path.join(annotator_ckpts_path, "body_pose_model.pth")
        hand_modelpath = os.path.join(annotator_ckpts_path, "hand_pose_model.pth")

        if not os.path.exists(hand_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(body_model_path, model_dir=annotator_ckpts_path)
            load_file_from_url(hand_model_path, model_dir=annotator_ckpts_path)

        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)

    def __call__(self, oriImg, hand=False):
        oriImg = oriImg[:, :, ::-1].copy()
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            canvas = np.zeros_like(oriImg)
            canvas = util.draw_bodypose(canvas, candidate, subset)
            if hand:
                hands_list = util.handDetect(candidate, subset, oriImg)
                all_hand_peaks = []
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :])
                    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                    all_hand_peaks.append(peaks)
                canvas = util.draw_handpose(canvas, all_hand_peaks)
            return canvas, dict(candidate=candidate.tolist(), subset=subset.tolist())

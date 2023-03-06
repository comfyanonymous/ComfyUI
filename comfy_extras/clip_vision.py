from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPImageProcessor
from comfy.sd import load_torch_file
import os

class ClipVisionModel():
    def __init__(self):
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config.json")
        config = CLIPVisionConfig.from_json_file(json_config)
        self.model = CLIPVisionModel(config)
        self.processor = CLIPImageProcessor(crop_size=224,
                                            do_center_crop=True,
                                            do_convert_rgb=True,
                                            do_normalize=True,
                                            do_resize=True,
                                            image_mean=[ 0.48145466,0.4578275,0.40821073],
                                            image_std=[0.26862954,0.26130258,0.27577711],
                                            resample=3, #bicubic
                                            size=224)

    def load_sd(self, sd):
        self.model.load_state_dict(sd, strict=False)

    def encode_image(self, image):
        inputs = self.processor(images=[image[0]], return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs

def load(ckpt_path):
    clip_data = load_torch_file(ckpt_path)
    clip = ClipVisionModel()
    clip.load_sd(clip_data)
    return clip

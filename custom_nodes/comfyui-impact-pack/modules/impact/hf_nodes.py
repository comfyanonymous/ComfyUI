import comfy
import re
from impact import utils


hf_transformer_model_urls = [
    "rizvandwiki/gender-classification-2",
    "NTQAI/pedestrian_gender_recognition",
    "Leilab/gender_class",
    "ProjectPersonal/GenderClassifier",
    "crangana/trained-gender",
    "cledoux42/GenderNew_v002",
    "ivensamdh/genderage2"
]


class HF_TransformersClassifierProvider:
    @classmethod
    def INPUT_TYPES(s):
        global hf_transformer_model_urls
        return {"required": {
                        "preset_repo_id": (hf_transformer_model_urls + ['Manual repo id'],),
                        "manual_repo_id": ("STRING", {"multiline": False}),
                        "device_mode": (["AUTO", "Prefer GPU", "CPU"],),
                     },
                }

    RETURN_TYPES = ("TRANSFORMERS_CLASSIFIER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/HuggingFace"

    def doit(self, preset_repo_id, manual_repo_id, device_mode):
        from transformers import pipeline

        if preset_repo_id == 'Manual repo id':
            url = manual_repo_id
        else:
            url = preset_repo_id

        if device_mode != 'CPU':
            device = comfy.model_management.get_torch_device()
        else:
            device = "cpu"

        classifier = pipeline('image-classification', model=url, device=device)

        return (classifier,)


preset_classify_expr = [
    '#Female > #Male',
    '#Female < #Male',
    'female > 0.5',
    'male > 0.5',
    'Age16to25 > 0.1',
    'Age50to69 > 0.1',
]

symbolic_label_map = {
    '#Female': {'female', 'Female', 'Human Female', 'woman', 'women', 'girl'},
    '#Male': {'male', 'Male', 'Human Male', 'man', 'men', 'boy'}
}

def is_numeric_string(input_str):
    return re.match(r'^-?\d+(\.\d+)?$', input_str) is not None


classify_expr_pattern = r'([^><= ]+)\s*(>|<|>=|<=|=)\s*([^><= ]+)'


class SEGS_Classify:
    @classmethod
    def INPUT_TYPES(s):
        global preset_classify_expr
        return {"required": {
                        "classifier": ("TRANSFORMERS_CLASSIFIER",),
                        "segs": ("SEGS",),
                        "preset_expr": (preset_classify_expr + ['Manual expr'],),
                        "manual_expr": ("STRING", {"multiline": False}),
                     },
                "optional": {
                     "ref_image_opt": ("IMAGE", ),
                    }
                }

    RETURN_TYPES = ("SEGS", "SEGS", "STRING")
    RETURN_NAMES = ("filtered_SEGS", "remained_SEGS", "detected_labels")
    OUTPUT_IS_LIST = (False, False, True)

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/HuggingFace"

    @staticmethod
    def lookup_classified_label_score(score_infos, label):
        global symbolic_label_map

        if label.startswith('#'):
            if label not in symbolic_label_map:
                return None
            else:
                label = symbolic_label_map[label]
        else:
            label = {label}

        for x in score_infos:
            if x['label'] in label:
                return x['score']

        return None

    def doit(self, classifier, segs, preset_expr, manual_expr, ref_image_opt=None):
        if preset_expr == 'Manual expr':
            expr_str = manual_expr
        else:
            expr_str = preset_expr

        match = re.match(classify_expr_pattern, expr_str)

        if match is None:
            return (segs[0], []), segs, []

        a = match.group(1)
        op = match.group(2)
        b = match.group(3)

        a_is_lab = not is_numeric_string(a)
        b_is_lab = not is_numeric_string(b)

        classified = []
        remained_SEGS = []
        provided_labels = set()

        for seg in segs[1]:
            cropped_image = None

            if seg.cropped_image is not None:
                cropped_image = seg.cropped_image
            elif ref_image_opt is not None:
                # take from original image
                cropped_image = utils.crop_image(ref_image_opt, seg.crop_region)

            if cropped_image is not None:
                cropped_image = utils.to_pil(cropped_image)
                res = classifier(cropped_image)
                classified.append((seg, res))

                for x in res:
                    provided_labels.add(x['label'])
            else:
                remained_SEGS.append(seg)

        filtered_SEGS = []
        for seg, res in classified:
            if a_is_lab:
                avalue = SEGS_Classify.lookup_classified_label_score(res, a)
            else:
                avalue = a

            if b_is_lab:
                bvalue = SEGS_Classify.lookup_classified_label_score(res, b)
            else:
                bvalue = b

            if avalue is None or bvalue is None:
                remained_SEGS.append(seg)
                continue

            avalue = float(avalue)
            bvalue = float(bvalue)

            if op == '>':
                cond = avalue > bvalue
            elif op == '<':
                cond = avalue < bvalue
            elif op == '>=':
                cond = avalue >= bvalue
            elif op == '<=':
                cond = avalue <= bvalue
            else:
                cond = avalue == bvalue

            if cond:
                filtered_SEGS.append(seg)
            else:
                remained_SEGS.append(seg)

        return (segs[0], filtered_SEGS), (segs[0], remained_SEGS), list(provided_labels)

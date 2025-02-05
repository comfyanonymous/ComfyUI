from typing import List, Union, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from typing_extensions import TypedDict, NotRequired

from comfy.component_model.tensor_types import RGBImageBatch, MaskBatch
from comfy.language.language_types import TOKENS_TYPE_NAME, LanguageModel
from comfy.language.transformers_model_management import TransformersManagedModel
from comfy.nodes.package_typing import CustomNode, InputTypes, ValidatedNodeResult
from comfy.utils import pil2mask

TASKS = ['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>', '<OD>', '<DENSE_REGION_CAPTION>', '<REGION_PROPOSAL>', '<CAPTION_TO_PHRASE_GROUNDING>', '<REFERRING_EXPRESSION_SEGMENTATION>', '<REGION_TO_SEGMENTATION>', '<OPEN_VOCABULARY_DETECTION>', '<REGION_TO_CATEGORY>', '<REGION_TO_DESCRIPTION>', '<OCR>', '<OCR_WITH_REGION>']
TASKS_TYPE_NAME = "FLORENCE2_TASK"
FLORENCE2_OUTPUT_TYPE_NAME = "FLORENCE2_OUTPUT"


class BoundingBoxResult(TypedDict):
    bboxes: List[List[float]]  # List of [x1, y1, x2, y2] coordinates
    labels: List[str]
    scores: Optional[List[float]]  # Only present if score mode is used


class QuadBoxResult(TypedDict):
    quad_boxes: List[List[float]]  # List of [x1, y1, x2, y2, x3, y3, x4, y4] coordinates
    labels: List[str]


class PolygonResult(TypedDict):
    polygons: List[List[float]]  # List of [x1, y1, x2, y2, ...] coordinates
    labels: List[str]


class BBoxesAndPolygonsResult(TypedDict):
    bboxes: List[List[float]]
    bboxes_labels: List[str]
    polygons: List[List[float]]
    polygons_labels: List[str]


PostProcessResult = TypedDict('PostProcessResult', {
    '<OCR>': NotRequired[Union[str, QuadBoxResult]],  # pure_text or ocr
    '<OCR_WITH_REGION>': NotRequired[QuadBoxResult],  # ocr
    '<CAPTION>': NotRequired[str],  # pure_text
    '<DETAILED_CAPTION>': NotRequired[str],  # pure_text
    '<MORE_DETAILED_CAPTION>': NotRequired[str],  # pure_text
    '<OD>': NotRequired[BoundingBoxResult],  # description_with_bboxes
    '<DENSE_REGION_CAPTION>': NotRequired[BoundingBoxResult],  # description_with_bboxes
    '<CAPTION_TO_PHRASE_GROUNDING>': NotRequired[BoundingBoxResult],  # phrase_grounding
    '<REFERRING_EXPRESSION_SEGMENTATION>': NotRequired[PolygonResult],  # polygons
    '<REGION_TO_SEGMENTATION>': NotRequired[PolygonResult],  # polygons
    '<OPEN_VOCABULARY_DETECTION>': NotRequired[BBoxesAndPolygonsResult],  # description_with_bboxes_or_polygons
    '<REGION_TO_CATEGORY>': NotRequired[str],  # pure_text
    '<REGION_TO_DESCRIPTION>': NotRequired[str],  # pure_text
    '<REGION_TO_OCR>': NotRequired[str],  # pure_text
    '<REGION_PROPOSAL>': NotRequired[BoundingBoxResult]  # bboxes
})


def draw_polygons(image: Image, prediction: PolygonResult) -> Image:
    """
    Draws segmentation masks with polygons on an image.

    Parameters:
    - image_path: Path to the image file.
    - prediction: Dictionary containing 'polygons' and 'labels' keys.
                  'polygons' is a list of lists, each containing vertices of a polygon.
                  'labels' is a list of labels corresponding to each polygon.
    - fill_mask: Boolean indicating whether to fill the polygons with color.
    """
    # Load the image

    draw = ImageDraw.Draw(image)

    # Set up scale factor if needed (use 1 if not scaling)
    scale = 1

    # Iterate over polygons and labels
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            _polygon = (_polygon * scale).reshape(-1).tolist()

            # Draw the polygon
            draw.polygon(_polygon, fill='white')
    return image


class Florence2TaskTokenize(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "task": (TASKS, {"default": TASKS[0]})
            },
            "optional": {
                "images": ("IMAGE", {}),
            }
        }

    CATEGORY = "language"
    RETURN_TYPES = (TOKENS_TYPE_NAME, TASKS_TYPE_NAME)
    RETURN_NAMES = ("tokens",)
    FUNCTION = "execute"

    def execute(self, model: LanguageModel, prompt: str, images: List[torch.Tensor] | torch.Tensor = None, task: str = "") -> ValidatedNodeResult:
        return model.tokenize(prompt, images, task + prompt),


class Florence2PostProcess(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL",),
                "generated_text": ("STRING", {"forceInput": True}),
                "task": (TASKS, {"default": TASKS[0]})
            },
            "optional": {
                "images": ("IMAGE", {}),
            }
        }

    CATEGORY = "language"
    RETURN_TYPES = (FLORENCE2_OUTPUT_TYPE_NAME,)
    RETURN_NAMES = ("florence2 output",)
    FUNCTION = "execute"

    def execute(self, model: TransformersManagedModel, generated_text: str = "", task: str = "", images: RGBImageBatch = None) -> tuple[PostProcessResult]:
        assert hasattr(model.processor, "post_process_generation")
        return model.processor.post_process_generation(generated_text, task=task, image_size=(images.shape[-2], images.shape[-3])),


class Florence2OutputToMask(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "florence2_output": (FLORENCE2_OUTPUT_TYPE_NAME, {}),
            },
            "optional": {
                "images": ("IMAGE", {}),
            }
        }

    CATEGORY = "language"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    def execute(self, florence2_output: PostProcessResult, images: RGBImageBatch = None) -> tuple[MaskBatch]:
        image = Image.new('RGB', (images.shape[-2], images.shape[-3]), color='black')
        for prediction in ('<REFERRING_EXPRESSION_SEGMENTATION>', '<REGION_TO_SEGMENTATION>', '<OPEN_VOCABULARY_DETECTION>'):
            if prediction in florence2_output:
                image = draw_polygons(image, florence2_output[prediction])
        return pil2mask(image),


NODE_CLASS_MAPPINGS = {}
for cls in (
        Florence2PostProcess,
        Florence2TaskTokenize,
        Florence2OutputToMask
):
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
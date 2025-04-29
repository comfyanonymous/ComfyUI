from __future__ import annotations



from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, conint


class RecraftStyle:
    def __init__(self, style: str, substyle: str=None):
        self.style = style
        self.substyle = substyle


class RecraftIO:
    STYLEV3 = "RECRAFT_V3_STYLE"


class RecraftStyleV3(str, Enum):
    #any = 'any' NOTE: this does not work for some reason... why?
    realistic_image = 'realistic_image'
    digital_illustration = 'digital_illustration'
    vector_illustration = 'vector_illustration'
    logo_raster = 'logo_raster'


def get_v3_substyles(style_v3: str, include_none=True) -> list[str]:
    substyles: list[str] = []
    if include_none:
        substyles.append("None")
    return substyles + dict_recraft_substyles_v3.get(style_v3, [])


dict_recraft_substyles_v3 = {
    RecraftStyleV3.realistic_image: [
        "b_and_w",
        "enterprise",
        "evening_light",
        "faded_nostalgia",
        "forest_life",
        "hard_flash",
        "hdr",
        "motion_blur",
        "mystic_naturalism",
        "natural_light",
        "natural_tones",
        "organic_calm",
        "real_life_glow",
        "retro_realism",
        "retro_snapshot",
        "studio_portrait",
        "urban_drama",
        "village_realism",
        "warm_folk"
    ],
    RecraftStyleV3.digital_illustration: [
        "2d_art_poster",
        "2d_art_poster_2",
        "antiquarian",
        "bold_fantasy",
        "child_book",
        "child_books",
        "cover",
        "crosshatch",
        "digital_engraving",
        "engraving_color",
        "expressionism",
        "freehand_details",
        "grain",
        "grain_20",
        "graphic_intensity",
        "hand_drawn",
        "hand_drawn_outline",
        "handmade_3d",
        "hard_comics",
        "infantile_sketch",
        "long_shadow",
        "modern_folk",
        "multicolor",
        "neon_calm",
        "noir",
        "nostalgic_pastel",
        "outline_details",
        "pastel_gradient",
        "pastel_sketch",
        "pixel_art",
        "plastic",
        "pop_art",
        "pop_renaissance",
        "seamless",
        "street_art",
        "tablet_sketch",
        "urban_glow",
        "urban_sketching",
        "vanilla_dreams",
        "young_adult_book",
        "young_adult_book_2"
    ],
    RecraftStyleV3.vector_illustration: [
        "bold_stroke",
        "chemistry",
        "colored_stencil",
        "contour_pop_art",
        "cosmics",
        "cutout",
        "depressive",
        "editorial",
        "emotional_flat",
        "engraving",
        "infographical",
        "line_art",
        "line_circuit",
        "linocut",
        "marker_outline",
        "mosaic",
        "naivector",
        "roundish_flat",
        "seamless",
        "segmented_colors",
        "sharp_contrast",
        "thin",
        "vector_photo",
        "vivid_shapes"
    ],
    RecraftStyleV3.logo_raster: [
        "emblem_graffiti",
        "emblem_pop_art",
        "emblem_punk",
        "emblem_stamp",
        "emblem_vintage"
    ],
}


class RecraftModel(str, Enum):
    recraftv3 = 'recraftv3'
    recraftv2 = 'recraftv2'


class RecraftImageSize(str, Enum):
    res_1024x1024 = '1024x1024'
    res_1365x1024 = '1365x1024'
    res_1024x1365 = '1024x1365'
    res_1536x1024 = '1536x1024'
    res_1024x1536 = '1024x1536'
    res_1820x1024 = '1820x1024'
    res_1024x1820 = '1024x1820'
    res_1024x2048 = '1024x2048'
    res_2048x1024 = '2048x1024'
    res_1434x1024 = '1434x1024'
    res_1024x1434 = '1024x1434'
    res_1024x1280 = '1024x1280'
    res_1280x1024 = '1280x1024'
    res_1024x1707 = '1024x1707'
    res_1707x1024 = '1707x1024'


class RecraftImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description='The text prompt describing the image to generate')
    size: RecraftImageSize = Field(..., description='The size of the generated image (e.g., "1024x1024")')
    n: conint(ge=1, le=6) = Field(..., description='The number of images to generate')
    negative_prompts: Optional[str] = Field(None, description='A text description of undesired elements on an image')
    model: Optional[RecraftModel] = Field(RecraftModel.recraftv3, description='The model to use for generation (e.g., "recraftv3")')
    style: Optional[str] = Field(None, description='The style to apply to the generated image (e.g., "digital_illustration")')
    substyle: Optional[str] = Field(None, description='The substyle to apply to the generated image, depending on the style input')
    # text_layout
    # controls


class RecraftReturnedObject(BaseModel):
    image_id: str = Field(..., description='Unique identifier for the generated image')
    url: str = Field(..., description='URL to access the generated image')


class RecraftImageGenerationResponse(BaseModel):
    created: int = Field(..., description='Unix timestamp when the generation was created')
    credits: int = Field(..., description='Number of credits used for the generation')
    data: list[RecraftReturnedObject] = Field(..., description=' Array of generated image information')

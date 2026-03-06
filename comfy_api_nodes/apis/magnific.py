from typing import TypedDict

from pydantic import AliasChoices, BaseModel, Field, model_validator


class InputPortraitMode(TypedDict):
    portrait_mode: str
    portrait_style: str
    portrait_beautifier: str


class InputAdvancedSettings(TypedDict):
    advanced_settings: str
    whites: int
    blacks: int
    brightness: int
    contrast: int
    saturation: int
    engine: str
    transfer_light_a: str
    transfer_light_b: str
    fixed_generation: bool


class InputSkinEnhancerMode(TypedDict):
    mode: str
    skin_detail: int
    optimized_for: str


class ImageUpscalerCreativeRequest(BaseModel):
    image: str = Field(...)
    scale_factor: str = Field(...)
    optimized_for: str = Field(...)
    prompt: str | None = Field(None)
    creativity: int = Field(...)
    hdr: int = Field(...)
    resemblance: int = Field(...)
    fractality: int = Field(...)
    engine: str = Field(...)


class ImageUpscalerPrecisionV2Request(BaseModel):
    image: str = Field(...)
    sharpen: int = Field(...)
    smart_grain: int = Field(...)
    ultra_detail: int = Field(...)
    flavor: str = Field(...)
    scale_factor: int = Field(...)


class ImageRelightAdvancedSettingsRequest(BaseModel):
    whites: int = Field(...)
    blacks: int = Field(...)
    brightness: int = Field(...)
    contrast: int = Field(...)
    saturation: int = Field(...)
    engine: str = Field(...)
    transfer_light_a: str = Field(...)
    transfer_light_b: str = Field(...)
    fixed_generation: bool = Field(...)


class ImageRelightRequest(BaseModel):
    image: str = Field(...)
    prompt: str | None = Field(None)
    transfer_light_from_reference_image: str | None = Field(None)
    light_transfer_strength: int = Field(...)
    interpolate_from_original: bool = Field(...)
    change_background: bool = Field(...)
    style: str = Field(...)
    preserve_details: bool = Field(...)
    advanced_settings: ImageRelightAdvancedSettingsRequest | None = Field(...)


class ImageStyleTransferRequest(BaseModel):
    image: str = Field(...)
    reference_image: str = Field(...)
    prompt: str | None = Field(None)
    style_strength: int = Field(...)
    structure_strength: int = Field(...)
    is_portrait: bool = Field(...)
    portrait_style: str | None = Field(...)
    portrait_beautifier: str | None = Field(...)
    flavor: str = Field(...)
    engine: str = Field(...)
    fixed_generation: bool = Field(...)


class ImageSkinEnhancerCreativeRequest(BaseModel):
    image: str = Field(...)
    sharpen: int = Field(...)
    smart_grain: int = Field(...)


class ImageSkinEnhancerFaithfulRequest(BaseModel):
    image: str = Field(...)
    sharpen: int = Field(...)
    smart_grain: int = Field(...)
    skin_detail: int = Field(...)


class ImageSkinEnhancerFlexibleRequest(BaseModel):
    image: str = Field(...)
    sharpen: int = Field(...)
    smart_grain: int = Field(...)
    optimized_for: str = Field(...)


class TaskResponse(BaseModel):
    """Unified response model that handles both wrapped and unwrapped API responses."""

    task_id: str = Field(...)
    status: str = Field(validation_alias=AliasChoices("status", "task_status"))
    generated: list[str] | None = Field(None)

    @model_validator(mode="before")
    @classmethod
    def unwrap_data(cls, values: dict) -> dict:
        if "data" in values and isinstance(values["data"], dict):
            return values["data"]
        return values

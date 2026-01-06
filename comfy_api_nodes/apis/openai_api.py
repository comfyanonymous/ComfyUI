from pydantic import BaseModel, Field


class Datum2(BaseModel):
    b64_json: str | None = Field(None, description="Base64 encoded image data")
    revised_prompt: str | None = Field(None, description="Revised prompt")
    url: str | None = Field(None, description="URL of the image")


class InputTokensDetails(BaseModel):
    image_tokens: int | None = None
    text_tokens: int | None = None


class Usage(BaseModel):
    input_tokens: int | None = None
    input_tokens_details: InputTokensDetails | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class OpenAIImageGenerationResponse(BaseModel):
    data: list[Datum2] | None = None
    usage: Usage | None = None


class OpenAIImageEditRequest(BaseModel):
    background: str | None = Field(None, description="Background transparency")
    model: str = Field(...)
    moderation: str | None = Field(None)
    n: int | None = Field(None, description="The number of images to generate")
    output_compression: int | None = Field(None, description="Compression level for JPEG or WebP (0-100)")
    output_format: str | None = Field(None)
    prompt: str = Field(...)
    quality: str | None = Field(None, description="Size of the image (e.g., 1024x1024, 1536x1024, auto)")
    size: str | None = Field(None, description="Size of the output image")


class OpenAIImageGenerationRequest(BaseModel):
    background: str | None = Field(None, description="Background transparency")
    model: str | None = Field(None)
    moderation: str | None = Field(None)
    n: int | None = Field(
        None,
        description="The number of images to generate.",
    )
    output_compression: int | None = Field(None, description="Compression level for JPEG or WebP (0-100)")
    output_format: str | None = Field(None)
    prompt: str = Field(...)
    quality: str | None = Field(None, description="The quality of the generated image")
    size: str | None = Field(None, description="Size of the image (e.g., 1024x1024, 1536x1024, auto)")
    style: str | None = Field(None, description="Style of the image (only for dall-e-3)")

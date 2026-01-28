from typing import TypedDict

from pydantic import BaseModel, Field, model_validator


class InputGenerateType(TypedDict):
    generate_type: str
    polygon_type: str
    pbr: bool


class Hunyuan3DViewImage(BaseModel):
    ViewType: str = Field(..., description="Valid values: back, left, right.")
    ViewImageUrl: str = Field(...)


class To3DProTaskRequest(BaseModel):
    Model: str = Field(...)
    Prompt: str | None = Field(None)
    ImageUrl: str | None = Field(None)
    MultiViewImages: list[Hunyuan3DViewImage] | None = Field(None)
    EnablePBR: bool | None = Field(...)
    FaceCount: int | None = Field(...)
    GenerateType: str | None = Field(...)
    PolygonType: str | None = Field(...)


class RequestError(BaseModel):
    Code: str = Field("")
    Message: str = Field("")


class To3DProTaskCreateResponse(BaseModel):
    JobId: str | None = Field(None)
    Error: RequestError | None = Field(None)

    @model_validator(mode="before")
    @classmethod
    def unwrap_data(cls, values: dict) -> dict:
        if "Response" in values and isinstance(values["Response"], dict):
            return values["Response"]
        return values


class ResultFile3D(BaseModel):
    Type: str = Field(...)
    Url: str = Field(...)
    PreviewImageUrl: str = Field("")


class To3DProTaskResultResponse(BaseModel):
    ErrorCode: str = Field("")
    ErrorMessage: str = Field("")
    ResultFile3Ds: list[ResultFile3D] = Field([])
    Status: str = Field(...)

    @model_validator(mode="before")
    @classmethod
    def unwrap_data(cls, values: dict) -> dict:
        if "Response" in values and isinstance(values["Response"], dict):
            return values["Response"]
        return values


class To3DProTaskQueryRequest(BaseModel):
    JobId: str = Field(...)

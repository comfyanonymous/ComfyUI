from typing import Optional

from comfy_api_nodes.apis import GeminiGenerationConfig, GeminiContent, GeminiSafetySetting, GeminiSystemInstructionContent, GeminiTool, GeminiVideoMetadata
from pydantic import BaseModel


class GeminiImageConfig(BaseModel):
    aspectRatio: Optional[str] = None


class GeminiImageGenerationConfig(GeminiGenerationConfig):
    responseModalities: Optional[list[str]] = None
    imageConfig: Optional[GeminiImageConfig] = None


class GeminiImageGenerateContentRequest(BaseModel):
    contents: list[GeminiContent]
    generationConfig: Optional[GeminiImageGenerationConfig] = None
    safetySettings: Optional[list[GeminiSafetySetting]] = None
    systemInstruction: Optional[GeminiSystemInstructionContent] = None
    tools: Optional[list[GeminiTool]] = None
    videoMetadata: Optional[GeminiVideoMetadata] = None

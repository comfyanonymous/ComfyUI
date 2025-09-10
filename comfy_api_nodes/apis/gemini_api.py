from __future__ import annotations

from typing import List, Optional

from comfy_api_nodes.apis import GeminiGenerationConfig, GeminiContent, GeminiSafetySetting, GeminiSystemInstructionContent, GeminiTool, GeminiVideoMetadata
from pydantic import BaseModel


class GeminiImageGenerationConfig(GeminiGenerationConfig):
    responseModalities: Optional[List[str]] = None


class GeminiImageGenerateContentRequest(BaseModel):
    contents: List[GeminiContent]
    generationConfig: Optional[GeminiImageGenerationConfig] = None
    safetySettings: Optional[List[GeminiSafetySetting]] = None
    systemInstruction: Optional[GeminiSystemInstructionContent] = None
    tools: Optional[List[GeminiTool]] = None
    videoMetadata: Optional[GeminiVideoMetadata] = None

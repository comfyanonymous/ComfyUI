from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

# IMPORTANT: The type definitions specified in pyproject.toml for custom nodes
# must remain synchronized with the corresponding files in the https://github.com/Comfy-Org/comfy-cli/blob/main/comfy_cli/registry/types.py.
# Any changes to one must be reflected in the other to maintain consistency.

class NodeVersion(BaseModel):
    changelog: str
    dependencies: List[str]
    deprecated: bool
    id: str
    version: str
    download_url: str


class Node(BaseModel):
    id: str
    name: str
    description: str
    author: Optional[str] = None
    license: Optional[str] = None
    icon: Optional[str] = None
    repository: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    latest_version: Optional[NodeVersion] = None


class PublishNodeVersionResponse(BaseModel):
    node_version: NodeVersion
    signedUrl: str


class URLs(BaseModel):
    homepage: str = Field(default="", alias="Homepage")
    documentation: str = Field(default="", alias="Documentation")
    repository: str = Field(default="", alias="Repository")
    issues: str = Field(default="", alias="Issues")


class Model(BaseModel):
    location: str
    model_url: str


class ComfyConfig(BaseModel):
    publisher_id: str = Field(default="", alias="PublisherId")
    display_name: str = Field(default="", alias="DisplayName")
    icon: str = Field(default="", alias="Icon")
    models: List[Model] = Field(default_factory=list, alias="Models")
    includes: List[str] = Field(default_factory=list)
    web: Optional[str] = None
    banner_url: str = ""

class License(BaseModel):
    file: str = ""
    text: str = ""


class ProjectConfig(BaseModel):
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    requires_python: str = Field(default=">= 3.9", alias="requires-python")
    dependencies: List[str] = Field(default_factory=list)
    license: License = Field(default_factory=License)
    urls: URLs = Field(default_factory=URLs)
    supported_os: List[str] = Field(default_factory=list)
    supported_accelerators: List[str] = Field(default_factory=list)
    supported_comfyui_version: str = ""
    supported_comfyui_frontend_version: str = ""

    @field_validator('license', mode='before')
    @classmethod
    def validate_license(cls, v):
        if isinstance(v, str):
            return License(text=v)
        elif isinstance(v, dict):
            return License(**v)
        elif isinstance(v, License):
            return v
        else:
            return License()


class PyProjectConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    tool_comfy: ComfyConfig = Field(default_factory=ComfyConfig)


class PyProjectSettings(BaseSettings):
    project: dict = Field(default_factory=dict)

    tool: dict = Field(default_factory=dict)

    model_config = SettingsConfigDict(extra='allow')

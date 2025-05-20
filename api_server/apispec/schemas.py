from marshmallow import Schema, fields, validate

# Common Schema
class ErrorResponseSchema(Schema):
    """Common error response model"""
    error = fields.Dict(keys=fields.Str(), values=fields.Raw(), required=True, description="Error details")
    node_errors = fields.Dict(required=False, description="Node-specific errors")

# Queue-related Schema
class PromptQueueItemSchema(Schema):
    """Queue item model"""
    prompt_id = fields.Str(description="Task ID")
    number = fields.Float(description="Task number")

class PromptResponseSchema(Schema):
    """Response model for submitting a generation task"""
    prompt_id = fields.Str(description="Task ID")
    number = fields.Float(description="Task number")
    node_errors = fields.Dict(description="Node validation errors")

class QueueStatusSchema(Schema):
    """Queue status response model"""
    exec_info = fields.Dict(description="Execution info")

class QueueRequestSchema(Schema):
    """Queue operation request model"""
    clear = fields.Bool(description="Whether to clear the queue", required=False)
    delete = fields.List(fields.Str(), description="List of queue item IDs to delete", required=False)

# System-related Schema
class SystemStatsSchema(Schema):
    """System status response model"""
    system = fields.Dict(description="System info")
    devices = fields.List(fields.Dict(), description="Device info")

# Model-related Schema
class ModelsListSchema(Schema):
    """Model type list response"""
    model_types = fields.List(fields.Str(), description="List of model types")

class ModelFilesSchema(Schema):
    """Response for model file list in a specific folder"""
    files = fields.List(fields.Str(), description="File list")

# Upload-related Schema
class UploadResponseSchema(Schema):
    """Upload response model"""
    name = fields.Str(description="File name")
    subfolder = fields.Str(description="Subfolder path")
    type = fields.Str(description="Upload type")

class UploadImageRequestSchema(Schema):
    """Upload image request model"""
    image = fields.Raw(description="Image file", required=True)
    overwrite = fields.Bool(description="Whether to overwrite", required=False, default=False)
    type = fields.Str(description="Target upload type", required=False, default="input")
    subfolder = fields.Str(description="Subfolder path", required=False, default="")

class UploadMaskRequestSchema(Schema):
    """Upload mask request model"""
    image = fields.Raw(description="Mask image file", required=True)
    original_ref = fields.Dict(description="Original image reference info", required=True)

# View image-related Schema
class ViewImageQuerySchema(Schema):
    """Query parameters for viewing images"""
    filename = fields.Str(description="File name", required=True)
    type = fields.Str(description="File type", required=False, default="output")
    subfolder = fields.Str(description="Subfolder", required=False)
    preview = fields.Str(description="Preview parameters", required=False)
    channel = fields.Str(description="Channel parameters", required=False, default="rgba")

# History-related Schema
class HistoryResponseSchema(Schema):
    """History response model"""
    history = fields.List(fields.Dict(), description="History list")

class HistoryItemResponseSchema(Schema):
    """Single history item response model"""
    prompt = fields.Dict(description="Prompt info")
    outputs = fields.Dict(description="Output info")
    created_at = fields.Str(description="Creation time")

class HistoryRequestSchema(Schema):
    """History operation request model"""
    clear = fields.Bool(description="Whether to clear history", required=False)
    delete = fields.List(fields.Str(), description="List of history item IDs to delete", required=False)

# Prompt/generation-related Schema
class PromptRequestSchema(Schema):
    """Prompt request model"""
    prompt = fields.Dict(description="Workflow graph", required=True)
    number = fields.Float(description="Task number", required=False)
    front = fields.Bool(description="Insert at the front of the queue", required=False)
    extra_data = fields.Dict(description="Extra data", required=False)
    client_id = fields.Str(description="Client ID", required=False)

# Node info-related Schema
class NodeInfoResponseSchema(Schema):
    """Node info response model"""
    input = fields.Dict(description="Input types")
    output = fields.List(fields.Str(), description="Output types")
    output_name = fields.List(fields.Str(), description="Output names")
    name = fields.Str(description="Node class name")
    display_name = fields.Str(description="Display name")
    description = fields.Str(description="Node description")
    category = fields.Str(description="Node category")

# Internal API Schema
class LogsResponseSchema(Schema):
    """Logs response model"""
    logs = fields.Str(description="Log content")

class RawLogsResponseSchema(Schema):
    """Raw logs response model"""
    entries = fields.List(fields.Dict(), description="Log entries")
    size = fields.Dict(description="Terminal size")

class SubscribeLogsRequestSchema(Schema):
    """Subscribe logs request model"""
    clientId = fields.Str(required=True, description="Client ID")
    enabled = fields.Bool(required=True, description="Enable subscription")

class FolderPathsResponseSchema(Schema):
    """Folder paths response model"""
    paths = fields.Dict(keys=fields.Str(), values=fields.Str(), description="Folder path mapping")

class FilesResponseSchema(Schema):
    """File list response model"""
    files = fields.List(fields.Str(), description="File list")

class FreeRequestSchema(Schema):
    """Free memory/model request model"""
    unload_models = fields.Bool(description="Whether to unload models", required=False)
    free_memory = fields.Bool(description="Whether to free memory", required=False) 
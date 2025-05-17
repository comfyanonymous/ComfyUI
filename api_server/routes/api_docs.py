from aiohttp_apispec import docs, response_schema, request_schema, querystring_schema
from aiohttp import web
import asyncio
from api_server.utils.schemas import *

def wrap_stable_routes(app):
    """Add Swagger documentation annotations for stable APIs (server.py)"""
    from server import PromptServer
    server_instance = PromptServer.instance
    
    # ===== GET Methods =====
    
    # GET /prompt - Get queue status
    @docs(
        tags=["Stable"],
        summary="Get current queue status",
        description="Return the execution status information of the current queue."
    )
    @response_schema(QueueStatusSchema(), 200)
    async def get_prompt_swagger(request):
        return await server_instance.routes._routes_by_method['GET']['/prompt'](request) 
    app.router.add_get("/prompt", get_prompt_swagger)
    
    # GET /queue - Get queue details
    @docs(
        tags=["Stable"],
        summary="Get queue details",
        description="Return detailed information of the current queue."
    )
    @response_schema(QueueStatusSchema(), 200) 
    async def get_queue_swagger(request):
        return await server_instance.routes._routes_by_method['GET']['/queue'](request)
    app.router.add_get("/queue", get_queue_swagger)
    
    # GET /history - Get history
    @docs(
        tags=["Stable"],
        summary="Get history",
        description="Return the history of completed generation tasks."
    )
    @response_schema(HistoryResponseSchema(), 200)
    async def get_history_swagger(request):
        return await server_instance.routes._routes_by_method['GET']['/history'](request)
    app.router.add_get("/history", get_history_swagger)
    
    # GET /history/{prompt_id} - Get specific history
    @docs(
        tags=["Stable"],
        summary="Get specific history",
        description="Get detailed information of a specific history by prompt_id."
    )
    @response_schema(HistoryItemResponseSchema(), 200)
    async def get_history_id_swagger(request):
        return await server_instance.routes._routes_by_method['GET']['/history/{prompt_id}'](request)
    app.router.add_get("/history/{prompt_id}", get_history_id_swagger)
    
    # GET /system_stats - Get system status
    @docs(
        tags=["Stable"],
        summary="Get system status",
        description="Return system and resource usage information."
    )
    @response_schema(SystemStatsSchema(), 200)
    async def get_system_stats_swagger(request):
        return await server_instance.routes._routes_by_method['GET']['/system_stats'](request)
    app.router.add_get("/system_stats", get_system_stats_swagger)

    # GET /models - Get model type list
    @docs(
        tags=["Stable"],
        summary="Get model type list",
        description="Return the list of supported model types."
    )
    @response_schema(ModelsListSchema(), 200)
    async def list_models_swagger(request):
        return await server_instance.routes._routes_by_method['GET']['/models'](request)
    app.router.add_get("/models", list_models_swagger)
    
    # GET /models/{folder} - Get model files in a specific type
    @docs(
        tags=["Stable"],
        summary="Get model file list of a specific type",
        description="Return the list of model files for the specified type."
    )
    @response_schema(ModelFilesSchema(), 200)
    async def get_models_folder_swagger(request):
        return await server_instance.routes._routes_by_method['GET']['/models/{folder}'](request)
    app.router.add_get("/models/{folder}", get_models_folder_swagger)
    
    # GET /view - View image
    @docs(
        tags=["Stable"], 
        summary="View image",
        description="Get and optionally process an image file."
    )
    @querystring_schema(ViewImageQuerySchema())
    async def view_image_swagger(request):
        return await server_instance.routes._routes_by_method['GET']['/view'](request)
    app.router.add_get("/view", view_image_swagger)
    
    # GET /object_info - Get all node info
    @docs(
        tags=["Stable"],
        summary="Get all node info",
        description="Return detailed information of all available nodes."
    )
    @response_schema(Schema.from_dict({"nodes": fields.Dict(description="Node info dictionary")}), 200)
    async def get_object_info_swagger(request):
        return await server_instance.routes._routes_by_method['GET']['/object_info'](request)
    app.router.add_get("/object_info", get_object_info_swagger)
    
    # GET /object_info/{node_class} - Get specific node info
    @docs(
        tags=["Stable"],
        summary="Get specific node info",
        description="Return detailed information of the specified node type."
    )
    @response_schema(NodeInfoResponseSchema(), 200)
    async def get_object_info_node_swagger(request):
        return await server_instance.routes._routes_by_method['GET']['/object_info/{node_class}'](request)
    app.router.add_get("/object_info/{node_class}", get_object_info_node_swagger)
    
    # ===== POST Methods =====
    
    # POST /prompt - Submit generation task
    @docs(
        tags=["Stable"],
        summary="Submit generation task",
        description="Submit a new generation task to the queue."
    )
    @request_schema(PromptRequestSchema())
    @response_schema(PromptResponseSchema(), 200)
    async def post_prompt_swagger(request):
        return await server_instance.routes._routes_by_method['POST']['/prompt'](request)
    app.router.add_post("/prompt", post_prompt_swagger)
    
    # POST /queue - Operate queue
    @docs(
        tags=["Stable"],
        summary="Operate queue",
        description="Clear the queue or delete specific queue items."
    )
    @request_schema(QueueRequestSchema())
    async def post_queue_swagger(request):
        return await server_instance.routes._routes_by_method['POST']['/queue'](request)
    app.router.add_post("/queue", post_queue_swagger)
    
    # POST /history - Operate history
    @docs(
        tags=["Stable"],
        summary="Operate history",
        description="Clear the history or delete specific history items."
    )
    @request_schema(HistoryRequestSchema())
    async def post_history_swagger(request):
        return await server_instance.routes._routes_by_method['POST']['/history'](request)
    app.router.add_post("/history", post_history_swagger)
    
    # POST /interrupt - Interrupt current task
    @docs(
        tags=["Stable"],
        summary="Interrupt current task",
        description="Interrupt the currently running generation task."
    )
    async def post_interrupt_swagger(request):
        return await server_instance.routes._routes_by_method['POST']['/interrupt'](request)
    app.router.add_post("/interrupt", post_interrupt_swagger)
    
    # POST /free - Release resources
    @docs(
        tags=["Stable"],
        summary="Release resources",
        description="Unload models and/or free memory."
    )
    @request_schema(FreeRequestSchema())
    async def post_free_swagger(request):
        return await server_instance.routes._routes_by_method['POST']['/free'](request)
    app.router.add_post("/free", post_free_swagger)
    
    # POST /upload/image - Upload image
    @docs(
        tags=["Stable"],
        summary="Upload image",
        description="Upload an image file to the server."
    )
    @response_schema(UploadResponseSchema(), 200)
    async def upload_image_swagger(request):
        return await server_instance.routes._routes_by_method['POST']['/upload/image'](request)
    app.router.add_post("/upload/image", upload_image_swagger)
    
    # POST /upload/mask - Upload mask
    @docs(
        tags=["Stable"],
        summary="Upload mask",
        description="Upload a mask image and apply it to the original image."
    )
    @response_schema(UploadResponseSchema(), 200)
    async def upload_mask_swagger(request):
        return await server_instance.routes._routes_by_method['POST']['/upload/mask'](request)
    app.router.add_post("/upload/mask", upload_mask_swagger)


def wrap_internal_routes(app):
    """Add Swagger documentation annotations for internal APIs"""
    
    # GET /internal/logs - Get logs
    @docs(
        tags=["internal"],
        summary="Get logs",
        description="Get system log content."
    )
    @response_schema(LogsResponseSchema(), 200)
    async def get_logs_swagger(request):
        request_path = web.Request.clone(request)
        request_path._match_info = {'tail': 'logs'}
        app = request.app.middlewares[0](lambda r: r)  # Get parent app
        return await app._subapps['/internal']._handle(request_path)
    app.router.add_get("/internal/logs", get_logs_swagger)
    
    # GET /internal/logs/raw - Get raw logs
    @docs(
        tags=["internal"],
        summary="Get raw logs",
        description="Get raw system logs and terminal size."
    )
    @response_schema(RawLogsResponseSchema(), 200)
    async def get_raw_logs_swagger(request):
        request_path = web.Request.clone(request)
        request_path._match_info = {'tail': 'logs/raw'}
        app = request.app.middlewares[0](lambda r: r)  # Get parent app
        return await app._subapps['/internal']._handle(request_path)
    app.router.add_get("/internal/logs/raw", get_raw_logs_swagger)
    
    # PATCH /internal/logs/subscribe - Subscribe logs
    @docs(
        tags=["internal"],
        summary="Subscribe logs",
        description="Enable or disable log subscription for the client."
    )
    @request_schema(SubscribeLogsRequestSchema())
    async def subscribe_logs_swagger(request):
        request_path = web.Request.clone(request)
        request_path._match_info = {'tail': 'logs/subscribe'}
        app = request.app.middlewares[0](lambda r: r)  # Get parent app
        return await app._subapps['/internal']._handle(request_path)
    app.router.add_patch("/internal/logs/subscribe", subscribe_logs_swagger)
    
    # GET /internal/folder_paths - Get folder paths
    @docs(
        tags=["internal"],
        summary="Get folder paths",
        description="Get the paths of various types of folders in the system."
    )
    @response_schema(FolderPathsResponseSchema(), 200)
    async def get_folder_paths_swagger(request):
        request_path = web.Request.clone(request)
        request_path._match_info = {'tail': 'folder_paths'}
        app = request.app.middlewares[0](lambda r: r)  # Get parent app
        return await app._subapps['/internal']._handle(request_path)
    app.router.add_get("/internal/folder_paths", get_folder_paths_swagger)
    
    # GET /internal/files/{directory_type} - Get file list
    @docs(
        tags=["internal"],
        summary="Get file list",
        description="Get the list of files in the specified type of directory."
    )
    @response_schema(FilesResponseSchema(), 200)
    async def get_files_swagger(request):
        directory_type = request.match_info['directory_type']
        request_path = web.Request.clone(request)
        request_path._match_info = {'tail': f'files/{directory_type}', 'directory_type': directory_type}
        app = request.app.middlewares[0](lambda r: r)  # Get parent app
        return await app._subapps['/internal']._handle(request_path)
    app.router.add_get("/internal/files/{directory_type}", get_files_swagger)


def register_api_docs(app):
    """Register API documentation routes on the main app"""
    wrap_stable_routes(app)  # Stable APIs from server.py
    wrap_internal_routes(app)  # Internal APIs from api_server/ 
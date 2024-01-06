


import io
import traceback

from aiohttp import web
    
import folder_paths
from framework.model import object_storage
from aiyo_api_server.aiyo_api_server import AIYoApiServer
from framework.app_log import AppLog


from framework.err_code import ErrorCode
from framework.image_util import ImageUtil
from aiyo_api_server.server_helper import ServerHelper


@AIYoApiServer.instance.routes.post("/open_api/service/upload_file")
async def upload_file(request):
    """
    Query{
        file, FileField
    }
    """
    AppLog.info("[API] upload_file, recieve.")
    try:
        post_data = await request.post()
    except Exception as e:
        AppLog.info(f"[API] upload file, faild to get post data from request.")
        return web.json_response({ "code": ErrorCode.INVALID_PARAM, "data":{"filename": ""}, 
                                  "message": "Faild to get post data from request"})
    
    file_field = post_data.get("file", None)
    if file_field is None:
        return web.json_response({
            "code": ErrorCode.MISSING_PARAM,
            "data":{"filename": ""},
            "message": "Missing param: file."
        })
        
    try:
        # upload
        remote_name = folder_paths.input_path_local_to_remote(file_field.filename, rename=True)
        file_data = file_field.file.read()
        object_storage.MinIOConnection().put_object(remote_name, io.BytesIO(file_data), len(file_data))
        AppLog.info(f"[API] upload resource, result: {remote_name}")    
    except Exception as e:
        AppLog.info(f"[API] upload file, unexpected error. \n{traceback.format_exc()}")
        return web.json_response({ "code": ErrorCode.UNEXPECTED, "data":{"filename": ""}, 
                                  "message": "Unexpected error."})
    if True:
        return web.json_response({
            "code": ErrorCode.SUCCESS,
            "data":{"filename": remote_name},
            "message": ""
        }, status=200)
    else:
        AppLog.info(f"[API] upload file, upload failed. ")
        return web.json_response({ "code": ErrorCode.UNEXPECTED, "data":{"filename": ""}, 
                                  "message": "Upload failed."})


@AIYoApiServer.instance.routes.post("/open_api/service/{flow_id}/run")
async def run_flow(request):
    """
    INPUTS:
    type: str,
    body: dict, flow params
    """
    flow_id = request.match_info.get("flow_id", None)
    AppLog.info("[API] run_flow, recieve. flow id: {flow_id}")
    
    code = ErrorCode.SUCCESS
    err = ""

    # get json data
    try:    
        json_data =  await request.json()
        AppLog.info(f"[API] run_flow, json data: {AppLog.visible_convert(json_data)} ")
    except Exception as e:
        code = ErrorCode.INVALID_PARAM
        err = "Faild to get json data from request"
        AppLog.warning(f"[API] run flow. {err}\n{traceback.format_exc()}")
        
      
    task_id = None
    if code == ErrorCode.SUCCESS:
        params = json_data.get("body", None)
        task_id, (code, err) = ServerHelper.add_user_flow_task(flow_id, params)
    
    return web.json_response({
        "code": code,
        "data": {"task_id": task_id},
        "message": err
    })
    


@AIYoApiServer.instance.routes.post("/open_api/service/{flow_id}/register_webhook")
async def register_webhook(request):
    """
    INPUTS:
    on_start: str
    on_end: str
    on_processing: str
    """
    flow_id = request.match_info.get("flow_id", None)
    AppLog.info("[API] register_webhook, recieve. flow id: {flow_id}")
    
    code = ErrorCode.SUCCESS
    err = ""
    
    # get json data
    try:    
        json_data =  await request.json()
        AppLog.info(f"[API] register_webhook, json data: {json_data} ")
    except Exception as e:
        code = ErrorCode.INVALID_PARAM
        err = "Faild to get json data from request"
        AppLog.info(f"[API] register_webhook, ERROR. {err}")
    
    if code == ErrorCode.SUCCESS:
        on_start = json_data.get("on_start", None)
        on_end = json_data.get("on_end", None)
        on_processing = json_data.get("on_processing", None)
        
        code, err = ServerHelper.register_webhook(flow_id=flow_id, on_start=on_start, on_processing=on_processing, on_end=on_end)
    
    return web.json_response({
        "code": code,
        "data": None,
        "message": err
    }) 
    
    
    
@AIYoApiServer.instance.routes.post("/test/on_start")
async def test_on_start(request):
    json_data =  await request.json()
    AppLog.info("On start")
    AppLog.info(json_data)
    AppLog.info("\n")
    return web.Response(status=200)
    
    
    
@AIYoApiServer.instance.routes.post("/test/on_end")
async def test_on_end(request):
    json_data =  await request.json()
    AppLog.info("On end")
    AppLog.info(AppLog.visible_convert(json_data))
    img_str = json_data['result']['out_image']
    img = ImageUtil.base64_to_image(img_str)
    img.save("output/b.png")
    AppLog.info("\n")
    return web.Response(status=200)
    
    
    
@AIYoApiServer.instance.routes.post("/test/on_processing")
async def test_on_processing(request):
    json_data =  await request.json()
    AppLog.info("On processing")
    AppLog.info(json_data)
    AppLog.info("\n")
    return web.Response(status=200)
    
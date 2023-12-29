

import uuid
import datetime
import os
import traceback
import requests

import aiohttp
from aiohttp import web
    
from aiyo_server.aiyo_server import AIYoServer
from framework.app_log import AppLog

from framework.model import tb_data
from framework.workflow_utils import WorkflowUtils
from framework.err_code import ErrorCode
from config.config import CONFIG
from framework.image_util import ImageUtil


@AIYoServer.instance.routes.post("/open_api/service/upload_file")
async def upload_file(request):
    """
    Queray{
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
        succ, filename = await WorkflowUtils.upload_resource(file_field)
        AppLog.info(f"[API] upload resource, result: {succ}, {filename}")    
    except Exception as e:
        AppLog.info(f"[API] upload file, unexpected error. \n{traceback.format_exc()}")
        return web.json_response({ "code": ErrorCode.UNEXPECTED, "data":{"filename": ""}, 
                                  "message": "Unexpected error."})
    if succ:
        return web.json_response({
            "code": ErrorCode.SUCCESS,
            "data":{"filename": filename},
            "message": ""
        }, status=200)
    else:
        AppLog.info(f"[API] upload file, upload failed. ")
        return web.json_response({ "code": ErrorCode.UNEXPECTED, "data":{"filename": ""}, 
                                  "message": "Upload failed."})


@AIYoServer.instance.routes.post("/open_api/service/{flow_id}/run")
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
        AppLog.info(f"[API] run_flow, json data: {json_data} ")
    except Exception as e:
        code = ErrorCode.INVALID_PARAM
        err = "Faild to get json data from request"
        AppLog.info(f"[API] run flow. {err}")
        
    if code == ErrorCode.SUCCESS:
        params = json_data.get("body", None)
        
        try:      
            # get flow data
            flow_infos = tb_data.Flow.objects(flowId=flow_id)
            flow_info = flow_infos[0] if flow_infos is not None and len(flow_infos)>0 else None
            webhooks = flow_info.webhook
            flow_input = flow_info.flowInput
            
            # upload image resource
            # ??
            for arg_name, arg_val in params.items():
                if arg_name in flow_input and flow_input[arg_name] == "IMAGE":
                    # http or https url
                    if arg_val is not None and (arg_val.startswith("http://") or arg_val.startswith("https://")):
                        
                        filename = WorkflowUtils.extract_filename_from_url(arg_val)
                        fileext = os.path.splitext(filename)[1]
                        local_dir = CONFIG["resource"]["in_img_path_local"]
                        basename = f"{str(uuid.uuid4())}{fileext}"
                        local_name = f"{local_dir}/{basename}"
                        # download image
                        response = requests.get(arg_val)
                        with open(local_name, 'wb') as file:
                            file.write(response.content)
                        # upload
                        remote_name = WorkflowUtils.upload_file(local_name)
                        # update parameter value
                        params[arg_name] = remote_name
                        
                    
                    # base64str image
                    else:
                        local_dir = CONFIG["resource"]["in_img_path_local"]
                        basename = f"{str(uuid.uuid4())}.png"
                        local_name = f"{local_dir}/{basename}"
                        cur_img = ImageUtil.base64_to_image(arg_val)
                        cur_img.save(local_name)
                        # upload
                        remote_name = WorkflowUtils.upload_file(local_name)
                        # update parameter value
                        params[arg_name] = remote_name
                        
                        
            AppLog.info(f"[API Run] params after parse: {params}")
            
            # generate task 
            task_id = str(uuid.uuid4())
            now = datetime.datetime.utcnow()
            task = tb_data.Task(taskId=task_id, flowId=flow_id, 
                                status=0, taskParams=params,
                                taskType="api",
                                createdBy="aiyoh", createdAt=now,
                                lastUpdatedAt=now,
                                webhook=webhooks)
            task.save()
    
            # add task into task queue
            AIYoServer.instance.prompt_queue.put(task_id)
            
            return web.json_response({
                "code": ErrorCode.SUCCESS,      # for success
                "data": {"task_id": task_id},
                "message": ""
            })
            
        except Exception as e:
            AppLog.warning(f"[API Run] unexpected error: {traceback.format_exc()}")
            code = ErrorCode.UNEXPECTED
            err = "Unexpected error."
    
    return web.json_response({
        "code": code,
        "data": {"task_id": ""},
        "message": err
    })
    


@AIYoServer.instance.routes.post("/open_api/service/{flow_id}/register_webhook")
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
        
        flow_info = tb_data.Flow.objects(flowId=flow_id).first()
        if flow_info is not None:
            flow_info.webhook = {"on_start": on_start, "on_end": on_end, "on_processing": on_processing}
            flow_info.save()
        else:
            code = ErrorCode.INVALID_PARAM
            err = f"Can not find flow: {flow_id}"  
            AppLog.info(f"[API] register_webhook, ERROR. {err}")
    
    return web.json_response({
        "code": code,
        "data": None,
        "message": err
    }) 
    
    
    
@AIYoServer.instance.routes.post("/test/on_start")
async def test_on_start(request):
    json_data =  await request.json()
    print("On start")
    print(json_data)
    print("\n")
    return web.Response(status=200)
    
    
    
@AIYoServer.instance.routes.post("/test/on_end")
async def test_on_end(request):
    json_data =  await request.json()
    print("On end")
    print(json_data)
    img_str = json_data['result']['out_image']
    img = ImageUtil.base64_to_image(img_str)
    img.save("output/b.png")
    print("\n")
    return web.Response(status=200)
    
    
    
@AIYoServer.instance.routes.post("/test/on_processing")
async def test_on_processing(request):
    json_data =  await request.json()
    print("On processing")
    print(json_data)
    print("\n")
    return web.Response(status=200)
    
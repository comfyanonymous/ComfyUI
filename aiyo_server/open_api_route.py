

import uuid
import datetime
import os
import traceback

import aiohttp
from aiohttp import web
    
from aiyo_server.aiyo_server import AIYoServer
from framework.app_log import AppLog

from framework.model import tb_data
from framework.workflow_utils import WorkflowUtils
from framework.err_code import ErrorCode


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
    except Exception as e:
        code = ErrorCode.INVALID_PARAM
        err = "Faild to get json data from request"
        AppLog.info(f"[API] run flow. {err}")
        
    if code == ErrorCode.SUCCESS:
        params = json_data.get("body", None)
         
        try:       
            # generate task 
            task_id = str(uuid.uuid4())
            now = datetime.datetime.utcnow()
            task = tb_data.Task(taskId=task_id, flowId=flow_id, 
                                status=0, taskParams=params,
                                taskType="api",
                                createdBy="aiyoh", createdAt=now,
                                lastUpdatedAt=now)
            task.save()
    
            # add task into task queue
            AIYoServer.instance.prompt_queue.put(task_id)
            
            return web.json_response({
                "code": ErrorCode.SUCCESS,      # for success
                "data": {"task_id": task_id},
                "message": ""
            })
            
        except Exception as e:
            code = ErrorCode.UNEXPECTED
            err = "Unexpected error."
    
    return web.json_response({
        "code": code,
        "data": {"task_id": ""},
        "message": err
    })
    
    
    
    
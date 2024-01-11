



import uuid
import datetime
import os
import traceback
import requests
import urllib
import base64

    
import folder_paths
from framework.model import object_storage
from aiyo_api_server.aiyo_api_server import AIYoApiServer
from framework.app_log import AppLog

from framework.model import tb_data
from framework.err_code import ErrorCode
from framework.image_util import ImageUtil

class ServerHelper:
    
    
    @staticmethod
    def extract_filename_from_url(url):
        parsed_url = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed_url.path)
        return filename
    
    
    @staticmethod
    def parse_user_flow_input(params, flow_input):
        """
        Parse user inputs. 
        upload image resources and update image path value in the inputs
        """
        try:      
            # upload image resource
            for arg_name, arg_val in params.items():
                if arg_name in flow_input and flow_input[arg_name] == "IMAGE":
                    # http or https url
                    if arg_val is not None and (arg_val.startswith("http://") or arg_val.startswith("https://")):
                        # local file name
                        filename = ServerHelper.extract_filename_from_url(arg_val)
                        local_name = folder_paths.input_path_remote_to_local(filename, rename=True)
                        # download image
                        response = requests.get(arg_val)
                        with open(local_name, 'wb') as file:
                            file.write(response.content)
                        # upload
                        remote_name = folder_paths.input_path_local_to_remote(local_name, rename=False)
                        object_storage.MinIOConnection().fput_object(remote_name, local_name)
                        # update parameter value
                        params[arg_name] = remote_name
                        
                    
                    # base64str image
                    else:
                        local_name = folder_paths.generate_local_filepath('png')
                        cur_img = ImageUtil.base64_to_image(arg_val)
                        cur_img.save(local_name)
                        # upload
                        remote_name = folder_paths.input_path_local_to_remote(local_name, rename=False)
                        object_storage.MinIOConnection().fput_object(remote_name, local_name)
                        # update parameter value
                        params[arg_name] = remote_name

            return True, params, (ErrorCode.SUCCESS, '')
                        
        except Exception as e:
            AppLog.warning(f"[ParseUserFlowInput] unexpected error: {traceback.format_exc()}")
            code = ErrorCode.UNEXPECTED
            err = "Unexpected error."
            
            return False, None, (code, err)
        
        
        
    @staticmethod
    def add_user_flow_task(flow_id, params):
        code = ErrorCode.SUCCESS
        err = ''
        try:      
            # get flow data
            flow_info = tb_data.Flow.objects(flowId=flow_id).first()
            if flow_info is not None:
            
                webhooks = flow_info.webhook
                flow_input = flow_info.flowInput
                
                # parse input data 
                # and upload image resource
                succ, params, status = ServerHelper.parse_user_flow_input(params=params, flow_input=flow_input)
                code = status[0]
                err = status[1]
                if not succ: 
                    AppLog.info(f"[AddUserTask] parse user input FAIL. code:{code}, err:{err}")
                    return None, (code, err)
                AppLog.info(f"[AddUserTask] params after parse: {params}")
                
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
                AIYoApiServer.instance.prompt_queue.put(task_id)
                return task_id, (code, err)
            
            else:
                code = ErrorCode.INVALID_PARAM
                err = "flow_id not exist."
                return None, (code, err)    
            
        except Exception as e:
            AppLog.warning(f"[AddUserTask] unexpected error: {traceback.format_exc()}")
            code = ErrorCode.UNEXPECTED
            err = "Unexpected error."
            return None, (code, err)
        
        
        
    @staticmethod
    def register_webhook(flow_id, on_start, on_processing, on_end):
        flow_info = tb_data.Flow.objects(flowId=flow_id).first()
        if flow_info is not None:
            flow_info.webhook = {"on_start": on_start, "on_end": on_end, "on_processing": on_processing}
            flow_info.save()
            code = ErrorCode.SUCCESS
            err = ""
        else:
            code = ErrorCode.INVALID_PARAM
            err = f"Can not find flow: {flow_id}"  
            AppLog.info(f"[RegisterWebhook] register_webhook, ERROR. {err}")
        return code, err
    
    
    @staticmethod
    def _parse_result_for_user(result_data, flow_output_info):
        # parse task result for user request
        for res_name, res_val in result_data.items():
            # for IMAGE data, download and convert to BASE64 data
            if flow_output_info is not None and res_name in flow_output_info and flow_output_info[res_name] == "IMAGE":
                # get image data
                data = object_storage.MinIOConnection().get_object(res_val)
                # convert to base64
                base64_data = base64.b64encode(data).decode('utf-8')
                result_data[res_name] = base64_data
        return result_data
    
    
    @staticmethod
    def get_task_progress(task_id):
        code = 1
        status = -1         # not exist
        progress = 0.0        # float, from 0.0 -1.0
        result = None
        fail_msg = ""
        err_msg = ""
        try:
            task_info = tb_data.Task.objects(taskId=task_id).first()
            if task_info is not None:
                status = task_info.status
                if status == 3:                 # task finished
                    # find task results
                    task_res = tb_data.TaskReuslt.objects(taskId=task_id).first()
                    if task_res is not None:
                        # get flow data
                        flow_info = tb_data.Flow.objects(flowId=task_info.flowId).first()
                        flow_out_info = flow_info.flowOutput
                        # parse result data
                        result_data = ServerHelper._parse_result_for_user(task_res.result, flow_out_info)
                        progress = 1.0
                        result = result_data
                        fail_msg = ""
                        status = 3
                    else:
                        result_data = None
                        progress = 1.0
                        fail_msg = "Task result not found. FAIL due to unexpected error."
                        status = 4
                        
                elif status == 4:               # task failed
                    # find task results
                    task_res = tb_data.TaskReuslt.objects(taskId=task_id).first()
                    if task_res is not None:
                        progress = 1.0
                        result = None
                        fail_msg = task_res.error
                    else:
                        progress = 1.0
                        result = None
                        fail_msg = "Task result not found. FAIL due to unexpected error." 
                else:
                    progress = 0.5 if status ==2 else 0.0
        
        except Exception as e:
            code = ErrorCode.EXE_UNEXP
            err_msg = f"Unexpected error"
            
            AppLog.warning(f"[Helper.get_task_progress] {err_msg}\n {traceback.format_exc()}")          
            
        AppLog.info(f"[Helper.get_task_progress] code: {status}, status: {status}, progress: {progress}, \nresult: {AppLog.visible_convert(result)}, \nfail_msg: {fail_msg}, \nerr: {err_msg}")
        return code, {
            "status": status,
            "progress": progress,
            "result": result,
            "fail_msg": fail_msg
        }, err_msg
                    
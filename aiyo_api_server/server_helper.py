



import uuid
import datetime
import os
import traceback
import requests
import urllib

    
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
            flow_infos = tb_data.Flow.objects(flowId=flow_id)
            flow_info = flow_infos[0] if flow_infos is not None and len(flow_infos)>0 else None
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
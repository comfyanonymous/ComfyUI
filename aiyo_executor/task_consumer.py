
import os
import time
import json
import traceback

from framework.app_log import AppLog
from framework.kafka_connection import KafkaConnection
from aiyo_executor.message_sender import APICall
from framework.model import tb_data
from framework.model import object_storage
from config.config import CONFIG


class TaskConsumerLocal:
    
    def get(self, timeout=None):
        """
        RETURN:
        dict, {prompt_id: str, prompt: dict, flows: dict, extra_data: dict}
        """
        topic = CONFIG["kafka_settings"]["topic"]
        url = CONFIG["server"]["url"]
        try:        
            succ, json_response = APICall.get_sync(f"{url}/task_exe/get_task", {"topic": topic})
        except Exception as e:
            succ = False
            AppLog.error(f"[Get Task] ERROR. {e}\n {traceback.format_exc()}")
        if not succ or json_response is None:
            return None
    
        prompt_id = json_response["prompt_id"]
        if prompt_id is not None:
            AppLog.info(f"[Get Task] new task: {json_response}")
        else:
            time.sleep(1)
        return json_response
    
    
    
class TaskConsumerDeploy:
    
    def __init__(self) -> None:
        self.consumer = KafkaConnection.create_consumer(CONFIG["kafka_settings"]["topic"])
    
    def get(self, timeout=None):
        
        if timeout is not None:
            msg = self.consumer.poll(timeout=timeout)
        else:
            msg = self.consumer.poll(timeout=1)
        
        if msg is not None:
            task_id = msg.value().decode('utf-8')
            AppLog.info(f"[ConsumeTask] new task: {task_id}")
            
            nd_prompt, flows, extra_data, flow_args, webhooks = self._get_task_item(task_id)
            return {
                "prompt_id": task_id,
                "prompt": nd_prompt,
                "flows": flows,
                "extra_data": extra_data, 
                "flow_args": flow_args,
                "webhooks": webhooks
            }
            
            
        return { "prompt_id": None, "prompt": None, "flows": None, "extra_data": None, "flow_args": None, "webhooks": None }
    
    
    def _get_task_item(self, task_id):
        
        # find task
        task_infos = tb_data.Task.objects(taskId=task_id)
        if task_infos is not None and len(task_infos) > 0:
            task_info = task_infos[0]
            flow_id = task_info.flowId
            flow_args = task_info.taskParams
            webhooks = task_info.webhook
            
            flow_infos = tb_data.Flow.objects(flowId=flow_id)
            flow_info = flow_infos[0] if flow_infos is not None and len(flow_infos)>0 else None
            prompt_filepath = flow_info.flowPrompt if flow_info is not None else None
            
            # download prompt file
            prompt_filepath = flow_info.flowPrompt if flow_info is not None else None
            if prompt_filepath is not None and prompt_filepath != "":
                prompt_filebase = os.path.basename(prompt_filepath)
                prompt_filepath_local = CONFIG['resource']['prompt_path_local']
                prompt_filepath_local = f"{prompt_filepath_local}/{prompt_filebase}"  
                
                # if not exist ????              
                object_storage.MinIOConnection().fget_object(obj_name=prompt_filepath, file_path=prompt_filepath_local)
                
                with open(prompt_filepath_local) as json_file:
                    data = json.load(json_file)
                    nd_prompt = data["prompt"]
                    flows = data["flows"]
                    extra_data = data["extra_data"]
                    
                    AppLog.info(f"[Promptfile] nd prompt: \n{nd_prompt}")
                    AppLog.info(f"[Promptfile] flows: \n{flows}")
                    AppLog.info(f"[Promptfile] extra data: \n{extra_data}")
                    
                    return nd_prompt, flows, extra_data, flow_args, webhooks
                
        return None, None, None, None, None

            
            
        
        
        
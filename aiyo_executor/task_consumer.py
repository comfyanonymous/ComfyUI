
import requests
import time
from config.config import CONFIG
from framework.app_log import AppLog

from framework.kafka_connection import KafkaConnection
from aiyo_executor.message_sender import MessageSender


class TaskConsumerLocal:
    
    def get(self, timeout=None):
        topic = CONFIG["kafka_settings"]["topic"]
        succ, json_response = MessageSender.get_sync("/task_exe/get_task", {"topic": topic})
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
        self.consumer = KafkaConnection.create_consumer(CONFIG["topic"])
    
    def get(self, timeout=None):
        
        item = self.consumer.poll(timeout=timeout)
        if item is None:
            return None
        # item id: ???????
        # item_id = item.id
        # self.currently_running[item_id] = copy.deepcopy(item)
        return item
        
        
        
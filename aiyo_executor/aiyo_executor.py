
import time
import gc
import requests

from aiohttp import web

from framework.flow_execution import FlowExecutor
import comfy.model_management

from aiyo_executor import task_consumer
from config.config import CONFIG
from aiyo_executor import message_sender
from framework.app_log import AppLog
from server import PromptServer


class AIYoExecutor:
    
    def __init__(self, msg_sender:message_sender.MessageSender) -> None:
        PromptServer.instance = self                # hack code for extension supports
        self.supports = []
        routes = web.RouteTableDef()
        self.routes = routes
        
        
        self.msg_sender = msg_sender
        
        self.task_que = None 
        if CONFIG["deploy"]:
            self.task_que = task_consumer.TaskConsumerDeploy()
        else:
            self.task_que = task_consumer.TaskConsumerLocal()
            
        self.executor = FlowExecutor(msg_sender)
        
        
    def task_done(self, prompt_id, output_data):    
        succ, response_data = message_sender.MessageSender.post_sync("/task_exe/task_done",
                                                                     {"prompt_id": prompt_id, "output": output_data})
        if not succ:
            AppLog.warning(f'[TaskDone] fail to inform TASK_DONE event to server.') 
            
    
    
    def run(self):
        last_gc_collect = 0
        need_gc = False
        gc_collect_interval = 10.0

        while True:
            timeout = None
            if need_gc:
                timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

            queue_item = self.task_que.get(timeout=timeout)
            prompt_id = queue_item.get("prompt_id", None) if queue_item is not None else None
            
            if prompt_id is not None and prompt_id != "":
                AppLog.info(f"[Prompt worker] new task: {queue_item}")
                
                execution_start_time = time.perf_counter()
                
                self.executor.execute(queue_item["prompt"], prompt_id, queue_item["flows"], queue_item["extra_data"], queue_item["outputs_to_execute"])
                need_gc = True
                self.task_done(prompt_id, self.executor.outputs_ui)
                if self.msg_sender.client_id is not None:
                    self.msg_sender.send_sync("executing", { "node": None, "prompt_id": prompt_id }, self.msg_sender.client_id)

                current_time = time.perf_counter()
                execution_time = current_time - execution_start_time
                AppLog.info("[Prompt worker] Prompt executed in {:.2f} seconds".format(execution_time))

            if need_gc:
                current_time = time.perf_counter()
                if (current_time - last_gc_collect) > gc_collect_interval:
                    gc.collect()
                    comfy.model_management.soft_empty_cache()
                    last_gc_collect = current_time
                    need_gc = False
        
        
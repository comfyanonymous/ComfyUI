
import time
import gc
import datetime

from aiohttp import web

from framework.flow_execution import FlowExecutor
import comfy.model_management

from aiyo_executor import task_consumer
from config.config import CONFIG
from aiyo_executor import message_sender
from framework.app_log import AppLog
from server import PromptServer
from framework.model import tb_data
from framework.model import object_storage


class AIYoExecutor:
    
    def __init__(self, msg_sender:message_sender.MessageManager) -> None:
        PromptServer.instance = self                # hack code for extension supports
        self.supports = []
        routes = web.RouteTableDef()
        self.routes = routes
        
        
        self.msg_sender = msg_sender
        
        self.task_que = None 
        if CONFIG["deploy"]:
            self.task_que = task_consumer.TaskConsumerDeploy()
            tb_data.default_connect()
            self.resource_mgr = object_storage.ResourceMgrRemote()
        else:
            self.task_que = task_consumer.TaskConsumerLocal()
            self.resource_mgr = object_storage.ResourceMgrLocal()
            
        self.executor = FlowExecutor(msg_sender)
        
        
        
    def task_start(self, prompt_id, task_info):
        # set message sender
        if CONFIG["deploy"]:
            webhooks = task_info.get("webhooks", {})
            self.msg_sender.message_sender = message_sender.WeebhookSender(on_start=webhooks.get("on_start", None),
                                                                            on_processing=webhooks.get("on_processing", None),
                                                                            on_end=webhooks.get("on_end", None))
        else:
            self.msg_sender.message_sender = message_sender.LocalAPISender()
            
        # send START evnet
        self.msg_sender.send_sync("execution_start", { "prompt_id": prompt_id}, self.msg_sender.client_id)
        
        
    def task_done(self, prompt_id, graph_output, output_ui):    
            
        output_data = {key:val["value"] for key,val in graph_output.items()}
        # update task result to db
        if CONFIG["deploy"] and prompt_id is not None:
            query = {"taskId": prompt_id}
            update_data = {"taskId": prompt_id, 
                           "status":3,
                           "endTime": datetime.datetime.utcnow(),
                           "result": output_data,
                           "error": ""}
            task_res = tb_data.TaskReuslt.objects(**query).modify(upsert=True, new=True, **update_data)
            AppLog.info(f"update result: {task_res}")
            
        self.msg_sender.send_sync("execution_end", {"prompt_id": prompt_id, 
                                                    "result_info": graph_output,
                                                    "result": output_data,
                                                    "output_ui": output_ui})

        # remove message_sender
        # self.msg_sender.message_sender = None
        AppLog.info(f"[Execute] End: {prompt_id}")
        
    
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
                
                extra_data = queue_item["extra_data"]
                if extra_data is not None and "client_id" in extra_data:
                    self.msg_sender.client_id = extra_data["client_id"]
                else:
                    self.msg_sender.client_id = None
                    
                # on task start
                self.task_start(prompt_id, queue_item)
                
                
                
                # execute workflow
                graph_outputs = self.executor.execute(queue_item["prompt"], prompt_id, queue_item["flows"], queue_item.get("flow_args", {}),
                                      queue_item["extra_data"])#, queue_item["outputs_to_execute"])
                need_gc = True
                
                # on task done
                self.task_done(prompt_id, graph_outputs, self.executor.outputs_ui)
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
        
        
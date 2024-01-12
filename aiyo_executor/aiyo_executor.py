
import time
import gc
import datetime
import traceback, json

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
        
        
    def on_task_start(self, prompt_id, task_info):
        # update task status
        if CONFIG["deploy"] and prompt_id is not None:
            tb_data.Task.objects(taskId=prompt_id).modify(status=2)
        
        
        
    def task_start(self, prompt_id, task_info):
        # set message sender
        if CONFIG["deploy"]:
            webhooks = task_info.get("webhooks", {})
            if webhooks is None:
                webhooks = {}
            self.msg_sender.message_sender = message_sender.WeebhookSender(on_start=webhooks.get("on_start", None),
                                                                            on_processing=webhooks.get("on_processing", None),
                                                                            on_end=webhooks.get("on_end", None))
        else:
            self.msg_sender.message_sender = message_sender.LocalAPISender()
            
        # 
        self.on_task_start(prompt_id, task_info)
            
        # send START evnet
        self.msg_sender.send_sync("execution_start", { "prompt_id": prompt_id}, self.msg_sender.client_id)
        
        
        
    def on_task_done(self, succ, prompt_id, output_data, output_ui, err, exp):
        # update task result to db
        if CONFIG["deploy"] and prompt_id is not None:
            status = 3 if succ else 4
            _err = err
            if isinstance(err, dict):
                _err = json.dumps(err)
                
            try:
                now = datetime.datetime.utcnow()
                query = {"taskId": prompt_id}
                update_data = {"taskId": prompt_id, 
                            "status":status,
                            "endTime": now,
                            "result": output_data,
                            "error": _err}
                AppLog.info(f"update data:{update_data}")
                task_res = tb_data.TaskResult.objects(**query).modify(upsert=True, new=True, **update_data)
                AppLog.info(f"update result: {task_res}")
            except Exception as e:
                AppLog.error(f"[OnTaskDonw] ERROR, fail to update status to TaskResult DB. {e}\n{traceback.format_exc()}")
            
            try:
                tb_data.Task.objects(**query).modify(status=status)
            except Exception as e:
                AppLog.error(f"[OnTaskDonw] ERROR, fail to update status to Task DB. {e}\n{traceback.format_exc()}")
            
        
    def task_done(self, succ, prompt_id, graph_output, output_ui, err, exp):    
            
        if graph_output is not None:
            output_data = {key:val["value"] for key,val in graph_output.items()}
        else:
            output_data = {}
        # on task done
        self.on_task_done(succ, prompt_id, output_data, output_ui, err, exp)
            
        # msg
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
                try: 
                    succ, graph_outputs,err, exp = self.executor.execute(queue_item["prompt"], prompt_id, queue_item["flows"], queue_item.get("flow_args", {}),
                                        queue_item["extra_data"])#, queue_item["outputs_to_execute"])
                except Exception as e:
                    graph_outputs = {}
                    msg = f"Unexpected error. \n {traceback.format_exc()}"
                    mes = {
                        "prompt_id": prompt_id,
                        "node_id": 0,
                        "node_type": "",
                        "executed": [],
                        
                        "exception_message": "Unexpected error.",
                        "exception_type": "",
                        "traceback": traceback.format_exc(),
                        "current_inputs": None,
                        "current_outputs": None
                    }
                    self.msg_sender.send_sync("execution_error", mes, None)
                    succ = False
                    err = msg
                    exp = e
                    
                need_gc = True
                
                # on task done
                self.task_done(succ, prompt_id, graph_outputs, self.executor.outputs_ui, err, exp)
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
        
    def add_on_prompt_handler(self, onprompt):
        pass
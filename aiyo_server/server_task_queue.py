

import heapq
import threading
import copy
import uuid

from framework.kafka_connection import KafkaConnection

from config.config import CONFIG
from framework.task_engine_util import validate_prompt
from framework.app_log import AppLog


MAXIMUM_HISTORY_SIZE = 10000




class TaskQueueLocal:
    def __init__(self, server):
        self.server = server
        self.mutex = threading.RLock()
        self.not_empty = threading.Condition(self.mutex)
        self.task_counter = 0
        self.queue = []
        self.currently_running = {}
        self.history = {}
        self.number = 0
        
        
    def put(self, task_info):
        if "number" in task_info:
            number = float(task_info['number'])
        else:
            number = self.number
            if "front" in task_info:
                if task_info['front']:
                    number = -number

            self.number += 1

        if "prompt" in task_info:
            prompt = task_info["prompt"]
            valid = validate_prompt(prompt)
            extra_data = {}
            if "extra_data" in task_info:
                extra_data = task_info["extra_data"]

            if "client_id" in task_info:
                extra_data["client_id"] = task_info["client_id"]
            if valid[0]:
                prompt_id = str(uuid.uuid4())
                outputs_to_execute = valid[2]
                flows = task_info.get("flows", {})
                self._put((number, prompt_id, prompt, extra_data, outputs_to_execute, flows))
                response = {"prompt_id": prompt_id, "number": number, "node_errors": valid[3]}
                return True, response
            else:
                AppLog.info(f"invalid prompt: {valid[1]}")
                return False, {"error": valid[1], "node_errors": valid[3]}
        else:
            return False, {"error": "no prompt", "node_errors": []}
        
        

    def _put(self, item):
        with self.mutex:
            heapq.heappush(self.queue, item)
            self.server.server_client_communicator.queue_updated() if self.server.server_client_communicator is not None else None
            self.not_empty.notify()

    def get(self, timeout=None):
        with self.not_empty:
            if len(self.queue) == 0:
                return None
            # while len(self.queue) == 0:
            #     self.not_empty.wait(timeout=timeout)
            #     if timeout is not None and len(self.queue) == 0:
            #         return None
            item = heapq.heappop(self.queue)
            # i = self.task_counter
            prompt_id = item[1]
            self.currently_running[prompt_id] = copy.deepcopy(item)
            # self.task_counter += 1
            self.server.server_client_communicator.queue_updated() if self.server.server_client_communicator is not None else None
            return (item, prompt_id)

    def task_done(self, prompt_id, outputs):
        with self.mutex:
            if prompt_id is None or prompt_id not in self.currently_running:
                return
            prompt = self.currently_running.pop(prompt_id)
            if len(self.history) > MAXIMUM_HISTORY_SIZE:
                self.history.pop(next(iter(self.history)))
            self.history[prompt[1]] = { "prompt": prompt, "outputs": {} }
            for o in outputs:
                self.history[prompt[1]]["outputs"][o] = outputs[o]
            self.server.server_client_communicator.queue_updated() if self.server.server_client_communicator is not None else None


    def get_current_queue(self):
        with self.mutex:
            out = []
            for x in self.currently_running.values():
                out += [x]
            return (out, copy.deepcopy(self.queue))

    def get_tasks_remaining(self):
        with self.mutex:
            return len(self.queue) + len(self.currently_running)

    def wipe_queue(self):
        with self.mutex:
            self.queue = []
            self.server.server_client_communicator.queue_updated() if self.server.server_client_communicator is not None else None


    def delete_queue_item(self, function):
        with self.mutex:
            for x in range(len(self.queue)):
                if function(self.queue[x]):
                    if len(self.queue) == 1:
                        self.wipe_queue()
                    else:
                        self.queue.pop(x)
                        heapq.heapify(self.queue)
                        self.server.server_client_communicator.queue_updated() if self.server.server_client_communicator is not None else None

                    return True
        return False

    def get_history(self, prompt_id=None, max_items=None, offset=-1):
        with self.mutex:
            if prompt_id is None:
                out = {}
                i = 0
                if offset < 0 and max_items is not None:
                    offset = len(self.history) - max_items
                for k in self.history:
                    if i >= offset:
                        out[k] = self.history[k]
                        if max_items is not None and len(out) >= max_items:
                            break
                    i += 1
                return out
            elif prompt_id in self.history:
                return {prompt_id: copy.deepcopy(self.history[prompt_id])}
            else:
                return {}

    def wipe_history(self):
        with self.mutex:
            self.history = {}

    def delete_history_item(self, id_to_delete):
        with self.mutex:
            self.history.pop(id_to_delete, None)


    def get_queue_info(self):
        prompt_info = {}
        exec_info = {}
        exec_info['queue_remaining'] = self.get_tasks_remaining()
        prompt_info['exec_info'] = exec_info
        return prompt_info




    

class TaskQueueKafka:
    def __init__(self, server):
        self.server = server
        self.mutex = threading.RLock()
        self.not_empty = threading.Condition(self.mutex)
        self.task_counter = 0
        self.queue = []
        self.currently_running = {}
        self.history = {}
        
        self.producer = None
        self.consumer = None
        
    def init_producer(self):
        self.producer = KafkaConnection.create_producer()   
        
        
    def init_consumer(self):
        self.consumer = KafkaConnection.create_consumer(CONFIG["topic"])
    

    def put(self, task_id):
        """
        Put a new task
        """
        with self.mutex:
            AppLog.info("Here put task to kafka")
            AppLog.info("put task into kafka")
            
            # update upload resource to object
            # ????
            
            # write task info to db
            # ?????
            
            # add task into kafka queue
            msg = bytes(task_id, encoding='utf-8')
            self.producer.produce(CONFIG["kafka_settings"]["topic"], value=msg)
            AppLog.info(msg)
            self.server.server_client_communicator.queue_updated() if self.server.server_client_communicator is not None else None

            self.not_empty.notify()
            

    def get(self, timeout=None):
        """
        Get a task 
        """
        with self.mutex:
            # get task info from kafka consumer
            item = self.consumer.poll(timeout = timeout)
            if item is None:
                return None
            # item id: ???????
            # item_id = item.id
            # self.currently_running[item_id] = copy.deepcopy(item)
            self.server.server_client_communicator.queue_updated() if self.server.server_client_communicator is not None else None

            return (item, 0)


    def task_done(self, item_id, outputs):
        """
        
        """
        with self.mutex:
            # prompt = self.currently_running.pop(item_id)
            # if len(self.history) > MAXIMUM_HISTORY_SIZE:
            #     self.history.pop(next(iter(self.history)))
            # self.history[prompt[1]] = { "prompt": prompt, "outputs": {} }
            # for o in outputs:
            #     self.history[prompt[1]]["outputs"][o] = outputs[o]
            
            # to do  ????????
            
            self.server.server_client_communicator.queue_updated() if self.server.server_client_communicator is not None else None


    def get_current_queue(self):
        # with self.mutex:
        #     out = []
        #     for x in self.currently_running.values():
        #         out += [x]
        #     return (out, copy.deepcopy(self.queue))
        
        # todo ???
        return []
        

    def get_tasks_remaining(self):
        # with self.mutex:
        #     return len(self.queue) + len(self.currently_running)
        
        # todo ???????
        return 0

    def wipe_queue(self):
        # with self.mutex:
        #     self.queue = []
        #     self.server.server_client_communicator.queue_updated()
        
        # todo ????????
        pass

    def delete_queue_item(self, function):
        # with self.mutex:
        #     for x in range(len(self.queue)):
        #         if function(self.queue[x]):
        #             if len(self.queue) == 1:
        #                 self.wipe_queue()
        #             else:
        #                 self.queue.pop(x)
        #                 heapq.heapify(self.queue)
        #             self.server.server_client_communicator.queue_updated()
        #             return True
        # return False
        
        # todo ????????
        return False
    

    def get_history(self, prompt_id=None, max_items=None, offset=-1):
        # with self.mutex:
        #     if prompt_id is None:
        #         out = {}
        #         i = 0
        #         if offset < 0 and max_items is not None:
        #             offset = len(self.history) - max_items
        #         for k in self.history:
        #             if i >= offset:
        #                 out[k] = self.history[k]
        #                 if max_items is not None and len(out) >= max_items:
        #                     break
        #             i += 1
        #         return out
        #     elif prompt_id in self.history:
        #         return {prompt_id: copy.deepcopy(self.history[prompt_id])}
        #     else:
        #         return {}
        
        return {}
            

    def wipe_history(self):
        # with self.mutex:
        #     self.history = {}
        
        # todo ????????
        pass
    

    def delete_history_item(self, id_to_delete):
        # with self.mutex:
        #     self.history.pop(id_to_delete, None)
        
        # todo ????????
        pass
    
    
    def get_queue_info(self):
        prompt_info = {}
        exec_info = {}
        exec_info['queue_remaining'] = self.get_tasks_remaining()
        prompt_info['exec_info'] = exec_info
        return prompt_info

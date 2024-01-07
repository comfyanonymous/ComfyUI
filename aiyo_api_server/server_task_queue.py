

import threading

from framework.kafka_connection import KafkaConnection

from config.config import CONFIG
from framework.app_log import AppLog




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

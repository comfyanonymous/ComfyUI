from abc import ABC, abstractmethod


class PromptQueueInterface(ABC):
    @abstractmethod
    def __init__(self, server):
        pass

    @abstractmethod
    def put(self, item):
        pass

    @abstractmethod
    def get(self, timeout=None):
        pass

    @abstractmethod
    def get_history(self, prompt_id=None, max_items=None, offset=-1):
        pass

    @abstractmethod
    def delete_queue_item(self, function):
        pass

    @abstractmethod
    def wipe_history(self):
        pass

    @abstractmethod
    def delete_history_item(self, id_to_delete):
        pass

    @abstractmethod
    def get_tasks_remaining(self):
        pass

    @abstractmethod
    def get_current_queue(self):
        pass

    @abstractmethod
    def wipe_queue(self):
        pass

    @abstractmethod
    def set_flag(self, name, data):
        pass


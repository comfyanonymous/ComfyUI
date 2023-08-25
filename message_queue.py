import queue


# This queue is loop-driven by second created thread that processes additional prompt messages

class PromptExecutorMessageQueue:
    __PROMPT_QUEUE = queue.LifoQueue()

    @staticmethod
    def get_prompt_queue():
        return PromptExecutorMessageQueue.__PROMPT_QUEUE

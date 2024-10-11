import concurrent.futures
from typing import Callable

from pebble import ProcessPool, ProcessFuture

from ..component_model.executor_types import Executor, ExecutePromptArgs


class ProcessPoolExecutor(ProcessPool, Executor):
    def __init__(self, max_workers: int = 1):
        super().__init__(max_workers=1)


    def shutdown(self, wait=True, *, cancel_futures=False):
        if cancel_futures:
            raise NotImplementedError("cannot cancel futures in this implementation")
        if wait:
            self.close()
        else:
            self.stop()
        return

    def schedule(self, function: Callable,
                 args: list = (),
                 kwargs=None,
                 timeout: float = None) -> ProcessFuture:
        # todo: restart worker when there is insufficient VRAM or the workflows are sufficiently different
        # try:
        #     args: ExecutePromptArgs
        #     prompt, prompt_id, client_id, span_context, progress_handler, configuration = args
        #
        # except ValueError:
        #     pass
        if kwargs is None:
            kwargs = {}
        return super().schedule(function, args, kwargs, timeout)

    def submit(self, fn, /, *args, **kwargs) -> concurrent.futures.Future:
        return self.schedule(fn, args=list(args), kwargs=kwargs, timeout=None)

import concurrent.futures

from pebble import ProcessPool

from ..component_model.executor_types import Executor


class ProcessPoolExecutor(ProcessPool, Executor):
    def shutdown(self, wait=True, *, cancel_futures=False):
        if cancel_futures:
            raise NotImplementedError("cannot cancel futures in this implementation")
        if wait:
            self.close()
        else:
            self.stop()
        return

    def submit(self, fn, /, *args, **kwargs) -> concurrent.futures.Future:
        return self.schedule(fn, args=list(args), kwargs=kwargs, timeout=None)
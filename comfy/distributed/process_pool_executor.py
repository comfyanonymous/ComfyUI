import concurrent.futures
import contextvars
import multiprocessing
import pickle
import logging
from functools import partial
from typing import Callable, Any

from pebble import ProcessPool, ProcessFuture

from ..component_model.executor_types import Executor

logger = logging.getLogger(__name__)

def _wrap_with_context(context_data: bytes, func: Callable, *args, **kwargs) -> Any:
    new_ctx: contextvars.Context = pickle.loads(context_data)
    return new_ctx.run(func, *args, **kwargs)


class ProcessPoolExecutor(ProcessPool, Executor):
    def __init__(self,
                 max_workers: int = 1,
                 max_tasks: int = 0,
                 initializer: Callable = None,
                 initargs: list | tuple = (),
                 context: multiprocessing.context.BaseContext = None):
        if context is not None:
            logger.warning(f"A context was passed to a ProcessPoolExecutor when only spawn is supported (context={context})")
        context = multiprocessing.get_context('spawn')
        super().__init__(max_workers=max_workers, max_tasks=max_tasks, initializer=initializer, initargs=initargs, context=context)

    def shutdown(self, wait=True, *, cancel_futures=False):
        if cancel_futures:
            raise NotImplementedError("cannot cancel futures in this implementation")
        if wait:
            self.close()
        else:
            self.stop()
        return

    def schedule(self, function: Callable,
                 args: list | tuple = (),
                 kwargs=None,
                 timeout: float = None) -> ProcessFuture:
        if kwargs is None:
            kwargs = {}

        context_bin = pickle.dumps(contextvars.copy_context())
        unpack_context_then_run_function = partial(_wrap_with_context, context_bin, function)

        return super().schedule(unpack_context_then_run_function, args=args, kwargs=kwargs, timeout=timeout)

    def submit(self, fn, /, *args, **kwargs) -> concurrent.futures.Future:
        return self.schedule(fn, args=list(args), kwargs=kwargs, timeout=None)

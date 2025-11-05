import contextvars
import typing
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial

__version__ = '0.0.1'

from .process_pool_executor import ProcessPoolExecutor


class ContextVarExecutor(ThreadPoolExecutor):

    def submit(self, fn: typing.Callable, *args, **kwargs) -> Future:
        ctx = contextvars.copy_context()  # type: contextvars.Context

        return super().submit(partial(ctx.run, partial(fn, *args, **kwargs)))  # pylint: disable=no-member


class ContextVarProcessPoolExecutor(ProcessPoolExecutor):
    pass

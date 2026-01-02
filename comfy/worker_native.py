"""Native (in-process) worker for prompt execution."""

import time
import gc


class NativeWorker:
    """Executes prompts in the same process as the server."""

    def __init__(self, server_instance, interrupt_checker=None):
        self.server_instance = server_instance
        self.interrupt_checker = interrupt_checker
        self.executor = None
        self.last_gc_collect = 0
        self.need_gc = False
        self.gc_collect_interval = 10.0

    async def initialize(self):
        """Load nodes and set up executor. Returns node count."""
        from execution import PromptExecutor, CacheType
        from comfy.cli_args import args
        from comfy.execution_core import init_execution_environment, setup_progress_hook
        import comfy.model_management as mm
        import hook_breaker_ac10a0

        hook_breaker_ac10a0.save_functions()
        try:
            node_count = await init_execution_environment()
        finally:
            hook_breaker_ac10a0.restore_functions()

        interrupt_checker = self.interrupt_checker or mm.throw_exception_if_processing_interrupted
        setup_progress_hook(self.server_instance, interrupt_checker=interrupt_checker)

        cache_type = CacheType.CLASSIC
        if args.cache_lru > 0:
            cache_type = CacheType.LRU
        elif args.cache_ram > 0:
            cache_type = CacheType.RAM_PRESSURE
        elif args.cache_none:
            cache_type = CacheType.NONE

        self.executor = PromptExecutor(
            self.server_instance,
            cache_type=cache_type,
            cache_args={"lru": args.cache_lru, "ram": args.cache_ram}
        )
        return node_count

    def execute_prompt(self, prompt, prompt_id, extra_data, execute_outputs, server=None):
        self.executor.execute(prompt, prompt_id, extra_data, execute_outputs)
        return {
            'success': self.executor.success,
            'history_result': self.executor.history_result,
            'status_messages': self.executor.status_messages,
            'prompt_id': prompt_id
        }

    def handle_flags(self, flags):
        import comfy.model_management as mm
        import hook_breaker_ac10a0

        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            mm.unload_all_models()
            self.need_gc = True
            self.last_gc_collect = 0

        if free_memory:
            if self.executor:
                self.executor.reset()
            self.need_gc = True
            self.last_gc_collect = 0

        if self.need_gc:
            current_time = time.perf_counter()
            if (current_time - self.last_gc_collect) > self.gc_collect_interval:
                gc.collect()
                mm.soft_empty_cache()
                self.last_gc_collect = current_time
                self.need_gc = False
                hook_breaker_ac10a0.restore_functions()

    def interrupt(self, value=True):
        import comfy.model_management
        comfy.model_management.interrupt_current_processing(value)

    def mark_needs_gc(self):
        self.need_gc = True

    def get_gc_timeout(self):
        if self.need_gc:
            return max(self.gc_collect_interval - (time.perf_counter() - self.last_gc_collect), 0.0)
        return 1000.0

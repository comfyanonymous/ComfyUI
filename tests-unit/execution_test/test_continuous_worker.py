import asyncio

import torch

import execution
import main


class StopWorker(Exception):
    pass


def test_cooperative_worker_enters_owned_inference_scope(monkeypatch):
    observed = []

    class Queue:
        def set_cooperative(self, enabled):
            pass

        def is_cancelled(self, prompt_id):
            return False

        def get_if(self, predicate):
            observed.append(torch.is_inference_mode_enabled())
            raise StopWorker()

    monkeypatch.setattr(main, "prompt_executor_config", lambda: (execution.CacheType.NONE, {"lru": 0, "ram": 0, "ram_inactive": 0}))
    monkeypatch.setattr(main.comfy.memory_management, "set_ram_cache_release_state", lambda *args: None)
    monkeypatch.setattr(main.comfy.continuous_batching, "set_cancel_checker", lambda checker: None)

    async def run():
        try:
            await main.cooperative_prompt_worker(Queue(), object(), 2)
        except StopWorker:
            pass

    asyncio.run(run())
    assert observed == [True]
    assert not torch.is_inference_mode_enabled()


def test_cooperative_group_owner_clears_interrupt_once_before_start(monkeypatch):
    interrupt_calls = []

    class Queue:
        def __init__(self):
            self.get_calls = 0
            self.finished_drains = 0

        def set_cooperative(self, enabled):
            pass

        def is_cancelled(self, prompt_id):
            return False

        def get_if(self, predicate):
            self.get_calls += 1
            if self.get_calls == 1:
                return ((0, "prompt", {}, {}, [], {}), 0)
            raise StopWorker()

        def finish_cooperative_drain(self):
            self.finished_drains += 1

        def get_flags(self):
            return {}

    async def execute_prompt(*args, **kwargs):
        return None

    queue = Queue()
    monkeypatch.setattr(main, "prompt_executor_config", lambda: (execution.CacheType.NONE, {"lru": 0, "ram": 0, "ram_inactive": 0}))
    monkeypatch.setattr(main, "execute_prompt_async", execute_prompt)
    monkeypatch.setattr(main.nodes, "interrupt_processing", lambda value=True: interrupt_calls.append(value))
    monkeypatch.setattr(main.comfy.memory_management, "set_ram_cache_release_state", lambda *args: None)
    monkeypatch.setattr(main.comfy.continuous_batching, "set_cancel_checker", lambda checker: None)
    monkeypatch.setattr(main.comfy.model_management, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(main.hook_breaker_ac10a0, "restore_functions", lambda: None)
    monkeypatch.setattr(main.gc, "collect", lambda: None)
    monkeypatch.setattr(main.asset_seeder, "is_disabled", lambda: True)
    monkeypatch.setattr(main.asset_seeder, "resume", lambda: None)

    async def run():
        try:
            await main.cooperative_prompt_worker(queue, object(), 1)
        except StopWorker:
            pass

    asyncio.run(run())
    assert interrupt_calls == [False]
    assert queue.finished_drains == 1


def test_prompt_worker_preserves_requested_max_prompts(monkeypatch):
    observed = []

    async def cooperative_worker(q, server_instance, max_prompts):
        observed.append(max_prompts)

    monkeypatch.setattr(main, "cooperative_prompt_worker", cooperative_worker)
    for max_prompts in (1, 3):
        monkeypatch.setattr(main.args, "continuous_batching", max_prompts)
        main.prompt_worker(object(), object())

    assert observed == [1, 3]


def _queue_item(prompt, outputs, preview_method=None):
    return (0, "prompt-id", prompt, {"preview_method": preview_method}, outputs)


def _continuous_prompt(sampler_ids, sampler_type="AnimaContinuousKSampler"):
    prompt = {
        "model": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
        "output": {"class_type": "SaveImage", "inputs": {}},
    }
    for sampler_id in sampler_ids:
        prompt[sampler_id] = {"class_type": sampler_type, "inputs": {"model": ["model", 0]}}
    return prompt


def test_continuous_prompt_key_accepts_all_connected_sampler_families():
    keys = []
    for sampler_type in main.comfy.continuous_batching.CONTINUOUS_SAMPLER_NODE_FAMILIES:
        prompt = _continuous_prompt(["sampler"], sampler_type)
        prompt["output"]["inputs"]["images"] = ["sampler", 0]
        key = main.continuous_prompt_key(_queue_item(prompt, ["output"]))
        assert key is not None
        assert key[0] == sampler_type
        keys.append(key)
    assert len(set(keys)) == 3


def test_continuous_prompt_key_requires_exactly_one_connected_sampler():
    disconnected = _continuous_prompt(["sampler"])
    assert main.continuous_prompt_key(_queue_item(disconnected, ["output"])) is None

    multiple = _continuous_prompt(["sampler-a", "sampler-b"])
    multiple["output"]["inputs"].update({"first": ["sampler-a", 0], "second": ["sampler-b", 0]})
    assert main.continuous_prompt_key(_queue_item(multiple, ["output"])) is None


def test_continuous_prompt_key_tracks_model_graph_and_preview_method():
    first = _continuous_prompt(["sampler"])
    first["output"]["inputs"]["images"] = ["sampler", 0]
    second = _continuous_prompt(["sampler"])
    second["output"]["inputs"]["images"] = ["sampler", 0]
    second["model"]["inputs"]["ckpt_name"] = "other.safetensors"

    first_key = main.continuous_prompt_key(_queue_item(first, ["output"], "auto"))
    assert first_key != main.continuous_prompt_key(_queue_item(first, ["output"], "none"))
    assert first_key != main.continuous_prompt_key(_queue_item(second, ["output"], "auto"))

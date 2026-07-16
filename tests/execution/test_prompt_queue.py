import execution


class TestServer:
    def queue_updated(self):
        pass


def test_cooperative_cancel_is_prompt_scoped_without_node_class_checks(monkeypatch):
    interrupted = []
    monkeypatch.setattr(execution.nodes, "interrupt_processing", lambda: interrupted.append(True))
    queue = execution.PromptQueue(TestServer())
    prompt = {"1": {"class_type": "UnrelatedNode", "inputs": {}}}
    queue.put((0, "first", prompt, {}, [], {}))
    queue.put((1, "second", prompt, {}, [], {}))
    _, first_id = queue.get(timeout=0)
    _, second_id = queue.get(timeout=0)

    queue.set_cooperative(True)
    assert queue.interrupt_if_running("first")
    assert queue.is_cancelled("first")
    assert not queue.is_cancelled("second")
    assert interrupted == []

    queue.task_done(first_id, {}, None)
    assert not queue.is_cancelled("first")
    queue.task_done(second_id, {}, None)


def test_single_cooperative_cancel_also_wakes_synchronous_execution(monkeypatch):
    interrupted = []
    monkeypatch.setattr(execution.nodes, "interrupt_processing", lambda: interrupted.append(True))
    queue = execution.PromptQueue(TestServer())
    queue.put((0, "only", {}, {}, [], {}))
    _, item_id = queue.get(timeout=0)
    queue.set_cooperative(True)

    assert queue.interrupt_if_running("only")
    assert queue.is_cancelled("only")
    assert queue.is_cooperative_draining()
    assert interrupted == [True]

    queue.task_done(item_id, {}, None)
    queue.finish_cooperative_drain()


def test_completed_result_wins_if_cancel_arrives_after_compute(monkeypatch):
    monkeypatch.setattr(execution.nodes, "interrupt_processing", lambda: None)
    queue = execution.PromptQueue(TestServer())
    queue.put((0, "completed", {}, {}, [], {}))
    _, item_id = queue.get(timeout=0)
    queue.set_cooperative(True)

    assert queue.interrupt_if_running("completed")
    status = execution.PromptQueue.ExecutionStatus("success", True, [])
    queue.task_done(item_id, {"outputs": {"node": "result"}}, status)

    assert not queue.is_cancelled("completed")
    history = queue.get_history("completed")["completed"]
    assert history["status"]["status_str"] == "success"
    assert history["outputs"] == {"node": "result"}


def test_global_cooperative_interrupt_cancels_all_and_blocks_admission_until_drain(monkeypatch):
    interrupted = []
    monkeypatch.setattr(execution.nodes, "interrupt_processing", lambda: interrupted.append(True))
    queue = execution.PromptQueue(TestServer())
    for number, prompt_id in enumerate(("first", "second", "next")):
        queue.put((number, prompt_id, {}, {}, [], {}))
    _, first_id = queue.get(timeout=0)
    _, second_id = queue.get(timeout=0)
    queue.set_cooperative(True)

    queue.interrupt_all_running()

    assert queue.is_cancelled("first")
    assert queue.is_cancelled("second")
    assert interrupted == [True]
    assert queue.get_if(lambda item: True) is None

    queue.task_done(first_id, {}, None)
    queue.task_done(second_id, {}, None)
    queue.finish_cooperative_drain()
    item, next_id = queue.get_if(lambda item: True)
    assert item[1] == "next"
    queue.task_done(next_id, {}, None)


def test_legacy_cancel_uses_global_interrupt(monkeypatch):
    interrupted = []
    monkeypatch.setattr(execution.nodes, "interrupt_processing", lambda: interrupted.append(True))
    queue = execution.PromptQueue(TestServer())
    queue.put((0, "prompt", {}, {}, [], {}))
    queue.get(timeout=0)

    assert queue.interrupt_if_running("prompt")
    assert interrupted == [True]
    assert not queue.is_cancelled("prompt")


def test_targeted_cancel_does_not_interrupt_different_serial_prompt(monkeypatch):
    interrupted = []
    monkeypatch.setattr(execution.nodes, "interrupt_processing", lambda: interrupted.append(True))
    queue = execution.PromptQueue(TestServer())
    queue.put((0, "running", {}, {}, [], {}))
    queue.get(timeout=0)

    assert not queue.interrupt_if_running("different")
    assert interrupted == []


def test_get_if_leaves_incompatible_head_queued():
    queue = execution.PromptQueue(TestServer())
    queue.put((0, "first", {}, {}, [], {}))

    assert queue.get_if(lambda item: item[1] == "other") is None
    assert queue.get(timeout=0)[0][1] == "first"

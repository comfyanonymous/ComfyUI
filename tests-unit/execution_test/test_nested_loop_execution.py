import nodes

import comfy_extras.nodes_loop as nodes_loop
from execution import PromptExecutor


class Constant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        return (value,)


class Increment:
    calls = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        value += 1
        self.calls.append(value)
        return (value,)


class Capture:
    values = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, value):
        self.values.append(value)
        return ()


class CapturePassthrough:
    values = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, value):
        self.values.append(value)
        return (value,)


class Pair:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("*",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"

    def execute(self, value):
        return ([value, str(value)],)


class ListBackedScalar:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("*",)
    FUNCTION = "execute"

    def execute(self, value):
        return ([[value]],)


class Server:
    client_id = None

    def send_sync(self, *args, **kwargs):
        pass


def test_nested_loops_execute_each_body_once_without_final_requeue(monkeypatch):
    Increment.calls = []
    Capture.values = []
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StartLoop", nodes_loop.StartLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EndLoop", nodes_loop.EndLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TestConstant", Constant)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TestIncrement", Increment)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TestCapture", Capture)
    monkeypatch.setattr(
        nodes_loop,
        "PromptServer",
        type("PromptServer", (), {"instance": type("Progress", (), {"send_progress_text": lambda *args: None})()}),
    )

    prompt = {
        "constant": {
            "class_type": "TestConstant",
            "inputs": {"value": 0},
        },
        "outer": {
            "class_type": "StartLoop",
            "inputs": {
                "mode": "simple",
                "mode.num_iterations": 2,
                "initial_iteration_value": ["constant", 0],
            },
        },
        "inner": {
            "class_type": "StartLoop",
            "inputs": {
                "mode": "simple",
                "mode.num_iterations": 2,
                "iteration_outer": ["outer", 0],
                "initial_iteration_value": ["outer", 4],
            },
        },
        "increment": {
            "class_type": "TestIncrement",
            "inputs": {"value": ["inner", 4]},
        },
        "inner_close": {
            "class_type": "EndLoop",
            "inputs": {
                "output_value": ["increment", 0],
                "next_iteration_value": ["increment", 0],
                "accumulate": False,
            },
        },
        "outer_close": {
            "class_type": "EndLoop",
            "inputs": {
                "output_value": ["inner_close", 0],
                "next_iteration_value": ["inner_close", 0],
                "accumulate": False,
            },
        },
        "capture": {
            "class_type": "TestCapture",
            "inputs": {"value": ["outer_close", 0]},
        },
    }
    executor = PromptExecutor(
        Server(),
        cache_type=False,
        cache_args={"ram": 0, "ram_inactive": 0},
    )

    executor.execute(prompt, "nested-loop-test", execute_outputs=["capture"])

    assert executor.success
    assert Increment.calls == [1, 2, 3, 4]
    assert Capture.values == [4]


def test_loop_executes_termination_without_carried_or_output_value(monkeypatch):
    Increment.calls = []
    CapturePassthrough.values = []
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StartLoop", nodes_loop.StartLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EndLoop", nodes_loop.EndLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TestIncrement", Increment)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TestCapturePassthrough", CapturePassthrough)
    monkeypatch.setattr(
        nodes_loop,
        "PromptServer",
        type("PromptServer", (), {"instance": type("Progress", (), {"send_progress_text": lambda *args: None})()}),
    )

    prompt = {
        "loop": {
            "class_type": "StartLoop",
            "inputs": {
                "mode": "simple",
                "mode.num_iterations": 2,
            },
        },
        "increment": {
            "class_type": "TestIncrement",
            "inputs": {"value": ["loop", 0]},
        },
        "preview": {
            "class_type": "TestCapturePassthrough",
            "inputs": {"value": ["increment", 0]},
        },
        "close": {
            "class_type": "EndLoop",
            "inputs": {
                "accumulate": False,
                "termination0": ["preview", 0],
            },
        },
    }
    executor = PromptExecutor(
        Server(),
        cache_type=False,
        cache_args={"ram": 0, "ram_inactive": 0},
    )

    executor.execute(prompt, "termination-only-loop-test", execute_outputs=["preview"])

    assert executor.success
    assert Increment.calls == [1, 2]
    assert CapturePassthrough.values == [1, 2]


def run_nested_accumulation(monkeypatch, producer_name, producer_class):
    Capture.values = []
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StartLoop", nodes_loop.StartLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EndLoop", nodes_loop.EndLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, producer_name, producer_class)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TestCapture", Capture)
    monkeypatch.setattr(
        nodes_loop,
        "PromptServer",
        type("PromptServer", (), {"instance": type("Progress", (), {"send_progress_text": lambda *args: None})()}),
    )

    prompt = {
        "outer": {
            "class_type": "StartLoop",
            "inputs": {"mode": "simple", "mode.num_iterations": 2},
        },
        "inner": {
            "class_type": "StartLoop",
            "inputs": {
                "mode": "simple",
                "mode.num_iterations": 2,
                "iteration_outer": ["outer", 0],
            },
        },
        "producer": {
            "class_type": producer_name,
            "inputs": {"value": ["inner", 0]},
        },
        "inner_close": {
            "class_type": "EndLoop",
            "inputs": {
                "output_value": ["producer", 0],
                "accumulate": True,
            },
        },
        "outer_close": {
            "class_type": "EndLoop",
            "inputs": {
                "output_value": ["inner_close", 0],
                "accumulate": True,
            },
        },
        "capture": {
            "class_type": "TestCapture",
            "inputs": {"value": ["outer_close", 0]},
        },
    }
    executor = PromptExecutor(
        Server(),
        cache_type=False,
        cache_args={"ram": 0, "ram_inactive": 0},
    )

    executor.execute(prompt, "nested-loop-accumulation-test", execute_outputs=["capture"])

    assert executor.success
    return Capture.values


def test_nested_loop_concatenates_zipped_lists(monkeypatch):
    assert run_nested_accumulation(monkeypatch, "TestPair", Pair) == [
        [0, 1, 0, 1],
        ["0", "1", "0", "1"],
    ]


def test_nested_loop_does_not_flatten_list_backed_scalars(monkeypatch):
    assert run_nested_accumulation(monkeypatch, "TestListBackedScalar", ListBackedScalar) == [
        [[[0]], [[0]]],
        [[[1]], [[1]]],
    ]

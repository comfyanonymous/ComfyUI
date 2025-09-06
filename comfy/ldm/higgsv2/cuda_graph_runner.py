import torch
import torch.nn as nn
from typing import Optional, Dict
import gc

_NUM_WARMUP_ITERS = 2

class CUDAGraphRunner(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

        self._graph: Optional[torch.cuda.CUDAGraph] = None

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(self, *args, **kwargs):
        assert self._graph is None

        for _ in range(_NUM_WARMUP_ITERS):
            self.model(*args, **kwargs)

        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool = kwargs.get("memory_pool", None), stream = kwargs.get("stream", None)):
            last_hidden_states = self.model(*args, **kwargs)
            gc.collect()

        torch.cuda.synchronize()

        self.input_buffers = {
            "args": [arg for arg in args if isinstance(arg, torch.Tensor)],
            "kwargs": {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)},
        }

        self.output_buffers = {
            "hidden_states": last_hidden_states
        }

    def forward(self, *args, **kwargs):

        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                self.input_buffers["args"][i].copy_(arg, non_blocking=True)

        for k, v in kwargs.items():
            if k in self.input_buffers["kwargs"] and isinstance(v, torch.Tensor):
                self.input_buffers["kwargs"][k].copy_(v, non_blocking=True)

        self.graph.replay()

        return self.output_buffers["hidden_states"]

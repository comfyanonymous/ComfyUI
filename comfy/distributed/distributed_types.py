from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal, List

from ..api.components.schema.prompt import PromptDict, Prompt
from ..auth.permissions import ComfyJwt, jwt_decode
from ..component_model.queue_types import NamedQueueTuple, TaskInvocation, ExecutionStatus


@dataclass
class DistributedBase:
    prompt_id: str
    user_token: str

    @property
    def user_id(self) -> str:
        return self.decoded_token["sub"]

    @property
    def decoded_token(self) -> ComfyJwt:
        return jwt_decode(self.user_token)


@dataclass
class RpcRequest(DistributedBase):
    prompt: dict | PromptDict

    async def as_queue_tuple(self) -> NamedQueueTuple:
        # this loads the nodes in this instance
        # should always be okay to call in an executor
        from ..cmd.execution import validate_prompt
        from ..component_model.make_mutable import make_mutable
        mutated_prompt_dict = make_mutable(self.prompt)
        validation_tuple = await validate_prompt(self.prompt_id, mutated_prompt_dict)
        return NamedQueueTuple(queue_tuple=(0, self.prompt_id, mutated_prompt_dict, {}, validation_tuple[2]))

    @classmethod
    def from_dict(cls, request_dict):
        request = RpcRequest(**request_dict)
        request.prompt = Prompt.validate(request.prompt)
        return request


@dataclass
class RpcReply(DistributedBase):
    outputs: dict
    execution_status: ExecutionStatus | Tuple[Literal['success', 'error'], bool, List[str]]

    @staticmethod
    def from_task_invocation(task_invocation: TaskInvocation, user_token: str) -> 'RpcReply':
        return RpcReply(str(task_invocation.item_id), user_token, task_invocation.outputs, task_invocation.status)

    def as_task_invocation(self):
        return TaskInvocation(self.prompt_id, self.outputs, ExecutionStatus(*self.execution_status))

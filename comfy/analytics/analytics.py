import asyncio
import json
import sys
import uuid
from typing import Optional

from .multi_event_tracker import MultiEventTracker
from .plausible import PlausibleTracker
from ..api.components.schema.prompt import Prompt

_event_tracker: MultiEventTracker


def initialize_event_tracking(loop: Optional[asyncio.AbstractEventLoop] = None):
    loop = loop or asyncio.get_event_loop()
    assert loop is not None
    _event_trackers = []
    # perform the imports at the time this is invoked to prevent side effects and ordering issues
    from ..cli_args import args

    identity = str(uuid.uuid4())
    if args.analytics_use_identity_provider and sys.platform == "nt":
        from .identity_provider_nt import get_user_name
        identity = get_user_name()

    if args.plausible_analytics_domain is not None and args.plausible_analytics_base_url is not None:
        _event_trackers.append(PlausibleTracker(loop, user_agent=identity, base_url=args.plausible_analytics_base_url,
                                                domain=args.plausible_analytics_domain))

    if len(_event_trackers) == 0:
        return

    _event_tracker = MultiEventTracker(_event_trackers)

    def track_event(name: str, url: str = "app://comfyui", props: Optional[dict] = None):
        # not awaited, we don't care about event tracking in terms of blocking
        loop.create_task(_event_tracker.track_event(name, url, props=props))

    # patch nodes
    from ..nodes.base_nodes import SaveImage, CLIPTextEncode, LoraLoader, CheckpointLoaderSimple
    from ..cmd.execution import PromptQueue
    from comfy.component_model.queue_types import QueueItem

    prompt_queue_put = PromptQueue.put

    def prompt_queue_put_tracked(self: PromptQueue, item: QueueItem):
        prompt = Prompt.validate(item.prompt)

        samplers = [v for _, v in prompt.items() if
                    "positive" in v.inputs and "negative" in v.inputs]

        positive_prompt_ids = []
        negative_prompt_ids = []
        for sampler in samplers:
            try:
                # duck typed
                key, _ = sampler.inputs['positive']
                positive_prompt_ids.append(key)
            except:
                pass
            try:
                key, _ = sampler.inputs['negative']
                negative_prompt_ids.append(key)
            except:
                pass

        positive_prompts = "; ".join(frozenset(str(prompt[x].inputs["text"]) for x in positive_prompt_ids if
                                               prompt[x].class_type == CLIPTextEncode.__name__))
        negative_prompts = "; ".join(frozenset(str(prompt[x].inputs["text"]) for x in negative_prompt_ids if
                                               prompt[x].class_type == CLIPTextEncode.__name__))
        loras = "; ".join(frozenset(
            str(node.inputs["lora_name"]) for node in prompt.values() if
            node.class_type == LoraLoader.__name__))
        checkpoints = "; ".join(frozenset(str(node.inputs["ckpt_name"]) for node in prompt.values() if
                                          node.class_type == CheckpointLoaderSimple.__name__))
        prompt_str = json.dumps(item.queue_tuple, separators=(',', ':'))
        len_prompt_str = len(prompt_str)
        prompt_str_pieces = []
        for i in range(0, len_prompt_str, 1000):
            prompt_str_pieces += [prompt_str[i:min(i + 1000, len_prompt_str)]]
        prompt_str_props = {}
        for i, prompt_str_piece in enumerate(prompt_str_pieces):
            prompt_str_props[f"prompt.{i}"] = prompt_str_piece
        try:
            track_event(SaveImage.__name__, props={
                "positive_prompts": positive_prompts,
                "negative_prompts": negative_prompts,
                "loras": loras,
                "checkpoints": checkpoints,
                **prompt_str_props
            })
        except:
            # prevent analytics exceptions from cursing us
            pass

        return prompt_queue_put(self, item)

    PromptQueue.put = prompt_queue_put_tracked

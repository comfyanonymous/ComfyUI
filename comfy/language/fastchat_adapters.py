from __future__ import annotations

from typing import Optional

from fastchat.conversation import Conversation, get_conv_template
from fastchat.model.model_adapter import BaseModelAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer


class Phi3Adapter(BaseModelAdapter):
    """The model adapter for Microsoft/Phi-3-mini-128k-instruct"""

    def match(self, model_path: str):
        return "phi-3-mini-128k-instruct" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        self.model = model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    def generate_prompt(self, instruction: str, input: Optional[str] = None) -> str:
        if input:
            prompt = f"<|user|>\n{instruction}\n{input}<|end|>\n<|assistant|>"
        else:
            prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>"
        return prompt

    def generate_response(self, messages, max_new_tokens=500, temperature=0.0, do_sample=False):
        prompt = self.generate_prompt(messages[-1]["content"])

        for i in range(len(messages) - 2, -1, -1):
            if messages[i]["role"] == "user":
                prompt = self.generate_prompt(messages[i]["content"]) + prompt
            elif messages[i]["role"] == "assistant":
                prompt = messages[i]["content"] + prompt

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        output_ids = self.model.generate(
            input_ids,
            **generation_kwargs
        )

        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output = output.replace(prompt, "").strip()

        return output

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("phi-3-mini")

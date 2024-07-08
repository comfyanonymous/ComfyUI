import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration,T5Config,modeling_utils
import os
import comfy.utils

class MT5Embedder(nn.Module):
    available_models = ["t5-v1_1-xxl"]

    def __init__(
        self,
        model_dir="t5-v1_1-xxl",
        model_kwargs=None,
        torch_dtype=None,
        use_tokenizer_only=False,
        conditional_generation=False,
        max_length=128,
        ksampler = False
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.max_length = max_length
        if model_kwargs is None:
            model_kwargs = {
                # "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }
        model_kwargs["device_map"] = {"shared": self.device, "encoder": self.device}
        if not ksampler:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            if use_tokenizer_only:
                return
            """
            if conditional_generation:
                self.model = None
                self.generation_model = T5ForConditionalGeneration.from_pretrained(
                    model_dir
                )
                return
            """
            self.model = T5EncoderModel.from_pretrained(model_dir, **model_kwargs).eval().to(self.torch_dtype)
        else:
            tokenizer_path = os.path.join( # local
				os.path.dirname(os.path.realpath(__file__)), "..", "..",
				"tokenizer_mt5",
			)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            #self.tokenizer.eval().to(self.device) 
            if use_tokenizer_only:
                return
            """
            if conditional_generation:
                self.model = None
                self.generation_model = T5ForConditionalGeneration.from_pretrained(
                    model_dir
                )
                return
            """
            textmodel_json_config = os.path.join(
				os.path.dirname(os.path.realpath(__file__)), "..", "..",
				f"config_mt5.json"
			)
            config = T5Config.from_json_file(textmodel_json_config)
            with modeling_utils.no_init_weights():
                self.model = T5EncoderModel(config)
            sd = comfy.utils.load_torch_file(model_dir)
            self.model.load_state_dict(sd, strict=False)
            self.model.eval().to(self.device)    
   
    def get_tokens_and_mask(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        tokens = text_tokens_and_mask["input_ids"][0]
        mask = text_tokens_and_mask["attention_mask"][0]
        # tokens = torch.tensor(tokens).clone().detach()
        # mask = torch.tensor(mask, dtype=torch.bool).clone().detach()
        return tokens, mask

    def get_text_embeddings(self, texts, attention_mask=True, layer_index=-1):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=text_tokens_and_mask["input_ids"].to(self.device),
                attention_mask=text_tokens_and_mask["attention_mask"].to(self.device)
                if attention_mask
                else None,
                output_hidden_states=True,
            )
            text_encoder_embs = outputs["hidden_states"][layer_index].detach()

        return text_encoder_embs, text_tokens_and_mask["attention_mask"].to(self.device)

    @torch.no_grad()
    def __call__(self, tokens, attention_mask, layer_index=-1):
        with torch.cuda.amp.autocast():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        z = outputs.hidden_states[layer_index].detach()
        return z

    def general(self, text: str):
        # input_ids = input_ids = torch.tensor([list(text.encode("utf-8"))]) + num_special_tokens
        input_ids = self.tokenizer(text, max_length=128).input_ids
        print(input_ids)
        outputs = self.generation_model(input_ids)
        return outputs
    

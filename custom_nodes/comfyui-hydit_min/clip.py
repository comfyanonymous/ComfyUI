import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
from .hydit.modules.text_encoder import MT5Embedder
from transformers import BertModel, BertTokenizer, AutoTokenizer
import torch
import os
from transformers import BertConfig, BertModel
from transformers import AutoTokenizer, modeling_utils
#import pdb

class CLIP:
    def __init__(self, root, text_encoder_path = None, t5_text_encoder_path = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if text_encoder_path == None:
            text_encoder_path = os.path.join(root,"clip_text_encoder")
            clip_text_encoder = BertModel.from_pretrained(str(text_encoder_path), False, revision=None).to(self.device)
        else:         
            config = BertConfig.from_json_file(os.path.join(os.path.dirname(os.path.realpath(__file__)),"config_clip.json"))
            with modeling_utils.no_init_weights():
                clip_text_encoder = BertModel(config) 
            sd = comfy.utils.load_torch_file(text_encoder_path)
            prefix = "bert."
            state_dict = {}
            for key in sd:
                nkey = key
                if key.startswith(prefix):
                        nkey = key[len(prefix):]
                state_dict[nkey] = sd[key]

            m, e = clip_text_encoder.load_state_dict(state_dict, strict=False)
            if len(m) > 0 or len(e) > 0:
                print(f"HYDiT: clip missing {len(m)} keys ({len(e)} extra)")
                #clip_text_encoder.load_state_dict(sd, strict=False)
            clip_text_encoder.eval().to(self.device)    

        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tokenizer_clip")
        self.tokenizer = HyBertTokenizer(tokenizer_path)
        #assert(0)
        
        if t5_text_encoder_path == None:
            t5_text_encoder_path = os.path.join(root,'mt5')
            embedder_t5 = MT5Embedder(t5_text_encoder_path, torch_dtype=torch.float16, max_length=256)
            self.tokenizer_t5 = HyT5Tokenizer(embedder_t5.tokenizer, max_length=embedder_t5.max_length)
            self.embedder_t5 = embedder_t5.model
        else:

            embedder_t5 = MT5Embedder(t5_text_encoder_path, torch_dtype=torch.float16, max_length=256, ksampler=True)
            self.tokenizer_t5 = HyT5Tokenizer(embedder_t5.tokenizer, max_length=embedder_t5.max_length)
            self.embedder_t5 = embedder_t5.model
        

        self.cond_stage_model = clip_text_encoder
        

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        t5_tokens = self.tokenizer_t5.tokenize(text)
        tokens.update(t5_tokens)
        return tokens
    
    def tokenize_t5(self, text):
        return self.tokenizer_t5.tokenize(text)

    def encode_from_tokens(self, tokens, return_pooled=False):
        attention_mask = tokens['attention_mask'].to(self.device)
        with torch.no_grad():
            prompt_embeds = self.cond_stage_model(
                    tokens['text_input_ids'].to(self.device),
                    attention_mask=attention_mask
                )
            prompt_embeds = prompt_embeds[0]
        t5_attention_mask = tokens['t5_attention_mask'].to(self.device)
        with torch.no_grad():
            t5_prompt_cond = self.embedder_t5(
                    tokens['t5_text_input_ids'].to(self.device),
                    attention_mask=t5_attention_mask
                )
            t5_embeds = t5_prompt_cond[0]

        addit_embeds = {
                "t5_embeds": t5_embeds,
                "attention_mask": attention_mask.float(),
                "t5_attention_mask": t5_attention_mask.float()
            }
        prompt_embeds.addit_embeds = addit_embeds

        if return_pooled:
            return prompt_embeds, None
        else:
            return prompt_embeds

class HyBertTokenizer:
    def __init__(self, tokenizer_path=None, max_length=77, truncation=True, return_attention_mask=True, device='cpu'):
        self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
        self.max_length = self.tokenizer.model_max_length or max_length
        self.truncation = truncation
        self.return_attention_mask = return_attention_mask
        self.device = device

    def tokenize(self, text:str):
        text_inputs = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_length,
                truncation=self.truncation,
                return_attention_mask=self.return_attention_mask,
			    add_special_tokens = True,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        tokens = {
            'text_input_ids': text_input_ids,
            'attention_mask': attention_mask
        }
        return tokens
    
class HyT5Tokenizer:
    def __init__(self, tokenizer, tokenizer_path = None, max_length=77, truncation=True, return_attention_mask=True, device='cpu'):
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            self.tokenizer = tokenizer

        self.max_length = max_length
        self.truncation = truncation
        self.return_attention_mask = return_attention_mask
        self.device = device

    def tokenize(self, text:str):
        text_inputs = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_length,
                truncation=self.truncation,
                return_attention_mask=self.return_attention_mask,
			    add_special_tokens = True,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        tokens = {
            't5_text_input_ids': text_input_ids,
            't5_attention_mask': attention_mask
        }
        return tokens


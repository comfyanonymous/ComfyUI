from comfy import sd1_clip
from .qwen_image import QwenImageTokenizer, QwenImageTEModel
from .llama import Qwen25_7BVLI


class Kandinsky5Tokenizer(QwenImageTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.llama_template = "<|im_start|>system\nYou are a prompt engineer. Describe the video in detail.\nDescribe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.\nDescribe the location of the video, main characters or objects and their action.\nDescribe the dynamism of the video and presented actions.\nName the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or screen content.\nDescribe the visual effects, postprocessing and transitions if they are presented in the video.\nPay attention to the order of key actions shown in the scene.<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = super().tokenize_with_weights(text, return_word_ids, **kwargs)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs)

        return out


class Kandinsky5TokenizerImage(Kandinsky5Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.llama_template = "<|im_start|>system\nYou are a promt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>"


class Qwen25_7BVLIModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-1, dtype=None, attention_mask=True, model_options={}):
        llama_quantization_metadata = model_options.get("llama_quantization_metadata", None)
        if llama_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = llama_quantization_metadata
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=Qwen25_7BVLI, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)


class Kandinsky5TEModel(QwenImageTEModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super(QwenImageTEModel, self).__init__(device=device, dtype=dtype, name="qwen25_7b", clip_model=Qwen25_7BVLIModel, model_options=model_options)
        self.clip_l = sd1_clip.SDClipModel(device=device, dtype=dtype, return_projected_pooled=False, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        cond, p, extra = super().encode_token_weights(token_weight_pairs, template_end=-1)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs["l"])

        return cond, l_pooled, extra

    def set_clip_options(self, options):
        super().set_clip_options(options)
        self.clip_l.set_clip_options(options)

    def reset_clip_options(self):
        super().reset_clip_options()
        self.clip_l.reset_clip_options()

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return super().load_sd(sd)

def te(dtype_llama=None, llama_quantization_metadata=None):
    class Kandinsky5TEModel_(Kandinsky5TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["llama_quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Kandinsky5TEModel_

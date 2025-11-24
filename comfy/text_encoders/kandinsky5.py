from comfy import sd1_clip
from .qwen_image import QwenImageTokenizer, QwenImageTEModel, Qwen25_7BVLIModel

class Kandinsky5Tokenizer(QwenImageTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        # yes the typo "promt" was also in the original template...
        self.llama_template = "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.\nDescribe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.\nDescribe the location of the video, main characters or objects and their action.\nDescribe the dynamism of the video and presented actions.\nName the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.\nDescribe the visual effects, postprocessing and transitions if they are presented in the video.\nPay attention to the order of key actions shown in the scene.<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
        self.llama_template_image2video = "<|im_start|>system\nYou are a promt engineer. Your task is to create a highly detailed and effective video description based on a provided input image.\nDescribe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.\nDescribe main characters actions.\nDescribe the dynamism of the video and presented actions.\nName the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.\nDescribe the visual effects, postprocessing and transitions if they are presented in the video.\nPay attention to the order of key actions shown in the scene.<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)


    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = super().tokenize_with_weights(text, return_word_ids, **kwargs)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs)

        return out


class Kandinsky5TEModel(QwenImageTEModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super(QwenImageTEModel, self).__init__(device=device, dtype=dtype, name="qwen25_7b", clip_model=Qwen25_7BVLIModel, model_options=model_options)
        self.clip_l = sd1_clip.SDClipModel(device=device, dtype=dtype, return_projected_pooled=False, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        #tok_pairs = token_weight_pairs["qwen25_7b"][0]
        token_weight_pairs_l = token_weight_pairs["l"]
        template_end = -1

        cond, p, extra = super().encode_token_weights(token_weight_pairs, template_end=template_end)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)

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

def te(dtype_llama=None, llama_scaled_fp8=None):
    class Kandinsky5TEModel_(Kandinsky5TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_scaled_fp8 is not None and "scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["qwen_scaled_fp8"] = llama_scaled_fp8
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Kandinsky5TEModel_

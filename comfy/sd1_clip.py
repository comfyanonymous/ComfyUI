import os

from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig
import torch

class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        z_empty = self.encode(self.empty_tokens)
        output = []
        for x in token_weight_pairs:
            tokens = [list(map(lambda a: a[0], x))]
            z = self.encode(tokens)
            for i in range(len(z)):
                for j in range(len(z[i])):
                    weight = x[j][1]
                    z[i][j] = (z[i][j] - z_empty[0][j]) * weight + z_empty[0][j]
            output += [z]
        if (len(output) == 0):
            return self.encode(self.empty_tokens)
        return torch.cat(output, dim=-2)

class SD1ClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77,
                 freeze=True, layer="last", layer_idx=None, textmodel_json_config=None, textmodel_path=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        if textmodel_path is not None:
            self.transformer = CLIPTextModel.from_pretrained(textmodel_path)
        else:
            if textmodel_json_config is None:
                textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_clip_config.json")
            config = CLIPTextConfig.from_json_file(textmodel_json_config)
            self.transformer = CLIPTextModel(config)

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.empty_tokens = [[49406] + [49407] * 76]
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) <= 12
            self.clip_layer(layer_idx)

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def clip_layer(self, layer_idx):
        if abs(layer_idx) >= 12:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def forward(self, tokens):
        tokens = torch.LongTensor(tokens).to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")

        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
            z = self.transformer.text_model.final_layer_norm(z)

        return z

    def encode(self, tokens):
        return self(tokens)

def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result

def token_weights(string, current_weight):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ')' and x[0] == '(':
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx+1:])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out

def escape_important(text):
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    return text

def unescape_important(text):
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    return text

class SD1Tokenizer:
    def __init__(self, tokenizer_path=None, max_length=77, pad_with_end=True):
        if tokenizer_path is None:
            tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_tokenizer")
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        empty = self.tokenizer('')["input_ids"]
        self.start_token = empty[0]
        self.end_token = empty[1]
        self.pad_with_end = pad_with_end
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def tokenize_with_weights(self, text):
        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        tokens = []
        for t in parsed_weights:
            tt = self.tokenizer(unescape_important(t[0]))["input_ids"][1:-1]
            for x in tt:
                tokens += [(x, t[1])]

        out_tokens = []
        for x in range(0, len(tokens), self.max_length - 2):
            o_token = [(self.start_token, 1.0)] + tokens[x:min(self.max_length - 2 + x, len(tokens))]
            o_token += [(self.end_token, 1.0)]
            if self.pad_with_end:
                o_token +=[(self.end_token, 1.0)] * (self.max_length - len(o_token))
            else:
                o_token +=[(0, 1.0)] * (self.max_length - len(o_token))

            out_tokens += [o_token]

        return out_tokens

    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))

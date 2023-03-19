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

    def set_up_textual_embeddings(self, tokens, current_embeds):
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0]
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, int):
                    tokens_temp += [y]
                else:
                    embedding_weights += [y]
                    tokens_temp += [next_new_token]
                    next_new_token += 1
            out_tokens += [tokens_temp]

        if len(embedding_weights) > 0:
            new_embedding = torch.nn.Embedding(next_new_token, current_embeds.weight.shape[1])
            new_embedding.weight[:token_dict_size] = current_embeds.weight[:]
            n = token_dict_size
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
            self.transformer.set_input_embeddings(new_embedding)
        return out_tokens

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        self.transformer.set_input_embeddings(backup_embeds)

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

def load_embed(embedding_name, embedding_directory):
    if isinstance(embedding_directory, str):
        embedding_directory = [embedding_directory]

    valid_file = None
    for embed_dir in embedding_directory:
        embed_path = os.path.join(embed_dir, embedding_name)
        if not os.path.isfile(embed_path):
            extensions = ['.safetensors', '.pt', '.bin']
            for x in extensions:
                t = embed_path + x
                if os.path.isfile(t):
                    valid_file = t
                    break
        else:
            valid_file = embed_path
        if valid_file is not None:
            break

    if valid_file is None:
        return None

    embed_path = valid_file

    if embed_path.lower().endswith(".safetensors"):
        import safetensors.torch
        embed = safetensors.torch.load_file(embed_path, device="cpu")
    else:
        if 'weights_only' in torch.load.__code__.co_varnames:
            embed = torch.load(embed_path, weights_only=True, map_location="cpu")
        else:
            embed = torch.load(embed_path, map_location="cpu")
    if 'string_to_param' in embed:
        values = embed['string_to_param'].values()
    else:
        values = embed.values()
    return next(iter(values))

class SD1Tokenizer:
    def __init__(self, tokenizer_path=None, max_length=77, pad_with_end=True, embedding_directory=None):
        if tokenizer_path is None:
            tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_tokenizer")
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.max_tokens_per_section = self.max_length - 2

        empty = self.tokenizer('')["input_ids"]
        self.start_token = empty[0]
        self.end_token = empty[1]
        self.pad_with_end = pad_with_end
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8

    def tokenize_with_weights(self, text):
        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        tokens = []
        for t in parsed_weights:
            to_tokenize = unescape_important(t[0]).replace("\n", " ").split(' ')
            while len(to_tokenize) > 0:
                word = to_tokenize.pop(0)
                temp_tokens = []
                embedding_identifier = "embedding:"
                if word.startswith(embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(embedding_identifier):].strip('\n')
                    embed = load_embed(embedding_name, self.embedding_directory)
                    if embed is None:
                        stripped = embedding_name.strip(',')
                        if len(stripped) < len(embedding_name):
                            embed = load_embed(stripped, self.embedding_directory)
                            if embed is not None:
                                to_tokenize.insert(0, embedding_name[len(stripped):])

                    if embed is not None:
                        if len(embed.shape) == 1:
                            temp_tokens += [(embed, t[1])]
                        else:
                            for x in range(embed.shape[0]):
                                temp_tokens += [(embed[x], t[1])]
                    else:
                        print("warning, embedding:{} does not exist, ignoring".format(embedding_name))
                elif len(word) > 0:
                    tt = self.tokenizer(word)["input_ids"][1:-1]
                    for x in tt:
                        temp_tokens += [(x, t[1])]
                tokens_left = self.max_tokens_per_section - (len(tokens) % self.max_tokens_per_section)

                #try not to split words in different sections
                if tokens_left < len(temp_tokens) and len(temp_tokens) < (self.max_word_length):
                    for x in range(tokens_left):
                        tokens += [(self.end_token, 1.0)]
                tokens += temp_tokens

        out_tokens = []
        for x in range(0, len(tokens), self.max_tokens_per_section):
            o_token = [(self.start_token, 1.0)] + tokens[x:min(self.max_tokens_per_section + x, len(tokens))]
            o_token += [(self.end_token, 1.0)]
            if self.pad_with_end:
                o_token +=[(self.end_token, 1.0)] * (self.max_length - len(o_token))
            else:
                o_token +=[(0, 1.0)] * (self.max_length - len(o_token))

            out_tokens += [o_token]

        return out_tokens

    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))

import os

from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig, modeling_utils
import comfy.ops
import torch
import traceback
import zipfile
from . import model_management
import contextlib

class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        to_encode = list(self.empty_tokens)
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            to_encode.append(tokens)

        out, pooled = self.encode(to_encode)
        z_empty = out[0:1]
        if pooled.shape[0] > 1:
            first_pooled = pooled[1:2]
        else:
            first_pooled = pooled[0:1]

        output = []
        for k in range(1, out.shape[0]):
            z = out[k:k+1]
            for i in range(len(z)):
                for j in range(len(z[i])):
                    weight = token_weight_pairs[k - 1][j][1]
                    z[i][j] = (z[i][j] - z_empty[0][j]) * weight + z_empty[0][j]
            output.append(z)

        if (len(output) == 0):
            return z_empty.cpu(), first_pooled.cpu()
        return torch.cat(output, dim=-2).cpu(), first_pooled.cpu()

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
        self.num_layers = 12
        if textmodel_path is not None:
            self.transformer = CLIPTextModel.from_pretrained(textmodel_path)
        else:
            if textmodel_json_config is None:
                textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_clip_config.json")
            config = CLIPTextConfig.from_json_file(textmodel_json_config)
            self.num_layers = config.num_hidden_layers
            with comfy.ops.use_comfy_ops():
                with modeling_utils.no_init_weights():
                    self.transformer = CLIPTextModel(config)

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.empty_tokens = [[49406] + [49407] * 76]
        self.text_projection = None
        self.layer_norm_hidden_state = True
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) <= self.num_layers
            self.clip_layer(layer_idx)
        self.layer_default = (self.layer, self.layer_idx)

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def clip_layer(self, layer_idx):
        if abs(layer_idx) >= self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_layer(self):
        self.layer = self.layer_default[0]
        self.layer_idx = self.layer_default[1]

    def set_up_textual_embeddings(self, tokens, current_embeds):
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0] - 1
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, int):
                    if y == token_dict_size: #EOS token
                        y = -1
                    tokens_temp += [y]
                else:
                    if y.shape[0] == current_embeds.weight.shape[1]:
                        embedding_weights += [y]
                        tokens_temp += [next_new_token]
                        next_new_token += 1
                    else:
                        print("WARNING: shape mismatch when trying to apply embedding, embedding will be ignored", y.shape[0], current_embeds.weight.shape[1])
            while len(tokens_temp) < len(x):
                tokens_temp += [self.empty_tokens[0][-1]]
            out_tokens += [tokens_temp]

        n = token_dict_size
        if len(embedding_weights) > 0:
            new_embedding = torch.nn.Embedding(next_new_token + 1, current_embeds.weight.shape[1], device=current_embeds.weight.device, dtype=current_embeds.weight.dtype)
            new_embedding.weight[:token_dict_size] = current_embeds.weight[:-1]
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
            new_embedding.weight[n] = current_embeds.weight[-1] #EOS embedding
            self.transformer.set_input_embeddings(new_embedding)

        processed_tokens = []
        for x in out_tokens:
            processed_tokens += [list(map(lambda a: n if a == -1 else a, x))] #The EOS token should always be the largest one

        return processed_tokens

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(device)

        if backup_embeds.weight.dtype != torch.float32:
            precision_scope = torch.autocast
        else:
            precision_scope = contextlib.nullcontext

        with precision_scope(model_management.get_autocast_device(device)):
            outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
            self.transformer.set_input_embeddings(backup_embeds)

            if self.layer == "last":
                z = outputs.last_hidden_state
            elif self.layer == "pooled":
                z = outputs.pooler_output[:, None, :]
            else:
                z = outputs.hidden_states[self.layer_idx]
                if self.layer_norm_hidden_state:
                    z = self.transformer.text_model.final_layer_norm(z)

            pooled_output = outputs.pooler_output
            if self.text_projection is not None:
                pooled_output = pooled_output.to(self.text_projection.device) @ self.text_projection
        return z.float(), pooled_output.float()

    def encode(self, tokens):
        return self(tokens)

    def load_sd(self, sd):
        return self.transformer.load_state_dict(sd, strict=False)

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

def safe_load_embed_zip(embed_path):
    with zipfile.ZipFile(embed_path) as myzip:
        names = list(filter(lambda a: "data/" in a, myzip.namelist()))
        names.reverse()
        for n in names:
            with myzip.open(n) as myfile:
                data = myfile.read()
                number = len(data) // 4
                length_embed = 1024 #sd2.x
                if number < 768:
                    continue
                if number % 768 == 0:
                    length_embed = 768 #sd1.x
                num_embeds = number // length_embed
                embed = torch.frombuffer(data, dtype=torch.float)
                out = embed.reshape((num_embeds, length_embed)).clone()
                del embed
                return out

def expand_directory_list(directories):
    dirs = set()
    for x in directories:
        dirs.add(x)
        for root, subdir, file in os.walk(x, followlinks=True):
            dirs.add(root)
    return list(dirs)

def load_embed(embedding_name, embedding_directory, embedding_size, embed_key=None):
    if isinstance(embedding_directory, str):
        embedding_directory = [embedding_directory]

    embedding_directory = expand_directory_list(embedding_directory)

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

    embed_out = None

    try:
        if embed_path.lower().endswith(".safetensors"):
            import safetensors.torch
            embed = safetensors.torch.load_file(embed_path, device="cpu")
        else:
            if 'weights_only' in torch.load.__code__.co_varnames:
                try:
                    embed = torch.load(embed_path, weights_only=True, map_location="cpu")
                except:
                    embed_out = safe_load_embed_zip(embed_path)
            else:
                embed = torch.load(embed_path, map_location="cpu")
    except Exception as e:
        print(traceback.format_exc())
        print()
        print("error loading embedding, skipping loading:", embedding_name)
        return None

    if embed_out is None:
        if 'string_to_param' in embed:
            values = embed['string_to_param'].values()
            embed_out = next(iter(values))
        elif isinstance(embed, list):
            out_list = []
            for x in range(len(embed)):
                for k in embed[x]:
                    t = embed[x][k]
                    if t.shape[-1] != embedding_size:
                        continue
                    out_list.append(t.reshape(-1, t.shape[-1]))
            embed_out = torch.cat(out_list, dim=0)
        elif embed_key is not None and embed_key in embed:
            embed_out = embed[embed_key]
        else:
            values = embed.values()
            embed_out = next(iter(values))
    return embed_out

class SD1Tokenizer:
    def __init__(self, tokenizer_path=None, max_length=77, pad_with_end=True, embedding_directory=None, embedding_size=768, embedding_key='clip_l'):
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
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

    def _try_get_embedding(self, embedding_name:str):
        '''
        Takes a potential embedding name and tries to retrieve it.
        Returns a Tuple consisting of the embedding and any leftover string, embedding can be None.
        '''
        embed = load_embed(embedding_name, self.embedding_directory, self.embedding_size, self.embedding_key)
        if embed is None:
            stripped = embedding_name.strip(',')
            if len(stripped) < len(embedding_name):
                embed = load_embed(stripped, self.embedding_directory, self.embedding_size, self.embedding_key)
                return (embed, embedding_name[len(stripped):])
        return (embed, "")


    def tokenize_with_weights(self, text:str, return_word_ids=False):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        '''
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        #tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment).replace("\n", " ").split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                #if we find an embedding, deal with the embedding
                if word.startswith(self.embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        print(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                    #if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                #parse word
                tokens.append([(t, weight) for t in self.tokenizer(word)["input_ids"][1:-1]])

        #reshape token array to CLIP input size
        batched_tokens = []
        batch = [(self.start_token, 1.0, 0)]
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            #determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    #break word in two and add end token
                    if is_large:
                        batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                        batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    #add end token and pad
                    else:
                        batch.append((self.end_token, 1.0, 0))
                        batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                    #start new batch
                    batch = [(self.start_token, 1.0, 0)]
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t,w,i+1) for t,w in t_group])
                    t_group = []

        #fill last batch
        batch.extend([(self.end_token, 1.0, 0)] + [(pad_token, 1.0, 0)] * (self.max_length - len(batch) - 1))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w,_ in x] for x in batched_tokens]

        return batched_tokens


    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))

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
                    if y.shape[0] == current_embeds.weight.shape[1]:
                        embedding_weights += [y]
                        tokens_temp += [next_new_token]
                        next_new_token += 1
                    else:
                        print("WARNING: shape mismatch when trying to apply embedding, embedding will be ignored", y.shape[0], current_embeds.weight.shape[1])
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
        self.embedding_identifier = "embedding:"

    def _try_get_embedding(self, name:str):
        '''
        Takes a potential embedding name and tries to retrieve it.
        Returns a Tuple consisting of the embedding and any leftover string, embedding can be None.
        '''
        embedding_name = name[len(self.embedding_identifier):].strip('\n')
        embed = load_embed(embedding_name, self.embedding_directory)
        if embed is None:
            stripped = embedding_name.strip(',')
            if len(stripped) < len(embedding_name):
                embed = load_embed(stripped, self.embedding_directory)
                return (embed, embedding_name[len(stripped):])
        return (embed, "")


    def tokenize_with_weights(self, text:str):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        '''
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
                    embed, leftover = self._try_get_embedding(word)
                    if embed is None:
                        print(f"warning, embedding:{word} does not exist, ignoring")
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
        batch = []
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            #start a new batch if there is not enough room
            if len(t_group) + len(batch) > self.max_tokens_per_section:
                remaining_length = self.max_tokens_per_section - len(batch)
                #fill remaining space depending on length of tokens
                if len(t_group) > self.max_word_length:
                    #put part of group of tokens in the batch
                    batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                    t_group = t_group[remaining_length:]
                else:
                    #filler tokens
                    batch.extend([(self.end_token, 1.0, 0)] * remaining_length)
                batch = []
                batched_tokens.append(batch)
            #put current group of tokens in the batch
            batch.extend([(t,w,i+1) for t,w in t_group])
        
        #fill last batch
        batch.extend([(self.end_token, 1.0, 0)] * (self.max_tokens_per_section - len(batch)))
        
        #add start and end tokens
        batched_tokens = [[(self.start_token, 1.0, 0)] + x + [(self.end_token, 1.0, 0)] for x in batched_tokens]
        return batched_tokens


    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))

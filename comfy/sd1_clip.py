import os

from transformers import CLIPTokenizer
import comfy.ops
import torch
import traceback
import zipfile
from . import model_management
import comfy.clip_model
import json
import logging
import numbers
import re

def gen_empty_tokens(special_tokens, length):
    start_token = special_tokens.get("start", None)
    end_token = special_tokens.get("end", None)
    pad_token = special_tokens.get("pad")
    output = []
    if start_token is not None:
        output.append(start_token)
    if end_token is not None:
        output.append(end_token)
    output += [pad_token] * (length - len(output))
    return output

class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            if hasattr(self, "gen_empty_tokens"):
                to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))
            else:
                to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

        o = self.encode(to_encode)
        out, pooled = o[:2]

        if pooled is not None:
            first_pooled = pooled[0:1].to(model_management.intermediate_device())
        else:
            first_pooled = pooled

        output = []
        for k in range(0, sections):
            z = out[k:k+1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if (len(output) == 0):
            r = (out[-1:].to(model_management.intermediate_device()), first_pooled)
        else:
            r = (torch.cat(output, dim=-2).to(model_management.intermediate_device()), first_pooled)

        if len(o) > 2:
            extra = {}
            for k in o[2]:
                v = o[2][k]
                if k == "attention_mask":
                    v = v[:sections].flatten().unsqueeze(dim=0).to(model_management.intermediate_device())
                extra[k] = v

            r = r + (extra,)
        return r

class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, device="cpu", max_length=77,
                 freeze=True, layer="last", layer_idx=None, textmodel_json_config=None, dtype=None, model_class=comfy.clip_model.CLIPTextModel,
                 special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=True, enable_attention_masks=False, zero_out_masked=False,
                 return_projected_pooled=True, return_attention_masks=False, model_options={}):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS

        if textmodel_json_config is None:
            textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_clip_config.json")

        if isinstance(textmodel_json_config, dict):
            config = textmodel_json_config
        else:
            with open(textmodel_json_config) as f:
                config = json.load(f)

        operations = model_options.get("custom_operations", None)
        scaled_fp8 = None

        if operations is None:
            scaled_fp8 = model_options.get("scaled_fp8", None)
            if scaled_fp8 is not None:
                operations = comfy.ops.scaled_fp8_ops(fp8_matrix_mult=False, override_dtype=scaled_fp8)
            else:
                operations = comfy.ops.manual_cast

        self.operations = operations
        self.transformer = model_class(config, dtype, device, self.operations)
        if scaled_fp8 is not None:
            self.transformer.scaled_fp8 = torch.nn.Parameter(torch.tensor([], dtype=scaled_fp8))

        self.num_layers = self.transformer.num_layers

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens

        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = enable_attention_masks
        self.zero_out_masked = zero_out_masked

        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        self.return_attention_masks = return_attention_masks

        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_options(self):
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]
        self.return_projected_pooled = self.options_default[2]

    def set_up_textual_embeddings(self, tokens, current_embeds):
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0]
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, numbers.Integral):
                    tokens_temp += [int(y)]
                else:
                    if y.shape[0] == current_embeds.weight.shape[1]:
                        embedding_weights += [y]
                        tokens_temp += [next_new_token]
                        next_new_token += 1
                    else:
                        logging.warning("WARNING: shape mismatch when trying to apply embedding, embedding will be ignored {} != {}".format(y.shape[0], current_embeds.weight.shape[1]))
            while len(tokens_temp) < len(x):
                tokens_temp += [self.special_tokens["pad"]]
            out_tokens += [tokens_temp]

        n = token_dict_size
        if len(embedding_weights) > 0:
            new_embedding = self.operations.Embedding(next_new_token + 1, current_embeds.weight.shape[1], device=current_embeds.weight.device, dtype=current_embeds.weight.dtype)
            new_embedding.weight[:token_dict_size] = current_embeds.weight
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
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

        attention_mask = None
        if self.enable_attention_masks or self.zero_out_masked or self.return_attention_masks:
            attention_mask = torch.zeros_like(tokens)
            end_token = self.special_tokens.get("end", None)
            if end_token is None:
                cmp_token = self.special_tokens.get("pad", -1)
            else:
                cmp_token = end_token

            for x in range(attention_mask.shape[0]):
                for y in range(attention_mask.shape[1]):
                    attention_mask[x, y] = 1
                    if tokens[x, y] == cmp_token:
                        if end_token is None:
                            attention_mask[x, y] = 0
                        break

        attention_mask_model = None
        if self.enable_attention_masks:
            attention_mask_model = attention_mask

        outputs = self.transformer(tokens, attention_mask_model, intermediate_output=self.layer_idx, final_layer_norm_intermediate=self.layer_norm_hidden_state, dtype=torch.float32)
        self.transformer.set_input_embeddings(backup_embeds)

        if self.layer == "last":
            z = outputs[0].float()
        else:
            z = outputs[1].float()

        if self.zero_out_masked:
            z *= attention_mask.unsqueeze(-1).float()

        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        extra = {}
        if self.return_attention_masks:
            extra["attention_mask"] = attention_mask

        if len(extra) > 0:
            return z, pooled_output, extra

        return z, pooled_output

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

def bundled_embed(embed, prefix, suffix): #bundled embedding in lora format
    out_list = []
    for k in embed:
        if k.startswith(prefix) and k.endswith(suffix):
            out_list.append(embed[k])
    if len(out_list) == 0:
        return None

    return torch.cat(out_list, dim=0)

def load_embed(embedding_name, embedding_directory, embedding_size, embed_key=None):
    if isinstance(embedding_directory, str):
        embedding_directory = [embedding_directory]

    embedding_directory = expand_directory_list(embedding_directory)

    valid_file = None
    for embed_dir in embedding_directory:
        embed_path = os.path.abspath(os.path.join(embed_dir, embedding_name))
        embed_dir = os.path.abspath(embed_dir)
        try:
            if os.path.commonpath((embed_dir, embed_path)) != embed_dir:
                continue
        except:
            continue
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
            try:
                embed = torch.load(embed_path, weights_only=True, map_location="cpu")
            except:
                embed_out = safe_load_embed_zip(embed_path)
    except Exception:
        logging.warning("{}\n\nerror loading embedding, skipping loading: {}".format(traceback.format_exc(), embedding_name))
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
            embed_out = bundled_embed(embed, 'bundle_emb.', '.string_to_param.*')
            if embed_out is None:
                embed_out = bundled_embed(embed, 'bundle_emb.', '.{}'.format(embed_key))
            if embed_out is None:
                values = embed.values()
                embed_out = next(iter(values))
    return embed_out

class SDTokenizer:
    def __init__(self, tokenizer_path=None, max_length=77, pad_with_end=True, embedding_directory=None, embedding_size=768, embedding_key='clip_l', tokenizer_class=CLIPTokenizer, has_start_token=True, has_end_token=True, pad_to_max_length=True, min_length=None, pad_token=None, end_token=None, tokenizer_data={}, tokenizer_args={}):
        if tokenizer_path is None:
            tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_tokenizer")
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path, **tokenizer_args)
        self.max_length = max_length
        self.min_length = min_length
        self.end_token = None

        empty = self.tokenizer('')["input_ids"]
        self.tokenizer_adds_end_token = has_end_token
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            if end_token is not None:
                self.end_token = end_token
            else:
                if has_end_token:
                    self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            if end_token is not None:
                self.end_token = end_token
            else:
                self.end_token = empty[0]

        if pad_token is not None:
            self.pad_token = pad_token
        elif pad_with_end:
            self.pad_token = self.end_token
        else:
            self.pad_token = 0

        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length

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
        split_embed = embedding_name.split()
        embedding_name = split_embed[0]
        leftover = ' '.join(split_embed[1:])
        embed = load_embed(embedding_name, self.embedding_directory, self.embedding_size, self.embedding_key)
        if embed is None:
            stripped = embedding_name.strip(',')
            if len(stripped) < len(embedding_name):
                embed = load_embed(stripped, self.embedding_directory, self.embedding_size, self.embedding_key)
                return (embed, "{} {}".format(embedding_name[len(stripped):], leftover))
        return (embed, leftover)


    def tokenize_with_weights(self, text:str, return_word_ids=False):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        '''

        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        # tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment)
            split = re.split(' {0}|\n{0}'.format(self.embedding_identifier), to_tokenize)
            to_tokenize = [split[0]]
            for i in range(1, len(split)):
                to_tokenize.append("{}{}".format(self.embedding_identifier, split[i]))

            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                # if we find an embedding, deal with the embedding
                if word.startswith(self.embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        logging.warning(f"warning, embedding:{embedding_name} does not exist, ignoring")
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
                end = 999999999999
                if self.tokenizer_adds_end_token:
                    end = -1
                #parse word
                tokens.append([(t, weight) for t in self.tokenizer(word)["input_ids"][self.tokens_start:end]])

        #reshape token array to CLIP input size
        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            #determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length
            if self.end_token is not None:
                has_end_token = 1
            else:
                has_end_token = 0

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - has_end_token:
                    remaining_length = self.max_length - len(batch) - has_end_token
                    #break word in two and add end token
                    if is_large:
                        batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                        if self.end_token is not None:
                            batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    #add end token and pad
                    else:
                        if self.end_token is not None:
                            batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            batch.extend([(self.pad_token, 1.0, 0)] * (remaining_length))
                    #start new batch
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t,w,i+1) for t,w in t_group])
                    t_group = []

        #fill last batch
        if self.end_token is not None:
            batch.append((self.end_token, 1.0, 0))
        if self.pad_to_max_length:
            batch.extend([(self.pad_token, 1.0, 0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(self.pad_token, 1.0, 0)] * (self.min_length - len(batch)))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w,_ in x] for x in batched_tokens]

        return batched_tokens


    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))

    def state_dict(self):
        return {}

class SD1Tokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}, clip_name="l", tokenizer=SDTokenizer, name=None):
        if name is not None:
            self.clip_name = name
            self.clip = "{}".format(self.clip_name)
        else:
            self.clip_name = clip_name
            self.clip = "clip_{}".format(self.clip_name)

        tokenizer = tokenizer_data.get("{}_tokenizer_class".format(self.clip), tokenizer)
        setattr(self, self.clip, tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data))

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        out = {}
        out[self.clip_name] = getattr(self, self.clip).tokenize_with_weights(text, return_word_ids)
        return out

    def untokenize(self, token_weight_pair):
        return getattr(self, self.clip).untokenize(token_weight_pair)

    def state_dict(self):
        return getattr(self, self.clip).state_dict()

class SD1CheckpointClipModel(SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, return_projected_pooled=False, dtype=dtype, model_options=model_options)

class SD1ClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}, clip_name="l", clip_model=SD1CheckpointClipModel, name=None, **kwargs):
        super().__init__()

        if name is not None:
            self.clip_name = name
            self.clip = "{}".format(self.clip_name)
        else:
            self.clip_name = clip_name
            self.clip = "clip_{}".format(self.clip_name)

        clip_model = model_options.get("{}_class".format(self.clip), clip_model)
        setattr(self, self.clip, clip_model(device=device, dtype=dtype, model_options=model_options, **kwargs))

        self.dtypes = set()
        if dtype is not None:
            self.dtypes.add(dtype)

    def set_clip_options(self, options):
        getattr(self, self.clip).set_clip_options(options)

    def reset_clip_options(self):
        getattr(self, self.clip).reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs = token_weight_pairs[self.clip_name]
        out = getattr(self, self.clip).encode_token_weights(token_weight_pairs)
        return out

    def load_sd(self, sd):
        return getattr(self, self.clip).load_sd(sd)

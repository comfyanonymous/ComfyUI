from typing import (
    Dict, List, Optional, Union, Tuple, MutableMapping, Any, Mapping, Collection, get_type_hints, get_args, get_origin
)

from dataclasses import dataclass, fields, _FIELDS, _FIELD, _FIELD_INITVAR, is_dataclass
import torch.nn.functional as F
import numpy as np
import torch
import math
from functools import lru_cache

__MAX_SIZE = 2048

def _ceil_to_nearest(n, round_to):
    return (n + round_to - 1) // round_to * round_to

@torch.jit.script
def build_delay_pattern_mask(
    input_ids: torch.Tensor,
    bos_token_id: int,
    pad_token_id: int,
):
    bsz, num_codebooks, seq_len = input_ids.shape

    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones((bsz, num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device)
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    input_ids = input_ids_with_gen_mask.clone()
    input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return input_ids, input_ids_with_gen_mask


# implementation of dacite's from_dict with only necessary parts for ChatML
@lru_cache(maxsize=None)
def cache(function):
    return lru_cache(maxsize=__MAX_SIZE, typed=True)(function)

@cache
def get_fields(data_class):
    fields = getattr(data_class, _FIELDS)
    return [f for f in fields.values() if f._field_type is _FIELD or f._field_type is _FIELD_INITVAR]

def is_optional(type_) -> bool:
    return get_origin(type_) is Union and type(None) in get_args(type_)

def orig(data_class) -> Any:
    if is_dataclass(data_class):
        return data_class
    return get_origin(data_class)

@cache
def extract_generic(type_, defaults: Tuple = ()) -> tuple:
    try:
        if getattr(type_, "_special", False):
            return defaults
        if type_.__args__ == ():
            return (type_.__args__,)
        return type_.__args__ or defaults  # type: ignore
    except AttributeError:
        return defaults

def _build_value_for_collection(collection, data: Any) -> Any:
    if isinstance(data, Mapping):
        value_type = extract_generic(collection, defaults=(Any, Any))[1]
        return {
            key: _build_value(type_=value_type, data=value)
            for key, value in data.items()
        }

    elif isinstance(data, Collection) and not isinstance(data, (str, bytes, Mapping)):
        item_type = extract_generic(collection, defaults=(Any,))[0]
        return [
            _build_value(type_=item_type, data=item)
            for item in data
        ]

    return data

def _build_value(type_, data) -> Any:
    if is_optional(type_) and data is None:
        return data
    if get_origin(type_) is Union:
        data = _build_value_for_union(union=type_, data=data)
    elif hasattr(type_, "__origin__"):
        data = _build_value_for_collection(collection=type_, data=data)
    elif cache(is_dataclass)(orig(type_)) and isinstance(data, Mapping):
        data = from_dict(data_class=type_, data=data)
    return data

def _build_value_for_union(union: type, data: Any) -> Any:
    for inner_type in get_args(union):
        if data is None and inner_type is type(None):
            return None
        try:
            return _build_value(inner_type, data)
        except Exception:
            continue
    raise ValueError(f"Cannot match {data!r} to any type in {union}")

def is_instance(value: Any, type_) -> bool:
    if type_ is Any:
        return True

    origin = get_origin(type_)
    args = get_args(type_)

    if origin is Union:
        return any(is_instance(value, arg) for arg in args)

    if origin in (list, List):
        if not isinstance(value, list):
            return False
        (elem_type,) = args or (Any,)
        return all(is_instance(item, elem_type) for item in value)

    if origin in (dict, Dict, Mapping):
        if not isinstance(value, dict):
            return False
        key_type, val_type = args or (Any, Any)
        return all(
            is_instance(k, key_type) and is_instance(v, val_type)
            for k, v in value.items()
        )

    try:
        return isinstance(value, type_)
    except TypeError:
        return False

def from_dict(data_class, data):

    init_values: MutableMapping[str, Any] = {}
    post_init_values: MutableMapping[str, Any] = {}

    data_class_hints = get_type_hints(data_class)
    data_class_fields = cache(get_fields)(data_class)

    extra_fields = set(data.keys()) - {f.name for f in data_class_fields}
    if extra_fields:
        formatted_keys = ", ".join(f'"{key}"' for key in extra_fields)
        raise ValueError(f"cannot match {formatted_keys} to any data class field")

    for field in data_class_fields:
        field_type = data_class_hints[field.name]
        key = field.name

        if key in data:
            try:
                value = _build_value(type_=field_type, data=data[key])
            except Exception as error:
                raise ValueError(error)

            if not is_instance(value, field_type):
                raise ValueError((
                    f'wrong value type for field "{field.name}" - should be "{field_type}" '
                    f'instead of value "{value}" of type "{type(value)}"'
                ))

            init_values[field.name] = value

    instance = data_class(**init_values)

    for key, value in post_init_values.items():
        setattr(instance, key, value)

    return instance

def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    # Mapping of Chinese punctuation to English punctuation
    chinese_to_english_punct = {
        "，": ", ",  # comma
        "。": ".",  # period
        "：": ":",  # colon
        "；": ";",  # semicolon
        "？": "?",  # question mark
        "！": "!",  # exclamation mark
        "（": "(",  # left parenthesis
        "）": ")",  # right parenthesis
        "【": "[",  # left square bracket
        "】": "]",  # right square bracket
        "《": "<",  # left angle quote
        "》": ">",  # right angle quote
        "“": '"',  # left double quotation
        "”": '"',  # right double quotation
        "‘": "'",  # left single quotation
        "’": "'",  # right single quotation
        "、": ",",  # enumeration comma
        "—": "-",  # em dash
        "…": "...",  # ellipsis
        "·": ".",  # middle dot
        "「": '"',  # left corner bracket
        "」": '"',  # right corner bracket
        "『": '"',  # left double corner bracket
        "』": '"',  # right double corner bracket
    }

    # Replace each Chinese punctuation with its English counterpart
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)

    return text

def transcript_normalize(text: str):
    transcript = normalize_chinese_punctuation(text)

    transcript = transcript.replace("(", " ")
    transcript = transcript.replace(")", " ")
    transcript = transcript.replace("°F", " degrees Fahrenheit")
    transcript = transcript.replace("°C", " degrees Celsius")

    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE_s>[Humming]</SE_s>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        transcript = transcript.replace(tag, replacement)
    lines = transcript.split("\n")
    transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    transcript = transcript.strip()

    if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        transcript += "."

    return transcript

@dataclass
class AudioContent:
    audio_url: str
    raw_audio: Optional[str] = None
    offset: Optional[float] = None
    duration: Optional[float] = None
    row_id: Optional[int] = None
    type: str = "audio"


@dataclass
class TextContent:
    text: str
    type: str = "text"


@dataclass
class Message:
    role: str
    content: Union[str, AudioContent, TextContent, List[Union[str, AudioContent, TextContent]]]
    recipient: Optional[str] = None


@dataclass
class ChatMLSample:
    messages: List[Message]
    start_index: Optional[int] = None
    misc: Optional[Dict] = None
    speaker: Optional[str] = None

def prepare_chatml_sample(sample: Union[ChatMLSample, Dict], tokenizer):

    try:
        if not isinstance(sample, ChatMLSample):

            # replacing pd.isna
            def is_nan(x):
                if isinstance(x, float):
                    return math.isnan(x)
                if isinstance(x, np.generic):
                    return np.isnan(x)
                if isinstance(x, torch.Tensor) and x.numel() == 1:
                    return torch.isnan(x).item()
                return False

            if "speaker" in sample and is_nan(sample["speaker"]):
                sample["speaker"] = None
            if "start_index" in sample and is_nan(sample["start_index"]):
                sample["start_index"] = None
            if "content" in sample and is_nan(sample["content"]):
                sample["content"] = ""

            def convert_nan_to_none(obj):

                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, float) and math.isnan(obj):
                    return None
                elif isinstance(obj, dict):
                    return {k: convert_nan_to_none(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_nan_to_none(item) for item in obj]
                return obj

            clean_sample = convert_nan_to_none(sample)

            val_keys = []
            for field in fields(ChatMLSample):
                if field.name in clean_sample:
                    val_keys.append(field.name)
            clean_sample = {k: clean_sample[k] for k in val_keys}

            sample = from_dict(
                data_class=ChatMLSample, data=clean_sample,
            )

        input_tokens = []
        audio_contents = []
        speaker_id = None
        if sample.speaker is not None:
            speaker_id = sample.speaker
        elif sample.misc is not None:
            if "speaker" in sample.misc:
                speaker_id = sample.misc["speaker"]

        total_m = len(sample.messages)
        for turn_id, message in enumerate(sample.messages):
            role = message.role
            recipient = message.recipient
            content = message.content
            content_l = []

            if isinstance(content, str):
                content_l.append(TextContent(text=content))
            elif isinstance(content, TextContent):
                content_l.append(content)
            elif isinstance(content, AudioContent):
                content_l.append(content)
            elif isinstance(content, list):
                for ele in content:
                    if isinstance(ele, str):
                        content_l.append(TextContent(text=ele))
                    else:
                        content_l.append(ele)
            if turn_id == 0:
                prefix = f"<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n"
            else:
                prefix = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            eot_postfix = "<|eot_id|>"
            eom_postfix = "<|eom_id|>"

            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            input_tokens.extend(prefix_tokens)

            if recipient:
                assert role == "assistant", "Recipient is only available for assistant role."
                recipient_tokens = tokenizer.encode(f"{recipient}<|recipient|>", add_special_tokens=False)
                input_tokens.extend(recipient_tokens)

            for content in content_l:
                if content.type == "text":
                    text_tokens = tokenizer.encode(content.text, add_special_tokens=False)
                    input_tokens.extend(text_tokens)

                elif content.type == "audio":
                    audio_contents.append(content)
                    if role == "user" or role == "system":
                        text_tokens = tokenizer.encode(
                            f"<|audio_bos|><|AUDIO|><|audio_eos|>",
                            add_special_tokens=False,
                        )
                        input_tokens.extend(text_tokens)
                    elif role == "assistant":
                        text_tokens = tokenizer.encode(
                            f"<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>",
                            add_special_tokens=False,
                        )
                        input_tokens.extend(text_tokens)

            next_id = turn_id + 1
            if role == "assistant" and next_id != total_m and sample.messages[next_id].role == "assistant":
                postfix_tokens = tokenizer.encode(eom_postfix, add_special_tokens=False)
                input_tokens.extend(postfix_tokens)
            else:
                postfix_tokens = tokenizer.encode(eot_postfix, add_special_tokens=False)
                input_tokens.extend(postfix_tokens)

        return input_tokens, audio_contents, speaker_id

    except Exception as e:
        import json
        print(f"Error in prepare_chatml_sample: {str(e)}")
        print(f"Sample data: {json.dumps(sample, indent=2)}")
        return None, None, None

@dataclass
class HiggsAudioBatchInput:
    input_ids: torch.LongTensor  # shape (bsz, seq_len).
    attention_mask: torch.Tensor  # shape (bsz, seq_len).
    audio_out_ids: Optional[torch.LongTensor]  # shape (num_codebooks, audio_out_total_length)
    audio_out_ids_start: Optional[torch.LongTensor]  # shape (num_audio_out,)
    audio_out_ids_start_group_loc: Optional[torch.LongTensor]  # shape (num_audio_out,), specify which a sample's group location in the batch
    audio_in_ids: Optional[torch.LongTensor]  # shape (num_codebooks, audio_in_total_length)
    audio_in_ids_start: Optional[torch.LongTensor]  # shape (num_audio_in,)
    label_ids: Optional[torch.LongTensor]  # shape (bsz, seq_len)
    label_audio_ids: Optional[torch.LongTensor]  # shape (num_codebooks, audio_out_total_length)
    reward: Optional[float] = None

@dataclass
class ChatMLDatasetSample:

    input_ids: torch.LongTensor  # (seq_len,) Input text tokens
    label_ids: torch.LongTensor  # (seq_len,) Label IDs
    audio_ids_concat: torch.LongTensor  # (num_codebooks, audio_seq_len) Concatenated audio tokens
    audio_ids_start: torch.LongTensor  # (num_audios,) Start index of each audio token in `audio_ids_concat`
    audio_waveforms_concat: torch.Tensor  # (total_wv_length,) Concatenated audio waveforms
    audio_waveforms_start: torch.LongTensor  # (num_audios,) Start index of each waveform in `audio_waveforms_concat`
    audio_sample_rate: torch.Tensor  # (num_audios,) Sampling rate per audio waveform
    audio_speaker_indices: torch.LongTensor  # (num_audios,) Speaker indices per audio; -1 = unknown
    audio_label_ids_concat: Optional[torch.LongTensor] = None  # (num_codebooks, audio_seq_len) Optional audio token labels
    reward: Optional[float] = None  # Optional scalar reward


    def num_audios(self):
        return max(len(self.audio_waveforms_start), len(self.audio_ids_start))

    def get_audio_codes(self, idx):
        code_start = self.audio_ids_start[idx]
        if idx < len(self.audio_ids_start) - 1:
            code_end = self.audio_ids_start[idx + 1]
        else:
            code_end = self.audio_ids_concat.shape[-1]

        return self.audio_ids_concat[:, code_start:code_end]

    def get_audio_codes_labels(self, idx):
        if self.audio_label_ids_concat is None:
            return None
        code_start = self.audio_ids_start[idx]
        if idx < len(self.audio_ids_start) - 1:
            code_end = self.audio_ids_start[idx + 1]
        else:
            code_end = self.audio_ids_concat.shape[-1]

        return self.audio_label_ids_concat[:, code_start:code_end]

    def get_wv(self, idx):
        wv_start = self.audio_waveforms_start[idx]
        sr = self.audio_sample_rate[idx]
        if idx < len(self.audio_waveforms_start) - 1:
            wv_end = self.audio_waveforms_start[idx + 1]
        else:
            wv_end = self.audio_waveforms_concat.shape[-1]
        return self.audio_waveforms_concat[wv_start:wv_end], sr

class HiggsAudioSampleCollator:

    def __init__(
        self,
        audio_in_token_id,
        audio_out_token_id,
        pad_token_id,
        audio_stream_bos_id,
        audio_stream_eos_id,
        round_to=8,
        pad_left=False,
        return_audio_in_tokens=True,
        audio_num_codebooks=None,
        use_delay_pattern=False,
        disable_audio_codes_transform=False,
        add_new_bos_eos_for_long_chunk=True,
        mask_audio_out_token_label=True,
    ):
        self.round_to = round_to
        self.pad_left = pad_left
        self.audio_in_token_id = audio_in_token_id
        self.audio_out_token_id = audio_out_token_id
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.pad_token_id = pad_token_id
        self.return_audio_in_tokens = return_audio_in_tokens
        self.audio_num_codebooks = audio_num_codebooks
        self.use_delay_pattern = use_delay_pattern

        self.disable_audio_codes_transform = disable_audio_codes_transform
        self.add_new_bos_eos_for_long_chunk = add_new_bos_eos_for_long_chunk
        self.mask_audio_out_token_label = mask_audio_out_token_label

    def _process_and_duplicate_audio_tokens(
        self, input_ids: torch.Tensor, audio_idx: int, wv: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:

        total_samples = len(wv)
        num_chunks = math.ceil(total_samples / self.chunk_size_samples)

        if num_chunks <= 1:
            return input_ids, labels, 1

        audio_token_seq = input_ids[audio_idx - 1 : audio_idx + 2]
        duplicated_sequence = audio_token_seq.repeat(num_chunks)

        new_input_ids = torch.cat([input_ids[: audio_idx - 1], duplicated_sequence, input_ids[audio_idx + 2 :]])

        new_labels = None
        if labels is not None:
            label_seq = labels[audio_idx - 1 : audio_idx + 2]
            duplicated_labels = label_seq.repeat(num_chunks)
            new_labels = torch.cat([labels[: audio_idx - 1], duplicated_labels, labels[audio_idx + 2 :]])

        return new_input_ids, new_labels, num_chunks

    def __call__(self, batch: List[ChatMLDatasetSample]):

        label_ids = None
        label_audio_ids = None
        if all([ele.label_ids is None for ele in batch]):
            return_labels = False
        else:
            return_labels = True

        processed_batch = batch

        # Get the max sequence length based on processed batch
        max_seq_length = _ceil_to_nearest(max([len(sample.input_ids) for sample in processed_batch]), self.round_to)

        # Get the ids for audio-in and audio-out for each batch
        audio_in_ids_l = []
        audio_out_ids_l = []
        audio_out_ids_group_loc_l = []
        audio_in_label_ids_l = None
        audio_out_label_ids_l = None
        reward_l = []

        if return_labels:
            audio_out_no_train_flag = []  # Whether the audio-out data should be trained on or not.

        # Process the audio inputs and outputs
        for i in range(len(processed_batch)):
            audio_in_mask = processed_batch[i].input_ids == self.audio_in_token_id
            audio_out_mask = processed_batch[i].input_ids == self.audio_out_token_id
            audio_ids = torch.ones_like(processed_batch[i].input_ids)
            audio_ids[audio_in_mask ^ audio_out_mask] = torch.cumsum(audio_ids[audio_in_mask ^ audio_out_mask], 0) - 1
            audio_in_ids = audio_ids[audio_in_mask]
            audio_out_ids = audio_ids[audio_out_mask]

            if return_labels:
                audio_out_no_train_flag.append(processed_batch[i].label_ids[audio_out_mask] < 0)
                if self.mask_audio_out_token_label:
                    processed_batch[i].label_ids[audio_out_mask] = -100

            if self.return_audio_in_tokens:
                audio_in_ids_l.extend(
                    [processed_batch[i].get_audio_codes(idx)[: self.audio_num_codebooks, :] for idx in audio_in_ids]
                )
                if processed_batch[i].audio_label_ids_concat is not None:
                    if audio_in_label_ids_l is None:
                        audio_in_label_ids_l = []
                    audio_in_label_ids_l.extend(
                        [
                            processed_batch[i].get_audio_codes_labels(idx)[: self.audio_num_codebooks, :]
                            for idx in audio_in_ids
                        ]
                    )

            audio_out_ids_l.extend(
                [processed_batch[i].get_audio_codes(idx)[: self.audio_num_codebooks, :] for idx in audio_out_ids]
            )
            audio_out_ids_group_loc_l.append(i)
            if processed_batch[i].reward is not None:
                reward_l.append(processed_batch[i].reward)

            if processed_batch[i].audio_label_ids_concat is not None:
                if audio_out_label_ids_l is None:
                    audio_out_label_ids_l = []
                audio_out_label_ids_l.extend(
                    [
                        processed_batch[i].get_audio_codes_labels(idx)[: self.audio_num_codebooks, :]
                        for idx in audio_out_ids
                    ]
                )

        if return_labels:
            audio_out_no_train_flag = torch.cat(audio_out_no_train_flag, dim=0)

        if len(audio_in_ids_l) > 0:

            # I tried to remove the for-loop in original implementation
            # but to do batching with padding caused problem so I turned it into a list compre.
            lengths = [seg.shape[1] for seg in audio_in_ids_l]
            aug_lengths = [l + 2 for l in lengths]
            audio_in_ids_start = torch.cumsum(
                torch.tensor([0] + aug_lengths[:-1], dtype=torch.long), dim=0
            )

            if self.disable_audio_codes_transform:
                audio_in_ids = torch.cat(audio_in_ids_l, dim=1).long()
            else:
                with_tokens = [
                    torch.cat([
                        torch.full((seg.shape[0], 1), self.audio_stream_bos_id, dtype=torch.long),
                        seg,
                        torch.full((seg.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long),
                    ], dim=1)
                    for seg in audio_in_ids_l
                ]

                if self.use_delay_pattern:
                    with_tokens = [
                        build_delay_pattern_mask(
                            tok.unsqueeze(0),
                            bos_token_id=self.audio_stream_bos_id,
                            pad_token_id=self.audio_stream_eos_id
                        )[0]
                        for tok in with_tokens
                    ]
                audio_in_ids = torch.cat(with_tokens, dim=1).long()
        else:
            audio_in_ids = torch.zeros((0, 0), dtype=torch.long)
            audio_in_ids_start = torch.zeros(0, dtype=torch.long)

        audio_out_ids_start_group_loc = None
        if len(audio_out_ids_l) > 0:
            new_audio_out_ids_l = []
            label_audio_ids_l = []
            for idx, ele in enumerate(audio_out_ids_l):
                if self.disable_audio_codes_transform:

                    audio_codes = ele
                    if return_labels:
                        label_audio_ids = audio_out_label_ids_l[idx]
                else:
                    audio_codes = torch.cat(
                        [
                            torch.full((ele.shape[0], 1), self.audio_stream_bos_id, dtype=torch.long),
                            ele,
                            torch.full((ele.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long),
                        ],
                        dim=1,
                    )
                    if return_labels:
                        label_audio_ids = torch.cat(
                            [
                                torch.full((ele.shape[0], 1), -100, dtype=torch.long),
                                ele,
                                torch.full((ele.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long),
                            ],
                            dim=1,
                        )
                    if self.use_delay_pattern:
                        audio_codes = build_delay_pattern_mask(
                            audio_codes.unsqueeze(0),
                            bos_token_id=self.audio_stream_bos_id,
                            pad_token_id=self.audio_stream_eos_id,
                        )[0].squeeze(0)
                        if return_labels:
                            label_audio_ids = build_delay_pattern_mask(
                                label_audio_ids.unsqueeze(0),
                                bos_token_id=-100,
                                pad_token_id=-100,
                            )[0].squeeze(0)
                new_audio_out_ids_l.append(audio_codes)

                if return_labels:
                    if audio_out_no_train_flag[idx]:
                        label_audio_ids[:] = -100
                    label_audio_ids_l.append(label_audio_ids)

            audio_out_ids = torch.cat(new_audio_out_ids_l, dim=1).long()
            if return_labels:
                label_audio_ids = torch.cat(label_audio_ids_l, dim=1).long()
            audio_out_ids_start = torch.cumsum(
                torch.tensor([0] + [audio_codes.shape[1] for audio_codes in new_audio_out_ids_l[:-1]]), dim=0
            )
            audio_out_ids_start_group_loc = torch.tensor(audio_out_ids_group_loc_l, dtype=torch.long)
        else:
            audio_out_ids = torch.zeros((0, 0), dtype=torch.long)
            audio_out_ids_start = torch.zeros(0, dtype=torch.long)
            if return_labels:
                label_audio_ids = torch.zeros((0, 0), dtype=torch.long)

        reward = torch.tensor(reward_l, dtype=torch.float32)

        # cleaner and faster implementation
        def pad_sequence(seq, length, pad_value):
            if self.pad_left:
                padding = (length - len(seq), 0)
            else:
                padding = (0, length - len(seq))
            return F.pad(seq, padding, value=pad_value)

        input_ids = torch.stack([
            pad_sequence(ele.input_ids, max_seq_length, self.pad_token_id)
            for ele in processed_batch
        ])

        if return_labels:
            label_ids = torch.stack([
                pad_sequence(ele.label_ids, max_seq_length, -100)
                for ele in processed_batch
            ])

        attention_mask = torch.stack([
            pad_sequence(torch.ones_like(ele.input_ids), max_seq_length, 0)
            for ele in processed_batch
        ])

        if not self.return_audio_in_tokens:
            audio_in_ids = None
            audio_in_ids_start = None

        if self.audio_num_codebooks is not None:
            if audio_in_ids is not None:
                audio_in_ids = audio_in_ids[: self.audio_num_codebooks]
            if audio_out_ids is not None:
                audio_out_ids = audio_out_ids[: self.audio_num_codebooks]
            if label_audio_ids is not None:
                label_audio_ids = label_audio_ids[: self.audio_num_codebooks]

        return HiggsAudioBatchInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            audio_out_ids_start_group_loc=audio_out_ids_start_group_loc,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            label_ids=label_ids,
            label_audio_ids=label_audio_ids,
            reward=reward,
        )

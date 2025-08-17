import copy
import random

import torch
import torch.nn as nn
from transformers import CLIPTokenizer
from typing import Any, List, Optional, Union


class TokenizerWrapper:
    """Tokenizer wrapper for CLIPTokenizer. Only support CLIPTokenizer
    currently. This wrapper is modified from https://github.com/huggingface/dif
    fusers/blob/e51f19aee82c8dd874b715a09dbc521d88835d68/src/diffusers/loaders.
    py#L358  # noqa.

    Args:
        from_pretrained (Union[str, os.PathLike], optional): The *model id*
            of a pretrained model or a path to a *directory* containing
            model weights and config. Defaults to None.
        from_config (Union[str, os.PathLike], optional): The *model id*
            of a pretrained model or a path to a *directory* containing
            model weights and config. Defaults to None.

        *args, **kwargs: If `from_pretrained` is passed, *args and **kwargs
            will be passed to `from_pretrained` function. Otherwise, *args
            and **kwargs will be used to initialize the model by
            `self._module_cls(*args, **kwargs)`.
    """

    def __init__(self, tokenizer: CLIPTokenizer):
        self.wrapped = tokenizer
        self.token_map = {}

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return getattr(self, name)
        # if name == "wrapped":
        #    return getattr(self, 'wrapped')#super().__getattr__("wrapped")

        try:
            return getattr(self.wrapped, name)
        except AttributeError:
            raise AttributeError(
                "'name' cannot be found in both "
                f"'{self.__class__.__name__}' and "
                f"'{self.__class__.__name__}.tokenizer'."
            )

    def try_adding_tokens(self, tokens: Union[str, List[str]], *args, **kwargs):
        """Attempt to add tokens to the tokenizer.

        Args:
            tokens (Union[str, List[str]]): The tokens to be added.
        """
        num_added_tokens = self.wrapped.add_tokens(tokens, *args, **kwargs)
        assert num_added_tokens != 0, (
            f"The tokenizer already contains the token {tokens}. Please pass "
            "a different `placeholder_token` that is not already in the "
            "tokenizer."
        )

    def get_token_info(self, token: str) -> dict:
        """Get the information of a token, including its start and end index in
        the current tokenizer.

        Args:
            token (str): The token to be queried.

        Returns:
            dict: The information of the token, including its start and end
                index in current tokenizer.
        """
        token_ids = self.__call__(token).input_ids
        start, end = token_ids[1], token_ids[-2] + 1
        return {"name": token, "start": start, "end": end}

    def add_placeholder_token(self, placeholder_token: str, *args, num_vec_per_token: int = 1, **kwargs):
        """Add placeholder tokens to the tokenizer.

        Args:
            placeholder_token (str): The placeholder token to be added.
            num_vec_per_token (int, optional): The number of vectors of
                the added placeholder token.
            *args, **kwargs: The arguments for `self.wrapped.add_tokens`.
        """
        output = []
        if num_vec_per_token == 1:
            self.try_adding_tokens(placeholder_token, *args, **kwargs)
            output.append(placeholder_token)
        else:
            output = []
            for i in range(num_vec_per_token):
                ith_token = placeholder_token + f"_{i}"
                self.try_adding_tokens(ith_token, *args, **kwargs)
                output.append(ith_token)

        for token in self.token_map:
            if token in placeholder_token:
                raise ValueError(
                    f"The tokenizer already has placeholder token {token} "
                    f"that can get confused with {placeholder_token} "
                    "keep placeholder tokens independent"
                )
        self.token_map[placeholder_token] = output

    def replace_placeholder_tokens_in_text(
            self, text: Union[str, List[str]], vector_shuffle: bool = False, prop_tokens_to_load: float = 1.0
    ) -> Union[str, List[str]]:
        """Replace the keywords in text with placeholder tokens. This function
        will be called in `self.__call__` and `self.encode`.

        Args:
            text (Union[str, List[str]]): The text to be processed.
            vector_shuffle (bool, optional): Whether to shuffle the vectors.
                Defaults to False.
            prop_tokens_to_load (float, optional): The proportion of tokens to
                be loaded. If 1.0, all tokens will be loaded. Defaults to 1.0.

        Returns:
            Union[str, List[str]]: The processed text.
        """
        if isinstance(text, list):
            output = []
            for i in range(len(text)):
                output.append(self.replace_placeholder_tokens_in_text(text[i], vector_shuffle=vector_shuffle))
            return output

        for placeholder_token in self.token_map:
            if placeholder_token in text:
                tokens = self.token_map[placeholder_token]
                tokens = tokens[: 1 + int(len(tokens) * prop_tokens_to_load)]
                if vector_shuffle:
                    tokens = copy.copy(tokens)
                    random.shuffle(tokens)
                text = text.replace(placeholder_token, " ".join(tokens))
        return text

    def replace_text_with_placeholder_tokens(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Replace the placeholder tokens in text with the original keywords.
        This function will be called in `self.decode`.

        Args:
            text (Union[str, List[str]]): The text to be processed.

        Returns:
            Union[str, List[str]]: The processed text.
        """
        if isinstance(text, list):
            output = []
            for i in range(len(text)):
                output.append(self.replace_text_with_placeholder_tokens(text[i]))
            return output

        for placeholder_token, tokens in self.token_map.items():
            merged_tokens = " ".join(tokens)
            if merged_tokens in text:
                text = text.replace(merged_tokens, placeholder_token)
        return text

    def __call__(
            self,
            text: Union[str, List[str]],
            *args,
            vector_shuffle: bool = False,
            prop_tokens_to_load: float = 1.0,
            **kwargs,
    ):
        """The call function of the wrapper.

        Args:
            text (Union[str, List[str]]): The text to be tokenized.
            vector_shuffle (bool, optional): Whether to shuffle the vectors.
                Defaults to False.
            prop_tokens_to_load (float, optional): The proportion of tokens to
                be loaded. If 1.0, all tokens will be loaded. Defaults to 1.0
            *args, **kwargs: The arguments for `self.wrapped.__call__`.
        """
        replaced_text = self.replace_placeholder_tokens_in_text(
            text, vector_shuffle=vector_shuffle, prop_tokens_to_load=prop_tokens_to_load
        )

        return self.wrapped.__call__(replaced_text, *args, **kwargs)

    def encode(self, text: Union[str, List[str]], *args, **kwargs):
        """Encode the passed text to token index.

        Args:
            text (Union[str, List[str]]): The text to be encode.
            *args, **kwargs: The arguments for `self.wrapped.__call__`.
        """
        replaced_text = self.replace_placeholder_tokens_in_text(text)
        return self.wrapped(replaced_text, *args, **kwargs)

    def decode(self, token_ids, return_raw: bool = False, *args, **kwargs) -> Union[str, List[str]]:
        """Decode the token index to text.

        Args:
            token_ids: The token index to be decoded.
            return_raw: Whether keep the placeholder token in the text.
                Defaults to False.
            *args, **kwargs: The arguments for `self.wrapped.decode`.

        Returns:
            Union[str, List[str]]: The decoded text.
        """
        text = self.wrapped.decode(token_ids, *args, **kwargs)
        if return_raw:
            return text
        replaced_text = self.replace_text_with_placeholder_tokens(text)
        return replaced_text

    def __repr__(self):
        """The representation of the wrapper."""
        s = super().__repr__()
        prefix = f"Wrapped Module Class: {self._module_cls}\n"
        prefix += f"Wrapped Module Name: {self._module_name}\n"
        if self._from_pretrained:
            prefix += f"From Pretrained: {self._from_pretrained}\n"
        s = prefix + s
        return s


class EmbeddingLayerWithFixes(nn.Module):
    """The revised embedding layer to support external embeddings. This design
    of this class is inspired by https://github.com/AUTOMATIC1111/stable-
    diffusion-webui/blob/22bcc7be428c94e9408f589966c2040187245d81/modules/sd_hi
    jack.py#L224  # noqa.

    Args:
        wrapped (nn.Emebdding): The embedding layer to be wrapped.
        external_embeddings (Union[dict, List[dict]], optional): The external
            embeddings added to this layer. Defaults to None.
    """

    def __init__(self, wrapped: nn.Embedding, external_embeddings: Optional[Union[dict, List[dict]]] = None):
        super().__init__()
        self.wrapped = wrapped
        self.num_embeddings = wrapped.weight.shape[0]

        self.external_embeddings = []
        if external_embeddings:
            self.add_embeddings(external_embeddings)

        self.trainable_embeddings = nn.ParameterDict()

    @property
    def weight(self):
        """Get the weight of wrapped embedding layer."""
        return self.wrapped.weight

    def check_duplicate_names(self, embeddings: List[dict]):
        """Check whether duplicate names exist in list of 'external
        embeddings'.

        Args:
            embeddings (List[dict]): A list of embedding to be check.
        """
        names = [emb["name"] for emb in embeddings]
        assert len(names) == len(set(names)), (
            "Found duplicated names in 'external_embeddings'. Name list: " f"'{names}'"
        )

    def check_ids_overlap(self, embeddings):
        """Check whether overlap exist in token ids of 'external_embeddings'.

        Args:
            embeddings (List[dict]): A list of embedding to be check.
        """
        ids_range = [[emb["start"], emb["end"], emb["name"]] for emb in embeddings]
        ids_range.sort()  # sort by 'start'
        # check if 'end' has overlapping
        for idx in range(len(ids_range) - 1):
            name1, name2 = ids_range[idx][-1], ids_range[idx + 1][-1]
            assert ids_range[idx][1] <= ids_range[idx + 1][0], (
                f"Found ids overlapping between embeddings '{name1}' " f"and '{name2}'."
            )

    def add_embeddings(self, embeddings: Optional[Union[dict, List[dict]]]):
        """Add external embeddings to this layer.
        Use case:
        Args:
            embeddings (Union[dict, list[dict]]): The external embeddings to
                be added. Each dict must contain the following 4 fields: 'name'
                (the name of this embedding), 'embedding' (the embedding
                tensor), 'start' (the start token id of this embedding), 'end'
                (the end token id of this embedding). For example:
                `{name: NAME, start: START, end: END, embedding: torch.Tensor}`
        """
        if isinstance(embeddings, dict):
            embeddings = [embeddings]

        self.external_embeddings += embeddings
        self.check_duplicate_names(self.external_embeddings)
        self.check_ids_overlap(self.external_embeddings)

        # set for trainable
        added_trainable_emb_info = []
        for embedding in embeddings:
            trainable = embedding.get("trainable", False)
            if trainable:
                name = embedding["name"]
                embedding["embedding"] = torch.nn.Parameter(embedding["embedding"])
                self.trainable_embeddings[name] = embedding["embedding"]
                added_trainable_emb_info.append(name)

        added_emb_info = [emb["name"] for emb in embeddings]
        added_emb_info = ", ".join(added_emb_info)
        print(f"Successfully add external embeddings: {added_emb_info}.", "current")

        if added_trainable_emb_info:
            added_trainable_emb_info = ", ".join(added_trainable_emb_info)
            print("Successfully add trainable external embeddings: " f"{added_trainable_emb_info}", "current")

    def replace_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace external input ids to 0.

        Args:
            input_ids (torch.Tensor): The input ids to be replaced.

        Returns:
            torch.Tensor: The replaced input ids.
        """
        input_ids_fwd = input_ids.clone()
        input_ids_fwd[input_ids_fwd >= self.num_embeddings] = 0
        return input_ids_fwd

    def replace_embeddings(
            self, input_ids: torch.Tensor, embedding: torch.Tensor, external_embedding: dict
    ) -> torch.Tensor:
        """Replace external embedding to the embedding layer. Noted that, in
        this function we use `torch.cat` to avoid inplace modification.

        Args:
            input_ids (torch.Tensor): The original token ids. Shape like
                [LENGTH, ].
            embedding (torch.Tensor): The embedding of token ids after
                `replace_input_ids` function.
            external_embedding (dict): The external embedding to be replaced.

        Returns:
            torch.Tensor: The replaced embedding.
        """
        new_embedding = []

        name = external_embedding["name"]
        start = external_embedding["start"]
        end = external_embedding["end"]
        target_ids_to_replace = [i for i in range(start, end)]
        ext_emb = external_embedding["embedding"].to(embedding.device)

        # do not need to replace
        if not (input_ids == start).any():
            return embedding

        # start replace
        s_idx, e_idx = 0, 0
        while e_idx < len(input_ids):
            if input_ids[e_idx] == start:
                if e_idx != 0:
                    # add embedding do not need to replace
                    new_embedding.append(embedding[s_idx:e_idx])

                # check if the next embedding need to replace is valid
                actually_ids_to_replace = [int(i) for i in input_ids[e_idx: e_idx + end - start]]
                assert actually_ids_to_replace == target_ids_to_replace, (
                    f"Invalid 'input_ids' in position: {s_idx} to {e_idx}. "
                    f"Expect '{target_ids_to_replace}' for embedding "
                    f"'{name}' but found '{actually_ids_to_replace}'."
                )

                new_embedding.append(ext_emb)

                s_idx = e_idx + end - start
                e_idx = s_idx + 1
            else:
                e_idx += 1

        if e_idx == len(input_ids):
            new_embedding.append(embedding[s_idx:e_idx])

        return torch.cat(new_embedding, dim=0)

    def forward(self, input_ids: torch.Tensor, external_embeddings: Optional[List[dict]] = None, out_dtype = None):
        """The forward function.

        Args:
            input_ids (torch.Tensor): The token ids shape like [bz, LENGTH] or
                [LENGTH, ].
            external_embeddings (Optional[List[dict]]): The external
                embeddings. If not passed, only `self.external_embeddings`
                will be used.  Defaults to None.

        input_ids: shape like [bz, LENGTH] or [LENGTH].
        """
        assert input_ids.ndim in [1, 2]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        if external_embeddings is None and not self.external_embeddings:
            return self.wrapped(input_ids, out_dtype=out_dtype)

        input_ids_fwd = self.replace_input_ids(input_ids)
        inputs_embeds = self.wrapped(input_ids_fwd)

        vecs = []

        if external_embeddings is None:
            external_embeddings = []
        elif isinstance(external_embeddings, dict):
            external_embeddings = [external_embeddings]
        embeddings = self.external_embeddings + external_embeddings

        for input_id, embedding in zip(input_ids, inputs_embeds):
            new_embedding = embedding
            for external_embedding in embeddings:
                new_embedding = self.replace_embeddings(input_id, new_embedding, external_embedding)
            vecs.append(new_embedding)

        return torch.stack(vecs).to(out_dtype)


def add_tokens(
        tokenizer, text_encoder, placeholder_tokens: list, initialize_tokens: list = None,
        num_vectors_per_token: int = 1
):
    """Add token for training.

    # TODO: support add tokens as dict, then we can load pretrained tokens.
    """
    if initialize_tokens is not None:
        assert len(initialize_tokens) == len(
            placeholder_tokens
        ), "placeholder_token should be the same length as initialize_token"
    for ii in range(len(placeholder_tokens)):
        tokenizer.add_placeholder_token(placeholder_tokens[ii], num_vec_per_token=num_vectors_per_token)

    # text_encoder.set_embedding_layer()
    embedding_layer = text_encoder.text_model.embeddings.token_embedding
    text_encoder.text_model.embeddings.token_embedding = EmbeddingLayerWithFixes(embedding_layer)
    embedding_layer = text_encoder.text_model.embeddings.token_embedding

    assert embedding_layer is not None, (
        "Do not support get embedding layer for current text encoder. " "Please check your configuration."
    )
    initialize_embedding = []
    if initialize_tokens is not None:
        for ii in range(len(placeholder_tokens)):
            init_id = tokenizer(initialize_tokens[ii]).input_ids[1]
            temp_embedding = embedding_layer.weight[init_id]
            initialize_embedding.append(temp_embedding[None, ...].repeat(num_vectors_per_token, 1))
    else:
        for ii in range(len(placeholder_tokens)):
            init_id = tokenizer("a").input_ids[1]
            temp_embedding = embedding_layer.weight[init_id]
            len_emb = temp_embedding.shape[0]
            init_weight = (torch.rand(num_vectors_per_token, len_emb) - 0.5) / 2.0
            initialize_embedding.append(init_weight)

    # initialize_embedding  = torch.cat(initialize_embedding,dim=0)

    token_info_all = []
    for ii in range(len(placeholder_tokens)):
        token_info = tokenizer.get_token_info(placeholder_tokens[ii])
        token_info["embedding"] = initialize_embedding[ii]
        token_info["trainable"] = True
        token_info_all.append(token_info)
    embedding_layer.add_embeddings(token_info_all)

import argparse
from collections import Counter, defaultdict
import heapq
import json
import logging
import numpy as np
import os
import random
import sys
import torch
from torch import nn
from pts.data.diff_utils import get_edit_keywords

# --- Begin Dummy Vocabulary Fallback ---
try:
    from dpu_utils.mlutils import Vocabulary
except ModuleNotFoundError:
    print("Warning: dpu_utils.mlutils.Vocabulary not available; using dummy Vocabulary.")
    class Vocabulary:
        PAD = "<PAD>"
        UNK = "<UNK>"
        
        @staticmethod
        def get_pad():
            return Vocabulary.PAD

        @staticmethod
        def get_unk():
            return Vocabulary.UNK

        @staticmethod
        def create_vocabulary(tokens, max_size, count_threshold, add_pad):
            vocab = Vocabulary()
            vocab.token2id = {}
            vocab.id2token = {}
            if add_pad:
                vocab.token2id[Vocabulary.PAD] = 0
                vocab.id2token[0] = Vocabulary.PAD
            # Always add UNK
            vocab.token2id[Vocabulary.UNK] = len(vocab.token2id)
            vocab.id2token[len(vocab.token2id)-1] = Vocabulary.UNK
            # Sort tokens by frequency (descending) then alphabetically
            sorted_tokens = sorted(tokens.items(), key=lambda x: (-x[1], x[0]))
            for token, count in sorted_tokens:
                if count < count_threshold:
                    continue
                if len(vocab.token2id) >= max_size:
                    break
                if token in vocab.token2id:
                    continue
                vocab.token2id[token] = len(vocab.token2id)
                vocab.id2token[len(vocab.token2id)-1] = token
            return vocab

        def get_id_or_unk(self, token):
            return self.token2id.get(token, self.token2id.get(Vocabulary.UNK))

        def get_id_or_unk_multiple(self, tokens, pad_to_size, padding_element):
            ids = [self.get_id_or_unk(token) for token in tokens]
            if len(ids) < pad_to_size:
                ids.extend([padding_element] * (pad_to_size - len(ids)))
            else:
                ids = ids[:pad_to_size]
            return ids

        def get_name_for_id(self, idx):
            return self.id2token.get(idx, Vocabulary.UNK)

        def __len__(self):
            return len(self.token2id)

        @property
        def id_to_token(self):
            return self.id2token
# --- End Dummy Vocabulary Fallback ---


START = '<sos>'
END = '<eos>'
NL_EMBEDDING_SIZE = 64
CODE_EMBEDDING_SIZE = 64
MAX_VOCAB_SIZE = 10000
CODE_EMBEDDING_PATH = ""

class CodeEmbeddingStore(nn.Module):
    def __init__(self, code_threshold, code_embedding_size, code_token_counter,
                 dropout_rate, load_pretrained_embeddings=False, static=False):
        """Keeps track of the NL and code vocabularies and embeddings."""
        super(CodeEmbeddingStore, self).__init__()
        self.__code_vocabulary = Vocabulary.create_vocabulary(tokens=code_token_counter,
                                                              max_size=MAX_VOCAB_SIZE,
                                                              count_threshold=code_threshold,
                                                              add_pad=True)
        self.__code_embedding_layer = nn.Embedding(num_embeddings=len(self.__code_vocabulary),
                                                   embedding_dim=code_embedding_size,
                                                   padding_idx=self.__code_vocabulary.get_id_or_unk(
                                                       Vocabulary.get_pad()))
        self.code_embedding_dropout_layer = nn.Dropout(p=dropout_rate)

        print('Code vocabulary size: {}'.format(len(self.__code_vocabulary)))

        self.static = static
        if self.static and not load_pretrained_embeddings:
            raise ValueError(f"A static embedding must be pretrained!")

        if load_pretrained_embeddings:
            self.initialize_embeddings()

    def initialize_embeddings(self):
        with open(CODE_EMBEDDING_PATH) as f:
            code_embeddings = json.load(f)

        code_weights_matrix = np.zeros((len(self.__code_vocabulary), CODE_EMBEDDING_SIZE))
        code_word_count = 0
        # Note: using id2token from our dummy or dpu_utils Vocabulary.
        for i, word in enumerate(self.__code_vocabulary.id_to_token.values()):
            try:
                code_weights_matrix[i] = code_embeddings[word]
                code_word_count += 1
            except KeyError:
                if self.static:
                    code_weights_matrix[i] = code_embeddings.get("%UNK%", np.zeros((CODE_EMBEDDING_SIZE,)))
                else:
                    code_weights_matrix[i] = np.random.normal(scale=0.6, size=(CODE_EMBEDDING_SIZE,))
        self.__code_embedding_layer.weight = torch.nn.Parameter(torch.FloatTensor(code_weights_matrix),
                                                                requires_grad=(not self.static))

        print('Using {} pre-trained code embeddings'.format(code_word_count))

    def get_code_embeddings(self, token_ids):
        return self.code_embedding_dropout_layer(self.__code_embedding_layer(token_ids))

    @property
    def nl_vocabulary(self):
        return self.__nl_vocabulary

    @property
    def code_vocabulary(self):
        return self.__code_vocabulary

    @property
    def nl_embedding_layer(self):
        return self.__nl_embedding_layer

    @property
    def code_embedding_layer(self):
        return self.__code_embedding_layer

    def get_padded_code_ids(self, code_sequence, pad_length):
        return self.__code_vocabulary.get_id_or_unk_multiple(code_sequence,
                                                             pad_to_size=pad_length,
                                                             padding_element=self.__code_vocabulary.get_id_or_unk(
                                                                 Vocabulary.get_pad()),
                                                             )

    def get_padded_nl_ids(self, nl_sequence, pad_length):
        return self.__nl_vocabulary.get_id_or_unk_multiple(nl_sequence,
                                                           pad_to_size=pad_length,
                                                           padding_element=self.__nl_vocabulary.get_id_or_unk(
                                                               Vocabulary.get_pad()),
                                                           )

    def get_extended_padded_nl_ids(self, nl_sequence, pad_length, inp_ids, inp_tokens):
        # Derived from: https://github.com/microsoft/dpu-utils/blob/master/python/dpu_utils/mlutils/vocabulary.py
        nl_ids = []
        for token in nl_sequence:
            nl_id = self.get_nl_id(token)
            if self.is_nl_unk(nl_id) and token in inp_tokens:
                copy_idx = inp_tokens.index(token)
                nl_id = inp_ids[copy_idx]
            nl_ids.append(nl_id)

        if len(nl_ids) > pad_length:
            return nl_ids[:pad_length]
        else:
            padding = [self.__nl_vocabulary.get_id_or_unk(Vocabulary.get_pad())] * (pad_length - len(nl_ids))
            return nl_ids + padding

    def get_code_id(self, token):
        return self.__code_vocabulary.get_id_or_unk(token)

    def is_code_unk(self, id):
        return id == self.__code_vocabulary.get_id_or_unk(Vocabulary.get_unk())

    def get_code_token(self, token_id):
        return self.__code_vocabulary.get_name_for_id(token_id)

    def get_nl_id(self, token):
        return self.__nl_vocabulary.get_id_or_unk(token)

    def is_nl_unk(self, id):
        return id == self.__nl_vocabulary.get_id_or_unk(Vocabulary.get_unk())

    def get_nl_token(self, token_id):
        return self.__nl_vocabulary.get_name_for_id(token_id)

    def get_vocab_extended_nl_token(self, token_id, inp_ids, inp_tokens):
        if token_id < len(self.__nl_vocabulary):
            return self.get_nl_token(token_id)
        elif token_id in inp_ids:
            copy_idx = inp_ids.index(token_id)
            return inp_tokens[copy_idx]
        else:
            return Vocabulary.get_unk()

    def get_nl_tokens(self, token_ids, inp_ids, inp_tokens):
        tokens = [self.get_vocab_extended_nl_token(t, inp_ids, inp_tokens) for t in token_ids]
        if END in tokens:
            return tokens[:tokens.index(END)]
        return tokens

    def get_end_id(self):
        return self.get_nl_id(END)

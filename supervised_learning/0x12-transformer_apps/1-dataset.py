#!/usr/bin/env python3
"""Dataset class module"""


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """Prepares a dataset for translation"""

    def __init__(self):
        """Dataset initializer function"""
        self.data_train = tfds.load("ted_hrlr_translate/pt_to_en",
                                    split="train",
                                    as_supervised=True)
        self.data_valid = tfds.load("ted_hrlr_translate/pt_to_en",
                                    split="validation",
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Create subword tokenizers for dataset"""
        STE = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = STE.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=(2 ** 15))
        tokenizer_en = STE.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=(2 ** 15))

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode a translation into tokens"""
        port_start = self.tokenizer_pt.vocab_size
        port_end = port_start + 1
        eng_start = self.tokenizer_en.vocab_size
        eng_end = eng_start + 1
        pt_tokens = [port_start] + self.tokenizer_pt.encode(
            pt.numpy()) + [port_end]
        en_tokens = [eng_start] + self.tokenizer_en.encode(
            en.numpy()) + [eng_end]

        return pt_tokens, en_tokens

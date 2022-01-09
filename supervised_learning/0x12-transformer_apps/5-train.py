#!/usr/bin/env python3
"""Train module"""


import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Create and train a transformer for Port ot Eng"""
    return

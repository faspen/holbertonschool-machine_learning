#!/usr/bin/env python3
"""Extract Word2Vec"""


from gensim.models import Word2Vec


def gensim_to_keras(model):
    """Convert gensim word2vec model to keras"""
    return model.wv.get_keras_embedding(train_embeddings=True)

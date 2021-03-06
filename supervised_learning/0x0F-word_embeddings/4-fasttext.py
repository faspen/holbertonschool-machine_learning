#!/usr/bin/env python3
"""FastText model task"""


from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Create and train a gensim fasttext model"""
    model = FastText(sentences, size=size, window=window, min_count=min_count,
                     negative=negative, sg=cbow, seed=seed, workers=workers,
                     iter=iterations)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.iter)
    return model

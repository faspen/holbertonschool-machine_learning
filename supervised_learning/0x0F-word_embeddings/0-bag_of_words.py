#!/usr/bin/env python3
"""Bag of Words"""


from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix"""
    vect = CountVectorizer(vocabulary=vocab)
    mat = vect.fit_transform(sentences)

    emb = mat.toarray()
    feats = vect.get_feature_names()

    return emb, feats

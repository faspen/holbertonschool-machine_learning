#!/usr/bin/env python3
"""TF-IDF"""


from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Create TF-IDF embedding"""
    vect = TfidfVectorizer(vocabulary=vocab)
    mat = vect.fit_transform(sentences)

    emb = mat.toarray()
    feats = vect.get_feature_names()

    return emb, feats

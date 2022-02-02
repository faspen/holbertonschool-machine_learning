#!/usr/bin/env python3
"""Semantic Search task"""


import tensorflow_hub as hub
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    """Perform semantic search on corpus of documents"""
    doc = [sentence]
    model = hub.load('https://tfhub.dev/google/universal-sentence' +
                     '-encoder-large/5')

    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
        with open(corpus_path + '/' + filename, 'r', encoding='utf-8') as f:
            doc.append(f.read())

    embs = model(doc)
    comp = np.inner(embs, embs)
    close = np.argmax(comp[0, 1:])
    alike = doc[close + 1]

    return alike

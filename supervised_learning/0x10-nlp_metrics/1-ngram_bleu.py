#!/usr/bin/env python3
"""N-gram bleu score"""


import numpy as np


def helper(references, sentence, n):
    """Ngram bleu helper function"""
    if n == 1:
        return references, sentence

    n_sent = []
    slen = len(sentence)

    for i, word in enumerate(sentence):
        count = 0
        w = word

        for j in range(1, n):
            if slen > i + j:
                w += " " + sentence[i + j]
                count += 1

        if count == j:
            n_sent.append(w)

    n_ref = []

    for ref in references:
        n_tmp = []
        rlen = len(ref)

        for i, word in enumerate(ref):
            count = 0
            w = word

            for j in range(1, n):
                if rlen > i + j:
                    w += " " + ref[i + j]
                    count += 1

            if count == j:
                n_tmp.append(w)
        n_ref.append(n_tmp)

    return n_ref, n_sent


def ngram_bleu(references, sentence, n):
    """Calculate ngram bleu score for a sentence"""
    n_ref, n_sent = helper(references, sentence, n)
    ngs_len = len(n_sent)
    slen = len(sentence)
    sdict = {word: n_sent.count(word) for word in n_sent}
    rdict = {}

    for ref in n_ref:
        for gram in ref:
            if rdict.get(gram) is None or rdict[gram] < ref.count(gram):
                rdict[gram] = ref.count(gram)
    matches = {word: 0 for word in n_sent}

    for ref in n_ref:
        for gram in matches.keys():
            if gram in ref:
                matches[gram] = sdict[gram]
    for gram in matches.keys():
        if rdict.get(gram) is not None:
            matches[gram] = min(rdict[gram], matches[gram])

    acc = sum(matches.values()) / ngs_len
    idx = np.argmin([abs(len(word) - slen) for word in references])
    rlen = len(references[idx])

    if slen > rlen:
        bleu = 1
    else:
        bleu = np.exp(1 - float(rlen) / slen)

    score = bleu * acc

    return score

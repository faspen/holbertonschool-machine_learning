#!/usr/bin/env python3
"""Cumulative ngram bleu score"""


import numpy as np


def helper(references, sentence, n):
    """Cumulative helper function"""
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
        tmp = []
        rlen = len(ref)

        for i, word in enumerate(ref):
            count = 0
            w = word

            for j in range(1, n):
                if rlen > i + j:
                    w += " " + ref[i + j]
                    count += 1
            if count == j:
                tmp.append(w)
        n_ref.append(tmp)

    return n_ref, n_sent


def calc_prec(references, sentence, n):
    """Calculates precision of bleu"""
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
    prec = sum(matches.values()) / ngs_len

    return prec


def cumulative_bleu(references, sentence, n):
    """calculates the cumulative n-gram BLEU score"""
    slen = len(sentence)
    prec = [0] * n

    for i in range(n):
        prec[i] = calc_prec(references, sentence, i + 1)

    avg = np.exp(np.sum((1 / n) * np.log(prec)))
    idx = np.argmin([abs(len(word) - slen) for word in references])
    rlen = len(references[idx])

    if slen > rlen:
        bleau = 1
    else:
        bleu = np.exp(1 - float(rlen) / slen)
    score = bleu * avg

    return score

#!/usr/bin/env python3
"""Unigram BLEU score"""


import numpy as np


def uni_bleu(references, sentence):
    """Calculate unigram bleu score"""
    unique = list(set(sentence))
    wdict = {}
    for r in references:
        for w in r:
            if w in unique:
                if w not in wdict.keys():
                    wdict[w] = r.count(w)
                else:
                    num = r.count(w)
                    prev = wdict[w]
                    wdict[w] = max(num, prev)

    cand = len(sentence)
    prob = sum(wdict.values()) / cand
    match = []

    for r in references:
        rlen = len(r)
        diff = abs(rlen - cand)
        match.append((diff, rlen))

    sorter = sorted(match, key=(lambda x: x[0]))
    winner = sorter[0][1]

    if cand > winner:
        bleu = 1
    else:
        bleu = np.exp(1 - (winner / cand))
    score = bleu * np.exp(np.log(prob))

    return score

#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    tok = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    quest_toks = tok.tokenize(question)
    text_toks = tok.tokenize(reference)
    tokens = ['[CLS]'] + quest_toks + [
        '[SEP]'] + text_toks + ['[SEP]']

    input_id = tok.convert_tokens_to_ids(tokens)
    mask = [1] * len(input_id)
    input_type = [0] * (1 + len(quest_toks) + 1) + [1] * (len(text_toks) + 1)

    input_id, mask, input_type = map(lambda x: tf.expand_dims(
        tf.convert_to_tensor(x, tf.int32), 0), (input_id, mask, input_type))

    output = model([input_id, mask, input_type])
    start = tf.argmax(output[0][0][1:]) + 1
    end = tf.argmax(output[1][0][1:]) + 1
    answer_tok = tokens[start: end + 1]
    answer = tok.convert_tokens_to_string(answer_tok)

    return answer

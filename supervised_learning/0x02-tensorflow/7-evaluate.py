#!/usr/bin/env python3
"""Evaluate the results"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluate neural network"""
    with tf.Session() as trainer:
        storer = tf.train.import_meta_graph(save_path + ".meta")
        storer.restore(trainer, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        y_eval = trainer.run(y_pred, feed_dict={x: X, y: Y})
        acc_eval = trainer.run(accuracy, feed_dict={x: X, y: Y})
        loss_eval = trainer.run(loss, feed_dict={x: X, y: Y})
        return y_eval, acc_eval, loss_eval

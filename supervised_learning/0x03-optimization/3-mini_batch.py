#!/usr/bin/env python3
"""Mini batch"""


import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        batch_size=32,
        epochs=5,
        load_path="/tmp/model.ckpt",
        save_path="/tmp/model.ckpt"):
    """
    Train the mini batch
    """
    with tf.Session() as sess:
        storer = tf.train.import_meta_graph(load_path + ".meta")
        storer.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        mini_batches = X_train.shape[0] // batch_size

        for i in range(epochs + 1):
            np.random.permutation(X_train)
            np.random.permutation(Y_train)
            print("After {} epochs:".format(i))
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            print("\tTraining Cost: {}".format(cost_train))
            acc_train = sess.run(accuracy, feed_dict={
                                 x: X_train, y: Y_train})
            print("\tTraining Accuracy: {}".format(acc_train))
            cost_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            print("\tValidation Cost: {}".format(cost_valid))
            acc_valid = sess.run(accuracy, feed_dict={
                                 x: X_valid, y: Y_valid})
            print("\tValidation Accuracy: {}".format(acc_valid))

            for j in range(mini_batches + 1):
                beg = j * batch_size
                end = (j + 1) * batch_size
                sess.run(train_op, feed_dict={
                         x: X_train[beg:end], y: Y_train[beg:end]})

                if j % 100 == 0 and j != 0 and i != epochs:
                    print('\tStep {}:'.format(j))
                    steps = sess.run(loss, feed_dict={
                        x: X_train[beg:end], y: Y_train[beg:end]})
                    print('\t\tCost: {}'.format(steps))
                    acc_steps = sess.run(accuracy, feed_dict={
                        x: X_train[beg:end], y: Y_train[beg:end]})
                    print('\t\tAccuracy: {}'.format(acc_steps))

        return storer.save(sess, save_path)

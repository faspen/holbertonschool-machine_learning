#!/usr/bin/env python3
"""Put it all together"""


import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    """Placeholders function"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y


def create_layer(prev, n, activation):
    """Function that creates a layer"""
    function = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.Dense(units=n,
                            name='layer',
                            activation=activation,
                            kernel_initializer=function)
    return layer(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward prop function"""
    layer = create_layer(x, layer_sizes[0], activations[0])

    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer


def calculate_accuracy(y, y_pred):
    """validate predictions"""
    validate = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    acc = tf.reduce_mean(tf.cast(validate, tf.float32))
    return acc


def calculate_loss(y, y_pred):
    """Calculate softmax loss"""
    loss = tf.losses.softmax_cross_entropy(
        y, y_pred, reduction=tf.losses.Reduction.MEAN)
    return loss


def shuffle_data(X, Y):
    """Return shuffled matrices"""
    shuffle = np.random.permutation(len(X))
    return X[shuffle], Y[shuffle]


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Return tensorflow adam function"""
    return tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Return decay rate operation"""
    return tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True)


def create_batch_norm_layer(prev, n, activation):
    """Return tensor of output"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    base = tf.layers.Dense(units=n, kernel_initializer=kernel)
    A = base(prev)
    mean, var = tf.nn.moments(A, axes=[0])
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    gamma = tf.Variable(tf.ones([n]), trainable=True)

    batch = tf.nn.batch_normalization(A, mean, var, beta, gamma, 1e-8)
    return activation(batch)


def model(
        Data_train,
        Data_valid,
        layers,
        activations,
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        decay_rate=1,
        batch_size=32,
        epochs=5,
        save_path='/tmp/model.ckpt'):
    """Return path where model was saved"""
    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid

    x, y = create_placeholders(Data_train[0].shape[1],
                               Data_train[1].shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    global_step = tf.Variable(0)
    decay = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, decay, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        i = X_train.shape[0]

        if i % batch_size == 0:
            batch_num = i // batch_size
        else:
            batch_num = i // batch_size + 1

        for j in range(epochs + 1):
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            val_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            val_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(j))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(val_cost))
            print("\tValidation Accuracy: {}".format(val_acc))

            if j < epochs:
                X_shuff, Y_shuff = shuffle_data(X_train, Y_train)

                for a in range(batch_num):
                    beg = a * batch_size
                    end = (a + 1) * batch_size
                    #if end > i:
                     #   end = i
                    X_mini = X_shuff[beg:end]
                    Y_mini = Y_shuff[beg:end]

                    food = {x: X_mini, y: Y_mini}
                    sess.run(train_op, feed_dict=food)

                    if a % 100 == 0 and a != 0 and j != epochs:
                        steps = sess.run(loss, feed_dict=food)
                        acc_steps = sess.run(accuracy, feed_dict=food)
                        print('\tStep {}:'.format(a))
                        print('\t\tCost: {}'.format(steps))
                        print('\t\tAccuracy: {}'.format(acc_steps))

        sess.run(tf.assign(global_step, global_step + 1))

    return saver.save(sess, save_path)

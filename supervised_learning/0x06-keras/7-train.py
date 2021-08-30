#!/usr/bin/env python3
"""Train the model"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """Return the history"""

    def time_decay(epoch):
        """Time decay helper"""
        return alpha / (1 + decay_rate * epoch)

    time = []
    if validation_data:
        if early_stopping:
            early_stopping = K.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience)
            time.append(early_stopping)

        if learning_rate_decay:
            lrs = K.callbacks.LearningRateScheduler(time_decay, verbose=1)
            time.append(lrs)

    history = network.fit(
        x=data,
        y=labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=shuffle, validation_data=validation_data,
        callbacks=time)

    return history

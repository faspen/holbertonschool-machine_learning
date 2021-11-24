#!/usr/bin/env python3
"""Variational Autoencoder"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder"""

    def sampling(args):
        """Sampling function for autoencoder"""
        a1, a2 = args
        sh = keras.backend.shape(a1)[0]
        dims = keras.backend.int_shape(a1)[1]
        norm = keras.backend.random_normal(shape=(sh, dims))
        return a1 + keras.backend.exp(a2 / 2) * norm

    input_encoder = keras.Input(shape=(input_dims, ))
    input_decoder = keras.Input(shape=(latent_dims, ))
    setup = keras.layers.Dense(
        hidden_layers[0],
        activation='relu')(input_encoder)

    for lay in range(1, len(hidden_layers)):
        setup = keras.layers.Dense(
            hidden_layers[lay],
            activation='relu')(setup)

    mean = keras.layers.Dense(latent_dims, activation=None)(setup)
    log = keras.layers.Dense(latent_dims, activation=None)(setup)

    smp = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,))([mean, log])
    encoder = keras.Model(inputs=input_encoder, outputs=[smp, mean, log])

    decode = keras.layers.Dense(
        hidden_layers[-1], activation='relu')(input_decoder)

    for d in range(len(hidden_layers) - 2, -1, -1):
        decode = keras.layers.Dense(
            hidden_layers[d], activation='relu')(decode)

    final = keras.layers.Dense(input_dims, activation='sigmoid')(decode)
    decoder = keras.Model(inputs=input_decoder, outputs=final)

    encoder_output = encoder(input_encoder)[0]
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=input_encoder, outputs=decoder_output)

    def calculate_loss(i, j):
        """Loss helper function"""
        loss = keras.backend.binary_crossentropy(i, j)
        loss = keras.backend.sum(loss, axis=1)
        means = -0.5 * \
            keras.backend.mean(1 + log - keras.backend.square(
                mean) - keras.backend.exp(log), axis=-1)
        return loss + means

    auto.compile(optimizer='adam', loss=calculate_loss)

    return encoder, decoder, auto

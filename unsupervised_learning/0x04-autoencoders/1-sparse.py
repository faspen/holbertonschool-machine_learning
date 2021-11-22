#!/usr/bin/env python3
"""Sparse Autoencoder"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Create a sparse autoencoder"""
    input_encoder = keras.Input(shape=(input_dims, ))
    input_decoder = keras.Input(shape=(latent_dims, ))
    sparse = keras.regularizers.l1(lambtha)
    setup = keras.layers.Dense(
        hidden_layers[0],
        activity_regularizer=sparse,
        activation='relu')(input_encoder)

    for lay in range(1, len(hidden_layers)):
        setup = keras.layers.Dense(
            hidden_layers[lay],
            activity_regularizer=sparse,
            activation='relu')(setup)

    latent = keras.layers.Dense(latent_dims, activation='relu')(setup)
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    de = keras.layers.Dense(hidden_layers[-1],
                            activation='relu')(input_decoder)

    for d in range(len(hidden_layers) - 2, -1, -1):
        de = keras.layers.Dense(hidden_layers[d], activation='relu')(de)

    final = keras.layers.Dense(input_dims, activation='sigmoid')(de)
    decoder = keras.Model(inputs=input_decoder, outputs=final)
    encoder_output = encoder(input_encoder)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=input_encoder, outputs=decoder_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

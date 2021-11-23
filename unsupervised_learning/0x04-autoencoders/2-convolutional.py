#!/usr/bin/env python3
"""Convolutional Autoencoder"""


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Create convolutional autoencoder"""
    input_encoder = keras.Input(shape=input_dims)
    input_decoder = keras.Input(shape=latent_dims)

    setup = keras.layers.Conv2D(
        filters[0],
        kernel_size=(3, 3),
        padding='same',
        activation='relu')(input_encoder)
    setup = keras.layers.MaxPool2D((2, 2), padding='same')(setup)

    for lay in range(1, len(filters)):
        setup = keras.layers.Conv2D(
            filters[lay], kernel_size=(
                3, 3), padding='same', activation='relu')(setup)
        setup = keras.layers.MaxPool2D((2, 2), padding='same')(setup)

    latent = setup
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    decode = keras.layers.Conv2D(filters[-1],
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu')(input_decoder)
    decode = keras.layers.UpSampling2D((2, 2))(decode)

    for d in range(len(filters) - 2, 0, -1):
        decode = keras.layers.Conv2D(filters[d], kernel_size=(
            3, 3), padding='same', activation='relu')(decode)
        decode = keras.layers.UpSampling2D((2, 2))(decode)

    final = keras.layers.Conv2D(filters[0], kernel_size=(
        3, 3), padding='valid', activation='relu')(decode)
    final = keras.layers.UpSampling2D((2, 2))(final)
    final = keras.layers.Conv2D(input_dims[-1],
                                kernel_size=(3, 3),
                                padding='same',
                                activation='sigmoid')(final)
    decoder = keras.Model(inputs=input_decoder, outputs=final)

    encoder_output = encoder(input_encoder)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=input_encoder, outputs=decoder_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
